"""
Ratio estimation, confidence intervals, classification, and Rubber Band params.

Algorithm
---------
Each file independently produces a list of per-window pitch estimates and a
list of per-window tempo estimates.  The two lists need not be the same length
(because windows from different files are never aligned).

For each quantity (pitch, tempo) we compute:

  ratio = median(nightcore_values) / median(source_values)

and use non-parametric bootstrap resampling to obtain a 95 % confidence
interval on that ratio.

Classification
--------------
  pure_nightcore          pitch_ratio ≈ tempo_ratio
                          (simple playback speedup — the classic case)

  independent_pitch_shift pitch_ratio is meaningfully larger than tempo_ratio
                          (additional upward pitch processing on top of speed-up)

  time_stretch_only       tempo_ratio is meaningfully > 1 but pitch is unchanged
                          (Rubber-Band-style lengthening without pitch change)

  ambiguous               the CIs overlap in a way that prevents a clear call

Rubber Band reconstruction parameters
--------------------------------------
To reconstruct the original from the nightcore track using Rubber Band:

  --time  tempo_ratio          (stretch nightcore duration back to original)
  --pitch pitch_correction_st  (semitones to undo any net pitch shift)

Where:
  pitch_correction_st = −12 · log₂(pitch_ratio)

For a pure nightcore the pitch_ratio equals the tempo_ratio, so the pitch
correction exactly cancels the pitch shift introduced by the speed change.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── tunables ──────────────────────────────────────────────────────────────────
N_BOOTSTRAP: int        = 2000   # bootstrap resamples for CI
CI_LEVEL: float         = 0.95   # confidence level (95 %)
PURE_NC_TOLERANCE: float = 0.02  # ratio difference below this → same factor
MIN_VALID: int           = 3     # minimum valid estimates to attempt consensus

# ── sanity-check thresholds ───────────────────────────────────────────────────
NIGHTCORE_RATIO_MIN: float             = 1.05   # tempo ratio below this → suspicious
NIGHTCORE_RATIO_MAX: float             = 1.50   # tempo ratio above this → unusual
NEAR_UNITY_TOLERANCE: float            = 0.05   # |ratio − 1| < this → same-speed warning
WIDE_CI_RELATIVE: float                = 2.0    # (hi − lo) / point > this → pitch CI unreliable
DURATION_TEMPO_MISMATCH_TOLERANCE: float = 0.08 # |dur_ratio − tempo_ratio| / tempo_ratio > this → wrong version


# ── result container ──────────────────────────────────────────────────────────
@dataclass
class AnalysisResult:
    """Full output of the windowed consensus pipeline."""

    # ── core ratios ───────────────────────────────────────────────────────────
    tempo_ratio: float
    """Nightcore tempo ÷ source tempo.  >1 means the nightcore is faster."""

    pitch_ratio: float
    """Nightcore median pitch ÷ source median pitch.  >1 means pitch is higher."""

    # ── confidence intervals ──────────────────────────────────────────────────
    tempo_ci: Tuple[float, float]
    """Bootstrap 95 % CI for the tempo ratio (lower, upper)."""

    pitch_ci: Tuple[float, float]
    """Bootstrap 95 % CI for the pitch ratio (lower, upper)."""

    # ── classification ────────────────────────────────────────────────────────
    classification: str
    """
    One of:
      pure_nightcore | independent_pitch_shift | time_stretch_only | ambiguous
    """

    # ── window counts ─────────────────────────────────────────────────────────
    n_source_pitch_windows: int
    n_nc_pitch_windows: int
    n_source_tempo_windows: int
    n_nc_tempo_windows: int

    # ── reconstruction parameters ─────────────────────────────────────────────
    rubberband: dict = field(default_factory=dict)
    """
    Parameters to pass to Rubber Band to reconstruct the original.

    Keys
    ----
    time_ratio      : float  — ``rubberband --time`` value (> 1 lengthens)
    pitch_semitones : float  — ``rubberband --pitch`` value (negative = lower)
    cli_command     : str    — example rubberband command string

    For pyrubberband:
      pyrubberband.time_stretch(y, sr, rate=time_ratio,
                                rbargs={'--pitch': str(pitch_semitones)})
    """

    # ── raw per-window data (optional, populated by pipeline) ─────────────────
    # These are the raw per-window estimates before consensus, used by the GUI
    # histogram widget.  None means the pipeline did not store them (e.g. when
    # called from the CLI with no GUI attached).
    src_pitches_raw: Optional[List[Optional[float]]] = None
    nc_pitches_raw:  Optional[List[Optional[float]]] = None
    src_tempos_raw:  Optional[List[Optional[float]]] = None
    nc_tempos_raw:   Optional[List[Optional[float]]] = None

    # ── file durations (seconds) ──────────────────────────────────────────────
    nc_duration:  Optional[float] = None
    src_duration: Optional[float] = None

    # ── median BPMs from librosa (for debugging) ──────────────────────────────
    nc_median_bpm:  Optional[float] = None
    src_median_bpm: Optional[float] = None

    # ── sanity warnings (populated by build_result) ───────────────────────────
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        ci_t = self.tempo_ci
        ci_p = self.pitch_ci
        rb   = self.rubberband
        lines: List[str] = []

        # Warnings — printed first so they're impossible to miss
        for w in self.warnings:
            lines.append(f"WARNING  : {w}")
        if self.warnings:
            lines.append("")

        lines.append(f"Classification  : {self.classification}")

        # Duration ratio — shown alongside tempo ratio for easy cross-check
        dur_note = ""
        if self.nc_duration and self.src_duration:
            dur_ratio = self.src_duration / self.nc_duration
            dur_note = (
                f"  |  duration ratio {dur_ratio:.6f}×"
                f" ({self.src_duration:.1f} s / {self.nc_duration:.1f} s)"
            )

        lines.append(
            f"Tempo ratio     : {self.tempo_ratio:.6f}"
            f"  95% CI [{ci_t[0]:.6f}, {ci_t[1]:.6f}]"
            f"  (from {self.n_source_tempo_windows} src / {self.n_nc_tempo_windows} nc windows)"
            + dur_note
        )
        lines.append(
            f"Pitch ratio     : {self.pitch_ratio:.6f}"
            f"  95% CI [{ci_p[0]:.6f}, {ci_p[1]:.6f}]"
            f"  (from {self.n_source_pitch_windows} src / {self.n_nc_pitch_windows} nc windows)"
        )

        # Plain-English speed summary
        tr = self.tempo_ratio
        if tr > 0:
            inv = 1.0 / tr
            lines.append("")
            lines.append(f"Speed summary   : nightcore is {tr:.4f}× the source speed")
            lines.append(f"                  to hear original tempo → play nightcore at {inv:.4f}× speed")
            lines.append(f"                  (source was sped up by {tr:.4f}× to create the nightcore)")

        # Median BPMs — useful for debugging librosa half-time / quantisation issues
        if self.nc_median_bpm is not None and self.src_median_bpm is not None:
            lines.append(
                f"Median BPMs     : nightcore {self.nc_median_bpm:.2f}  |"
                f"  source {self.src_median_bpm:.2f}"
                f"  (raw detected; ratio = {self.nc_median_bpm / self.src_median_bpm:.6f})"
            )

        lines.append("")
        lines.append(
            f"Rubber Band     : --time {rb.get('time_ratio', '?'):.6f}"
            f"  --pitch {rb.get('pitch_semitones', '?'):.4f} st"
            "  (beat-detected ratio)"
        )
        lines.append(f"CLI (detected)  : {rb.get('cli_command', '')}")

        # Duration-based alternative — shown when available; prefer when CI is degenerate
        if rb.get("duration_time_ratio"):
            lines.append(
                f"Duration-based  : --time {rb['duration_time_ratio']:.6f}"
                f"  --pitch {rb['duration_pitch_semitones']:.4f} st"
                "  (uses file-length ratio — prefer this when CI is degenerate)"
            )
            lines.append(f"CLI (duration)  : {rb.get('duration_cli_command', '')}")

        return "\n".join(lines)


# ── internals ─────────────────────────────────────────────────────────────────
def _valid(values: List[Optional[float]]) -> np.ndarray:
    """Filter out None / NaN / non-positive entries and return as ndarray."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v) and v > 0],
                   dtype=np.float64)
    return arr


def _bootstrap_ratio(
    nc_vals: np.ndarray,
    src_vals: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute median(nc) / median(src) and a bootstrap CI for that ratio.

    Both arrays are resampled independently in each bootstrap iteration,
    since the windows from the two files are not aligned.
    """
    rng = np.random.default_rng(seed=42)
    point = float(np.median(nc_vals) / np.median(src_vals))

    boot = np.empty(n_boot)
    for i in range(n_boot):
        nc_s  = rng.choice(nc_vals,  size=len(nc_vals),  replace=True)
        src_s = rng.choice(src_vals, size=len(src_vals), replace=True)
        boot[i] = np.median(nc_s) / np.median(src_s)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot, alpha * 100))
    hi = float(np.percentile(boot, (1.0 - alpha) * 100))
    return point, (lo, hi)


def _classify(
    tempo_ratio: float,
    pitch_ratio: float,
    tempo_ci: Tuple[float, float],
    pitch_ci: Tuple[float, float],
    tol: float = PURE_NC_TOLERANCE,
) -> str:
    ratio_diff = pitch_ratio - tempo_ratio

    # CIs overlap — the two ratios are statistically indistinguishable
    ci_overlap = tempo_ci[0] <= pitch_ci[1] and pitch_ci[0] <= tempo_ci[1]

    if abs(ratio_diff) <= tol or (ci_overlap and abs(ratio_diff) <= 2 * tol):
        return "pure_nightcore"

    if ratio_diff > tol:
        return "independent_pitch_shift"

    if tempo_ratio > 1.0 + tol and ratio_diff < -tol:
        return "time_stretch_only"

    return "ambiguous"


def _rubberband_params(
    tempo_ratio: float,
    pitch_ratio: float,
    nc_duration: Optional[float] = None,
    src_duration: Optional[float] = None,
) -> dict:
    """
    Rubber Band parameters to reconstruct the original FROM the nightcore.

    --time  : make the file longer by tempo_ratio (undo the speed-up)
    --pitch : lower the pitch to undo the net pitch shift

    When durations are provided, also computes duration-based parameters
    (using src/nc length ratio directly) as a more reliable fallback when
    the beat-detected tempo ratio has a degenerate CI.
    """
    pitch_st = -12.0 * math.log2(pitch_ratio)
    nc_to_source_speed = round(1.0 / tempo_ratio, 6) if tempo_ratio != 0 else None
    rb = {
        "time_ratio":         round(tempo_ratio, 6),
        "pitch_semitones":    round(pitch_st, 4),
        # Playback speed to set in a media player to hear nightcore at original tempo:
        #   e.g. in VLC: Playback → Speed → set to this value
        "nc_to_source_speed": nc_to_source_speed,
        "cli_command": (
            f"rubberband --time {tempo_ratio:.6f} --pitch {pitch_st:.4f}"
            f" nightcore.flac reconstructed.flac"
        ),
    }

    # Duration-based alternative: for a pure speed-up, duration_ratio = true tempo ratio.
    # Pitch ratio also equals the tempo ratio for a pure speed-up (linked by physics).
    if nc_duration and src_duration and nc_duration > 0:
        dur_ratio = src_duration / nc_duration
        dur_pitch_st = -12.0 * math.log2(dur_ratio)
        rb["duration_time_ratio"]       = round(dur_ratio, 6)
        rb["duration_pitch_semitones"]  = round(dur_pitch_st, 4)
        rb["duration_cli_command"]      = (
            f"rubberband --time {dur_ratio:.6f} --pitch {dur_pitch_st:.4f}"
            f" nightcore.flac reconstructed.flac"
        )

    return rb


def _check_sanity(
    tempo_ratio: float,
    pitch_ratio: float,
    tempo_ci: Tuple[float, float],
    pitch_ci: Tuple[float, float],
    nc_duration: Optional[float] = None,
    src_duration: Optional[float] = None,
    tempo_was_corrected: bool = False,
) -> List[str]:
    """
    Return a list of human-readable warning strings for suspicious results.

    Checks performed
    ----------------
    1. Beat-tracker half-time artefact (auto-corrected if durations supplied)
    2. Both files the same duration → probably same-version input mistake
    3. Tempo ratio outside the normal nightcore range (1.05 – 1.50)
       when no duration correction was possible
    4. Duration ratio vs tempo ratio mismatch → files are different versions
    5. Degenerate tempo CI (lo == hi) → librosa BPM quantisation artefact
    6. Pitch CI too wide → CREPE could not track pitch reliably
    """
    warnings: List[str] = []

    if tempo_was_corrected:
        # Already flipped in build_result; explain what happened
        warnings.append(
            f"Beat-tracker half-time artefact corrected: librosa returned a raw tempo "
            f"ratio < 1 (nightcore beat-detected at half-time), but the nightcore file "
            f"({nc_duration:.1f} s) is shorter than the source ({src_duration:.1f} s), "
            "confirming the nightcore IS faster. The ratio has been inverted "
            f"to {tempo_ratio:.4f}× automatically. This is a known librosa artefact "
            "for high-BPM music (>~130 BPM)."
        )
    elif nc_duration is not None and src_duration is not None:
        dur_ratio = nc_duration / src_duration   # < 1 means nc is shorter (faster)

        # Same-duration → likely same version (nightcore-vs-nightcore etc.)
        if abs(dur_ratio - 1.0) < NEAR_UNITY_TOLERANCE:
            warnings.append(
                f"Both files are nearly the same duration "
                f"({nc_duration:.1f} s vs {src_duration:.1f} s). "
                "Did you accidentally provide two nightcore files, or two originals? "
                "A real nightcore should be ~10–35 % shorter than the source."
            )
    else:
        # No duration info — ratio-only fallback checks
        if abs(tempo_ratio - 1.0) < NEAR_UNITY_TOLERANCE:
            warnings.append(
                f"Tempo ratio is {tempo_ratio:.4f} — both files appear to be at the "
                "same speed. Did you accidentally provide two nightcore files, or two "
                "originals? A real nightcore should be 1.05–1.50× faster than the source."
            )
        elif tempo_ratio < 1.0:
            inv = round(1.0 / tempo_ratio, 4)
            warnings.append(
                f"Tempo ratio is {tempo_ratio:.4f} < 1.0. Two possible causes: "
                "(1) librosa half-time detection artefact — the true ratio may be "
                f"{inv:.4f}× (the inverse); or (2) the files are in the wrong order. "
                "Re-run with the correct original FLAC as --source to disambiguate."
            )
        elif tempo_ratio > NIGHTCORE_RATIO_MAX:
            warnings.append(
                f"Tempo ratio is {tempo_ratio:.4f}, above the typical nightcore range "
                f"({NIGHTCORE_RATIO_MIN}–{NIGHTCORE_RATIO_MAX}×). Verify the input files."
            )

    # Duration ratio vs tempo ratio mismatch → different song versions
    if nc_duration is not None and src_duration is not None:
        dur_speed_ratio = src_duration / nc_duration   # equivalent to tempo_ratio for pure speed-up
        discrepancy = abs(dur_speed_ratio - tempo_ratio) / tempo_ratio
        if discrepancy > DURATION_TEMPO_MISMATCH_TOLERANCE:
            warnings.append(
                f"Duration ratio ({dur_speed_ratio:.4f}×) and detected tempo ratio "
                f"({tempo_ratio:.4f}×) differ by {discrepancy * 100:.1f}%. For a pure "
                "speed-up these should be nearly equal. Most likely cause: the two files "
                "are different edits or versions of the same song (e.g. radio edit vs. "
                "extended mix). Find the exact version used to create the nightcore, or "
                f"use the duration ratio ({dur_speed_ratio:.4f}×) directly as the "
                "rubberband --time factor."
            )

    # Degenerate tempo CI (lo == hi) → librosa BPM quantisation locked every window
    if abs(tempo_ci[1] - tempo_ci[0]) < 0.001:
        dur_hint = (
            f" Use the 'Duration-based' CLI command instead of 'CLI (detected)'."
            if (nc_duration is not None and src_duration is not None) else
            ""
        )
        warnings.append(
            f"Tempo CI is degenerate [lo = hi = {tempo_ci[0]:.6f}]: every analysis "
            "window returned the same BPM from librosa. This is a quantisation "
            "artefact — the beat tracker snapped all windows to the same grid BPM. "
            "The beat-detected ratio is unreliable; the duration ratio is more "
            "trustworthy for this track." + dur_hint
        )

    # Wide pitch CI → CREPE could not reliably track F0
    if pitch_ratio > 0:
        ci_span = pitch_ci[1] - pitch_ci[0]
        if ci_span > WIDE_CI_RELATIVE * pitch_ratio:
            warnings.append(
                f"Pitch CI is very wide ({pitch_ci[0]:.3f}–{pitch_ci[1]:.3f}) relative "
                f"to the point estimate ({pitch_ratio:.4f}). CREPE could not reliably "
                "track pitch — this is common with polyphonic or heavily processed audio. "
                "Trust the tempo ratio; treat the pitch ratio and classification as "
                "approximate."
            )

    return warnings


# ── public API ────────────────────────────────────────────────────────────────
def build_result(
    src_pitches: List[Optional[float]],
    nc_pitches:  List[Optional[float]],
    src_tempos:  List[Optional[float]],
    nc_tempos:   List[Optional[float]],
    *,
    nc_duration:  Optional[float] = None,
    src_duration: Optional[float] = None,
) -> AnalysisResult:
    """
    Run the full consensus step and return an :class:`AnalysisResult`.

    Parameters
    ----------
    src_pitches, nc_pitches : per-window median F0 (Hz) or None
    src_tempos,  nc_tempos  : per-window BPM estimate or None
    nc_duration, src_duration : total audio duration in seconds (optional).
        When provided, the pipeline cross-checks the tempo ratio against the
        file-length ratio to detect and correct librosa half-time artefacts.
    """
    src_p = _valid(src_pitches)
    nc_p  = _valid(nc_pitches)
    src_t = _valid(src_tempos)
    nc_t  = _valid(nc_tempos)

    if len(src_p) < MIN_VALID or len(nc_p) < MIN_VALID:
        raise ValueError(
            f"Insufficient valid pitch windows (source: {len(src_p)}, "
            f"nightcore: {len(nc_p)}).  Need ≥ {MIN_VALID} each.\n"
            "  Try: reducing --window size, lowering --energy-gate, "
            "or checking that both files contain music."
        )
    if len(src_t) < MIN_VALID or len(nc_t) < MIN_VALID:
        raise ValueError(
            f"Insufficient valid tempo windows (source: {len(src_t)}, "
            f"nightcore: {len(nc_t)}).  Need ≥ {MIN_VALID} each."
        )

    pitch_ratio, pitch_ci = _bootstrap_ratio(nc_p, src_p)
    tempo_ratio, tempo_ci = _bootstrap_ratio(nc_t, src_t)

    # ── duration cross-check: detect and correct half-time detection artefact ──
    # Librosa's beat tracker occasionally returns half-time BPMs for fast tracks
    # (>~130 BPM), making the ratio appear inverted.  When the nightcore is
    # measurably shorter than the source (i.e. it IS faster) but tempo_ratio < 1,
    # the evidence strongly indicates a half-time artefact: flip both the point
    # estimate and the CI.
    tempo_was_corrected = False
    if (nc_duration is not None and src_duration is not None
            and nc_duration < src_duration * 0.99   # nightcore measurably shorter
            and tempo_ratio < 1.0):                 # but detected as slower
        tempo_ratio = 1.0 / tempo_ratio
        lo, hi = tempo_ci
        tempo_ci = (1.0 / hi, 1.0 / lo)            # invert; bounds swap to stay ordered
        tempo_was_corrected = True

    nc_median_bpm  = float(np.median(nc_t))  if len(nc_t)  > 0 else None
    src_median_bpm = float(np.median(src_t)) if len(src_t) > 0 else None

    classification = _classify(tempo_ratio, pitch_ratio, tempo_ci, pitch_ci)
    rubberband     = _rubberband_params(tempo_ratio, pitch_ratio, nc_duration, src_duration)
    warnings       = _check_sanity(
        tempo_ratio, pitch_ratio, tempo_ci, pitch_ci,
        nc_duration, src_duration, tempo_was_corrected,
    )

    return AnalysisResult(
        tempo_ratio=tempo_ratio,
        pitch_ratio=pitch_ratio,
        tempo_ci=tempo_ci,
        pitch_ci=pitch_ci,
        classification=classification,
        n_source_pitch_windows=len(src_p),
        n_nc_pitch_windows=len(nc_p),
        n_source_tempo_windows=len(src_t),
        n_nc_tempo_windows=len(nc_t),
        rubberband=rubberband,
        nc_duration=nc_duration,
        src_duration=src_duration,
        nc_median_bpm=nc_median_bpm,
        src_median_bpm=src_median_bpm,
        warnings=warnings,
        src_pitches_raw=list(src_pitches),
        nc_pitches_raw=list(nc_pitches),
        src_tempos_raw=list(src_tempos),
        nc_tempos_raw=list(nc_tempos),
    )
