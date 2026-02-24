"""
Ratio estimation, confidence intervals, classification, and Rubber Band params.

Algorithm
---------
Each file independently produces a list of per-window pitch estimates and a
list of per-window tempo estimates.  The two lists need not be the same length
(because windows from different files are never aligned).

For pitch we compute:

  pitch_ratio = median(nightcore_pitches) / median(source_pitches)

and use non-parametric bootstrap resampling to obtain a 95 % confidence
interval on that ratio.

For tempo we use a two-stage approach:

  1. Duration ratio (src_samples / nc_samples) — exact, always available.
     Nightcore is produced by speeding up the entire file, so the file-length
     ratio IS the tempo ratio with sample-accurate precision.

  2. Per-window beat-tracking ratio — noisier but provides an independent
     cross-check.  If it agrees with the duration ratio within
     DURATION_BEAT_AGREE_TOL (15 %), the two are blended (weighted 2:1 in
     favour of duration).  If they disagree, the duration ratio wins.

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
N_BOOTSTRAP: int             = 2000   # bootstrap resamples for CI
CI_LEVEL: float              = 0.95   # confidence level (95 %)
PURE_NC_TOLERANCE: float     = 0.02   # ratio difference below this → same factor
MIN_VALID: int               = 3      # minimum valid estimates to attempt consensus
DURATION_BEAT_AGREE_TOL: float = 0.15 # 15 % — beyond this the beat ratio is discarded


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

    # ── duration / tempo provenance ───────────────────────────────────────────
    duration_ratio: float
    """
    Exact tempo ratio derived from file durations: src_samples / nc_samples.
    This is ground truth for nightcore (the speed-up affects the whole file).
    """

    tempo_method: str
    """
    How tempo_ratio was determined.
      'duration'   — beat-tracking was absent or disagreed; duration ratio used.
      'combined'   — beat-tracking agreed with duration; weighted 2:1 blend.
      'beat_track' — (legacy) beat-tracking only, no duration available.
    """

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

    def __str__(self) -> str:
        ci_t = self.tempo_ci
        ci_p = self.pitch_ci
        rb   = self.rubberband
        return (
            f"Classification  : {self.classification}\n"
            f"Tempo ratio     : {self.tempo_ratio:.6f}"
            f"  95% CI [{ci_t[0]:.6f}, {ci_t[1]:.6f}]"
            f"  method={self.tempo_method}  duration_ratio={self.duration_ratio:.6f}\n"
            f"  beat-track windows: {self.n_source_tempo_windows} src"
            f" / {self.n_nc_tempo_windows} nc\n"
            f"Pitch ratio     : {self.pitch_ratio:.6f}"
            f"  95% CI [{ci_p[0]:.6f}, {ci_p[1]:.6f}]"
            f"  (from {self.n_source_pitch_windows} src"
            f" / {self.n_nc_pitch_windows} nc windows)\n"
            f"Rubber Band     : --time {rb.get('time_ratio', '?'):.6f}"
            f"  --pitch {rb.get('pitch_semitones', '?'):.4f} st\n"
            f"CLI example     : {rb.get('cli_command', '')}"
        )


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


def _reconcile_tempo(
    beat_ratio: float,
    beat_ci: Tuple[float, float],
    duration_ratio: float,
    tol: float = DURATION_BEAT_AGREE_TOL,
) -> Tuple[float, Tuple[float, float], str]:
    """
    Reconcile the per-window beat-tracking ratio with the exact duration ratio.

    Duration ratio is ground truth: the nightcore speed-up shifts every sample
    equally, so src_samples / nc_samples == tempo_ratio exactly.  Beat tracking
    is a noisier secondary estimator.

    Returns (tempo_ratio, tempo_ci, tempo_method).
    """
    rel_diff = abs(beat_ratio - duration_ratio) / max(duration_ratio, 1e-9)

    if rel_diff <= tol:
        # They agree — weighted average (duration counts 2x; it's exact)
        combined = (2.0 * duration_ratio + beat_ratio) / 3.0
        return combined, beat_ci, "combined"
    else:
        # Beat tracking unreliable for this content; use duration as ground truth.
        # CI reflects only sample-count quantization (< 1 sample / 22050 Hz).
        d_ci = (duration_ratio * 0.9995, duration_ratio * 1.0005)
        return duration_ratio, d_ci, "duration"


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

    # Tempo is up but pitch is essentially unchanged → time-stretch without pitch shift.
    # Fix: check pitch_ratio ≈ 1.0, not pitch_ratio < tempo_ratio.
    if tempo_ratio > 1.0 + tol and abs(pitch_ratio - 1.0) <= tol:
        return "time_stretch_only"

    return "ambiguous"


def _rubberband_params(tempo_ratio: float, pitch_ratio: float) -> dict:
    """
    Rubber Band parameters to reconstruct the original FROM the nightcore.

    --time  : make the file longer by tempo_ratio (undo the speed-up)
    --pitch : lower the pitch to undo the net pitch shift
    """
    pitch_st = -12.0 * math.log2(pitch_ratio)
    rb = {
        "time_ratio":      round(tempo_ratio, 6),
        "pitch_semitones": round(pitch_st, 4),
        "cli_command": (
            f"rubberband --time {tempo_ratio:.6f} --pitch {pitch_st:.4f}"
            f" nightcore.flac reconstructed.flac"
        ),
    }
    return rb


# ── public API ────────────────────────────────────────────────────────────────
def build_result(
    src_pitches: List[Optional[float]],
    nc_pitches:  List[Optional[float]],
    src_tempos:  List[Optional[float]],
    nc_tempos:   List[Optional[float]],
    *,
    duration_ratio: float,
) -> AnalysisResult:
    """
    Run the full consensus step and return an :class:`AnalysisResult`.

    Parameters
    ----------
    src_pitches, nc_pitches : per-window median F0 (Hz) or None
    src_tempos,  nc_tempos  : per-window BPM estimate or None
    duration_ratio          : src_samples / nc_samples (exact tempo ground truth)
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

    # ── tempo: duration ratio is primary; beat-tracking is cross-check ────────
    if len(src_t) < MIN_VALID or len(nc_t) < MIN_VALID:
        # Not enough beat windows — fall back to pure duration ratio.
        tempo_ratio  = duration_ratio
        tempo_ci     = (duration_ratio * 0.9995, duration_ratio * 1.0005)
        tempo_method = "duration"
    else:
        beat_ratio, beat_ci = _bootstrap_ratio(nc_t, src_t)
        tempo_ratio, tempo_ci, tempo_method = _reconcile_tempo(
            beat_ratio, beat_ci, duration_ratio
        )

    pitch_ratio, pitch_ci = _bootstrap_ratio(nc_p, src_p)

    classification = _classify(tempo_ratio, pitch_ratio, tempo_ci, pitch_ci)
    rubberband     = _rubberband_params(tempo_ratio, pitch_ratio)

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
        duration_ratio=duration_ratio,
        tempo_method=tempo_method,
        rubberband=rubberband,
    )
