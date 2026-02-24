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

    def __str__(self) -> str:
        ci_t = self.tempo_ci
        ci_p = self.pitch_ci
        rb   = self.rubberband
        return (
            f"Classification  : {self.classification}\n"
            f"Tempo ratio     : {self.tempo_ratio:.6f}  "
            f"  95% CI [{ci_t[0]:.6f}, {ci_t[1]:.6f}]"
            f"  (from {self.n_source_tempo_windows} src / {self.n_nc_tempo_windows} nc windows)\n"
            f"Pitch ratio     : {self.pitch_ratio:.6f}  "
            f"  95% CI [{ci_p[0]:.6f}, {ci_p[1]:.6f}]"
            f"  (from {self.n_source_pitch_windows} src / {self.n_nc_pitch_windows} nc windows)\n"
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
) -> AnalysisResult:
    """
    Run the full consensus step and return an :class:`AnalysisResult`.

    Parameters
    ----------
    src_pitches, nc_pitches : per-window median F0 (Hz) or None
    src_tempos,  nc_tempos  : per-window BPM estimate or None
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
        rubberband=rubberband,
        src_pitches_raw=list(src_pitches),
        nc_pitches_raw=list(nc_pitches),
        src_tempos_raw=list(src_tempos),
        nc_tempos_raw=list(nc_tempos),
    )
