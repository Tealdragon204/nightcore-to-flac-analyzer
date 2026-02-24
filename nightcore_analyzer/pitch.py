"""
Per-window fundamental frequency estimation via CREPE.

CREPE is a deep neural network trained on monophonic audio.  It returns a
per-frame time series of (frequency, confidence) values.  We keep only frames
above MIN_CONFIDENCE and return the median frequency of the surviving frames.

Windows with fewer than MIN_CONFIDENT_FRAMES surviving frames are returned as
None — they will be dropped during consensus.
"""

from __future__ import annotations

import numpy as np
import crepe
from typing import List, Optional, Callable

from .io import AudioWindow

# ── tunables ──────────────────────────────────────────────────────────────────
MIN_CONFIDENCE: float    = 0.80   # CREPE per-frame confidence threshold
MIN_CONFIDENT_FRAMES: int = 10    # minimum frames that must survive filtering
CREPE_MODEL: str          = "full" # tiny | small | medium | large | full
CREPE_STEP_SIZE: int      = 20    # ms between CREPE frames (default 10 ms; 20 ms halves cost)


def estimate_pitch(window: AudioWindow) -> Optional[float]:
    """
    Return the median fundamental frequency (Hz) of *window*, or None if the
    window has too few high-confidence frames.
    """
    # CREPE resamples internally to 16 kHz; passing our 22050 Hz audio is fine.
    _, frequency, confidence, _ = crepe.predict(
        window.audio,
        window.sample_rate,
        model_capacity=CREPE_MODEL,
        step_size=CREPE_STEP_SIZE,
        viterbi=True,   # Viterbi decoding for smoother, more accurate F0 track
        verbose=0,
    )

    mask = confidence >= MIN_CONFIDENCE
    if mask.sum() < MIN_CONFIDENT_FRAMES:
        return None

    return float(np.median(frequency[mask]))


def batch_estimate_pitch(
    windows: List[AudioWindow],
    log: Optional[Callable[[str], None]] = None,
) -> List[Optional[float]]:
    """
    Estimate pitch for every window in *windows*.

    Parameters
    ----------
    windows : list of AudioWindow
    log     : optional callable for progress messages, e.g. ``print``

    Returns
    -------
    list of float | None, one entry per input window.
    """
    results: List[Optional[float]] = []
    n = len(windows)
    for i, w in enumerate(windows):
        if log:
            log(f"    pitch window {i + 1}/{n}  [{w.start_sec:.1f}–{w.end_sec:.1f} s]")
        results.append(estimate_pitch(w))

    valid = sum(1 for r in results if r is not None)
    if log:
        log(f"    {valid}/{n} windows yielded a confident pitch estimate")

    return results
