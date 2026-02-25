"""
Per-window tempo estimation via librosa beat tracking.

We use two independent estimators per window and accept the result when both
agree within AGREEMENT_TOLERANCE.  When they disagree beyond the tolerance we
still return the onset-strength-based estimate (usually more reliable on
short windows) rather than None, because tempo estimation over 10 s is
inherently noisier than pitch estimation.

Windows with fewer than MIN_BEATS detected beats are returned as None.
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import List, Optional, Callable

from .io import AudioWindow

# ── tunables ──────────────────────────────────────────────────────────────────
MIN_BEATS: int              = 4      # minimum beat events to trust the estimate
AGREEMENT_TOLERANCE: float  = 0.08   # 8% — treat the two estimators as agreeing below this
HOP_LENGTH: int             = 512    # librosa hop size for onset detection


def estimate_tempo(window: AudioWindow, start_bpm: float = 120.0) -> Optional[float]:
    """
    Return a BPM estimate for *window*, or None if the window is too short /
    silent to yield a reliable beat count.

    Parameters
    ----------
    start_bpm : float
        Centre of the tempo prior passed to librosa's beat tracker and
        tempogram estimator.  Defaults to 120 BPM (librosa default).
        Pass a file-specific hint (e.g. ``median_src_bpm × duration_ratio``)
        to steer the tracker away from wrong-harmonic snapping on high-BPM
        nightcore tracks.
    """
    y, sr = window.audio, window.sample_rate

    # ── estimator 1: librosa default beat tracker ─────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo_default, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
    )
    # librosa ≥ 0.10 may return a 1-element array; normalise to scalar
    tempo_default = float(np.atleast_1d(tempo_default)[0])

    if len(beat_frames) < MIN_BEATS:
        return None

    # ── estimator 2: onset-strength autocorrelation (tempogram) ──────────────
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH
    )
    tempo_tempogram = float(
        np.atleast_1d(
            librosa.feature.tempo(
                onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
                start_bpm=start_bpm,
            )
        )[0]
    )

    # ── consensus ─────────────────────────────────────────────────────────────
    if tempo_default > 0:
        rel_diff = abs(tempo_default - tempo_tempogram) / tempo_default
        if rel_diff <= AGREEMENT_TOLERANCE:
            return float((tempo_default + tempo_tempogram) / 2.0)

    # Disagreement — return whichever is non-zero, preferring default
    return tempo_default if tempo_default > 0 else (tempo_tempogram if tempo_tempogram > 0 else None)


def batch_estimate_tempo(
    windows: List[AudioWindow],
    log: Optional[Callable[[str], None]] = None,
    start_bpm: float = 120.0,
) -> List[Optional[float]]:
    """
    Estimate tempo for every window in *windows*.

    Parameters
    ----------
    start_bpm : float
        Passed through to :func:`estimate_tempo` as the centre of the tempo
        prior.  Use 120.0 (default) for source files; pass
        ``median_src_bpm × duration_ratio`` for the nightcore file to steer
        librosa away from wrong-harmonic snapping.

    Returns
    -------
    list of float | None, one entry per input window.
    """
    results: List[Optional[float]] = []
    n = len(windows)
    for i, w in enumerate(windows):
        if log:
            log(f"    tempo window {i + 1}/{n}  [{w.start_sec:.1f}–{w.end_sec:.1f} s]")
        results.append(estimate_tempo(w, start_bpm=start_bpm))

    valid = sum(1 for r in results if r is not None)
    if log:
        log(f"    {valid}/{n} windows yielded a confident tempo estimate")

    return results
