"""
Windowed cross-correlation speed estimator.

This module is designed for *verification* of an already-created HQNC against
NCOG — both files are expected to be at nearly the same speed (ratio ≈ 1.0).
It is NOT suitable for the initial HQ vs NCOG estimation where the speed ratio
is large and unknown (1.1–1.4×).

Algorithm
---------
1. Load both files at sr=22050 (mono).
2. Skip the first/last ``skip_edges`` fraction to avoid fades and intros.
3. Sample ``n_windows`` evenly-spaced 3 s reference windows from file A.
4. For each window, search file B within ±``search_range`` of the expected
   position and find the position that maximises normalised dot-product
   (cosine) correlation.
5. Fit a line to the (a_pos, b_pos) correspondence pairs via ``np.polyfit``.
   The slope equals speed_A / speed_B: if HQNC is 1% faster than NCOG, file B
   advances 1% more slowly, so slope = 1.01.
6. Return (slope, median_quality) where quality ∈ [0, 1] is the per-window
   normalised correlation (higher = sharper, more confident match).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import librosa

# ── tunables ──────────────────────────────────────────────────────────────────
XCORR_SR:           int   = 22050   # downsample for speed (sufficient for beat matching)
XCORR_N_WINDOWS:    int   = 20      # reference windows to sample from file A
XCORR_WINDOW_SEC:   float = 3.0     # duration of each reference window (s)
XCORR_SEARCH_RANGE: float = 0.05    # ±5% of file B length searched per window
XCORR_SKIP_EDGES:   float = 0.10    # skip first/last 10% to avoid fades
XCORR_RMS_GATE:     float = 1e-3    # skip silent windows below this RMS

# Quality interpretation thresholds (for display only)
XCORR_QUALITY_GOOD: float = 0.70
XCORR_QUALITY_FAIR: float = 0.40


def estimate_speed_xcorr(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    sr: int = XCORR_SR,
    n_windows: int = XCORR_N_WINDOWS,
    window_sec: float = XCORR_WINDOW_SEC,
    search_range: float = XCORR_SEARCH_RANGE,
    skip_edges: float = XCORR_SKIP_EDGES,
) -> Tuple[float, float]:
    """
    Estimate speed_A / speed_B by windowed cross-correlation.

    Parameters
    ----------
    path_a, path_b : str or Path
        Audio files to compare.  ``path_a`` is the reference (HQNC);
        ``path_b`` is the target (NCOG).
    sr : int
        Sample rate to use for loading (default 22050 Hz).
    n_windows : int
        Number of reference windows sampled from file A.
    window_sec : float
        Duration of each reference window in seconds.
    search_range : float
        Fraction of file B's length to search around the expected position
        (default 0.05 = ±5%).
    skip_edges : float
        Fraction of each file to skip at start and end (default 0.10 = 10%).

    Returns
    -------
    (ratio, quality)
        ratio   : float — estimated speed_A / speed_B (≈ 1.0 for a good match)
        quality : float — median per-window normalised correlation ∈ [0, 1]

    Returns ``(1.0, 0.0)`` when fewer than 3 valid correspondences are found
    (silent files, very short tracks, or completely mismatched content).
    """
    ya, _ = librosa.load(str(path_a), sr=sr, mono=True)
    yb, _ = librosa.load(str(path_b), sr=sr, mono=True)

    # Trim edges from both files
    min_len = min(len(ya), len(yb))
    s = int(min_len * skip_edges)
    e = int(min_len * (1.0 - skip_edges))
    ya = ya[s:e]
    yb = yb[s:e]

    win    = int(window_sec * sr)
    search = int(search_range * len(yb))
    stride = max(1, win // 4)   # search stride = window/4 for 4 candidates per win

    if len(ya) < win or len(yb) < win:
        return 1.0, 0.0

    a_positions = np.linspace(0, len(ya) - win, n_windows).astype(int)
    correspondences: list[tuple[int, int]] = []
    qualities: list[float] = []

    for pa in a_positions:
        wa = ya[pa : pa + win]
        if wa.shape[0] < win:
            continue
        if float(np.sqrt(np.mean(wa ** 2))) < XCORR_RMS_GATE:
            continue   # silent window

        # Expected position in B (assuming ≈1:1 speed)
        expected_pb = int(pa * len(yb) / len(ya))
        lo_b = max(0, expected_pb - search)
        hi_b = min(len(yb) - win, expected_pb + search)
        if lo_b >= hi_b:
            continue

        norm_a = float(np.linalg.norm(wa))
        if norm_a < 1e-10:
            continue

        best_corr: float = -1.0
        best_pb:   int   = expected_pb

        for pb in range(lo_b, hi_b, stride):
            wb = yb[pb : pb + win]
            if wb.shape[0] < win:
                continue
            norm_b = float(np.linalg.norm(wb))
            if norm_b < 1e-10:
                continue
            c = float(np.dot(wa, wb) / (norm_a * norm_b))
            if c > best_corr:
                best_corr = c
                best_pb   = pb

        if best_corr > 0:
            correspondences.append((pa, best_pb))
            qualities.append(best_corr)

    if len(correspondences) < 3:
        return 1.0, 0.0

    a_arr = np.array([c[0] for c in correspondences], dtype=float)
    b_arr = np.array([c[1] for c in correspondences], dtype=float)

    # slope = d(B_pos) / d(A_pos) ≈ speed_A / speed_B
    # (if A is 1% faster it covers content 1% faster, so B_pos grows 1% more
    # slowly relative to A_pos, yielding slope = len(A)/len(B) ≈ 1.01)
    slope = float(np.polyfit(a_arr, b_arr, 1)[0])
    quality = float(np.median(qualities))

    return slope, quality


def quality_label(quality: float) -> str:
    """Return a human-readable label for an xcorr quality score."""
    if quality >= XCORR_QUALITY_GOOD:
        return "good match"
    if quality >= XCORR_QUALITY_FAIR:
        return "moderate match"
    return "poor match — possible content mismatch or heavy lossy artefacts"
