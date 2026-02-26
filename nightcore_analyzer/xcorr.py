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

# ── content-alignment tunables ────────────────────────────────────────────────
ALIGN_SR:          int   = 11025   # downsample rate for envelope computation
ALIGN_HOP:         int   =   512   # hop ≈ 0.046 s at 11025 Hz
ALIGN_SPEED_LO:    float =  1.03   # minimum nightcore speed to search
ALIGN_SPEED_HI:    float =  1.50   # maximum nightcore speed to search
ALIGN_N_SPEEDS:    int   =    30   # candidate speed steps
ALIGN_MAX_OFFSET:  float = 120.0   # never trim more than 2 min from src
ALIGN_MIN_OFFSET:  float =   1.0   # ignore detected offsets below this (s)


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


def find_content_offset(
    src_audio: np.ndarray,
    nc_audio:  np.ndarray,
    sr: int,
    *,
    speed_lo: float = ALIGN_SPEED_LO,
    speed_hi: float = ALIGN_SPEED_HI,
    n_speeds: int   = ALIGN_N_SPEEDS,
    max_offset_sec: float = ALIGN_MAX_OFFSET,
) -> Tuple[float, float]:
    """
    Detect how many seconds of *src_audio* precede the content that matches
    the start of *nc_audio* (i.e. a musical intro present in src but not nc).

    Uses a coarse RMS energy envelope cross-correlation over a grid of
    candidate nightcore speeds.  For each speed the NCOG envelope is stretched
    to the HQ time scale (divided by speed) and cross-correlated against the
    HQ envelope within the first *max_offset_sec* seconds.  The (speed, lag)
    pair with the highest normalised peak is returned.

    Parameters
    ----------
    src_audio : np.ndarray
        Mono float32 source audio (HQ or HQNC, may have a musical intro).
    nc_audio : np.ndarray
        Mono float32 nightcore audio (NCOG, silence already stripped).
    sr : int
        Sample rate shared by both arrays (from ``load_audio``).
    speed_lo, speed_hi, n_speeds
        Grid of candidate nightcore speed ratios to search.
    max_offset_sec : float
        Cap on the intro length searched (default 120 s).

    Returns
    -------
    (offset_sec, speed_est) : Tuple[float, float]
        *offset_sec* — seconds to skip at the start of src_audio.
        *speed_est*  — rough speed estimate from the alignment search.
        Returns ``(0.0, 1.0)`` when the arrays are too short to analyse.
    """
    # Downsample both to a coarse rate — envelopes need no HF content
    src_ds = librosa.resample(src_audio, orig_sr=sr, target_sr=ALIGN_SR)
    nc_ds  = librosa.resample(nc_audio,  orig_sr=sr, target_sr=ALIGN_SR)

    # RMS energy envelopes (1-D)
    src_env = librosa.feature.rms(y=src_ds, hop_length=ALIGN_HOP)[0].astype(np.float64)
    nc_env  = librosa.feature.rms(y=nc_ds,  hop_length=ALIGN_HOP)[0].astype(np.float64)

    hop_sec           = ALIGN_HOP / ALIGN_SR          # ≈ 0.046 s
    max_offset_frames = int(max_offset_sec / hop_sec)

    best_score  = -1.0
    best_offset = 0.0
    best_speed  = (speed_lo + speed_hi) / 2.0

    for speed in np.linspace(speed_lo, speed_hi, n_speeds):
        # Stretch nc envelope to src time scale (slow it down by 1/speed)
        n_stretched = int(len(nc_env) / speed)
        if n_stretched < 4:
            continue
        if n_stretched >= len(src_env):
            continue  # stretched nc ≥ src — no room for an intro offset

        # Linear-interpolation resample (numpy-only; coarse envelope is fine)
        x_orig   = np.linspace(0.0, 1.0, len(nc_env))
        x_new    = np.linspace(0.0, 1.0, n_stretched)
        stretched = np.interp(x_new, x_orig, nc_env)

        # Search only the first max_offset_frames lags
        search_len = min(max_offset_frames, len(src_env) - n_stretched)
        if search_len <= 0:
            continue

        src_search = src_env[:search_len + n_stretched]
        corr = np.correlate(src_search, stretched, mode='valid')
        corr = corr[:search_len + 1]

        if len(corr) == 0:
            continue

        peak_idx = int(np.argmax(corr))
        peak_val = float(corr[peak_idx])

        # Normalise to cosine-like score
        win_energy   = float(np.sum(src_env[peak_idx:peak_idx + n_stretched] ** 2))
        query_energy = float(np.sum(stretched ** 2))
        denom = np.sqrt(win_energy * query_energy)
        score = peak_val / denom if denom > 1e-12 else 0.0

        if score > best_score:
            best_score  = score
            best_offset = peak_idx * hop_sec
            best_speed  = speed

    return best_offset, best_speed


def quality_label(quality: float) -> str:
    """Return a human-readable label for an xcorr quality score."""
    if quality >= XCORR_QUALITY_GOOD:
        return "good match"
    if quality >= XCORR_QUALITY_FAIR:
        return "moderate match"
    return "poor match — possible content mismatch or heavy lossy artefacts"
