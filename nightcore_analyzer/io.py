"""
Audio I/O, windowing, energy gating, and silence stripping.

All audio is loaded as mono float32 at SAMPLE_RATE.  Leading/trailing
silence is optionally trimmed before windowing (strip_silence).  Windows
are produced with a configurable duration and hop, then passed through an
energy gate that discards segments more than ENERGY_GATE_DB below the
loudest window.
"""

from __future__ import annotations

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List

# ── defaults ─────────────────────────────────────────────────────────────────
SAMPLE_RATE: int   = 22050   # Hz — works for both librosa and CREPE (which resamples internally)
WINDOW_SEC: float  = 10.0    # window duration
HOP_SEC: float     = 5.0     # 50% overlap
ENERGY_GATE_DB: float = -40.0  # discard windows this many dB below the peak window
SILENCE_STRIP_DB: float = 60.0  # top_db passed to librosa.effects.trim


# ── data container ────────────────────────────────────────────────────────────
@dataclass
class AudioWindow:
    """One time slice of an audio file."""
    audio: np.ndarray   # float32 mono, length = window_sec * sample_rate
    sample_rate: int
    start_sec: float
    end_sec: float
    energy_db: float    # RMS energy in dB (full-scale)


# ── helpers ───────────────────────────────────────────────────────────────────
def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-10))


# ── public API ────────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load *path* as mono float32 resampled to *sr* Hz.

    Returns
    -------
    (audio, sr)
        audio : 1-D float32 ndarray
        sr    : actual sample rate used
    """
    y, _ = librosa.load(path, sr=sr, mono=True, dtype=np.float32)
    return y, sr


def strip_silence(
    audio: np.ndarray,
    sr: int,
    top_db: float = SILENCE_STRIP_DB,
) -> tuple[np.ndarray, float, float]:
    """
    Trim leading and trailing silence from *audio*.

    Uses librosa.effects.trim with the given *top_db* threshold: frames more
    than *top_db* dB below the peak frame are considered silent.

    Returns
    -------
    (trimmed_audio, leading_sec, trailing_sec)
        trimmed_audio : silence-trimmed 1-D float32 ndarray
        leading_sec   : seconds of silence removed from the start
        trailing_sec  : seconds of silence removed from the end
    """
    trimmed, (start, end) = librosa.effects.trim(audio, top_db=top_db)
    leading_sec  = start / sr
    trailing_sec = (len(audio) - end) / sr
    return trimmed, leading_sec, trailing_sec


def slice_windows(
    audio: np.ndarray,
    sr: int,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
) -> List[AudioWindow]:
    """
    Slice *audio* into overlapping fixed-length windows.

    Windows shorter than *window_sec* at the end of the signal are discarded
    so every window has exactly the same length.
    """
    win_n = int(window_sec * sr)
    hop_n = int(hop_sec * sr)

    windows: List[AudioWindow] = []
    start = 0
    while start + win_n <= len(audio):
        chunk = audio[start : start + win_n]
        windows.append(
            AudioWindow(
                audio=chunk,
                sample_rate=sr,
                start_sec=start / sr,
                end_sec=(start + win_n) / sr,
                energy_db=_rms_db(chunk),
            )
        )
        start += hop_n

    return windows


def energy_gate(
    windows: List[AudioWindow],
    threshold_db: float = ENERGY_GATE_DB,
) -> List[AudioWindow]:
    """
    Remove windows whose RMS energy is more than *threshold_db* below the
    loudest window.  Returns whatever windows survive (may be empty).
    """
    if not windows:
        return windows
    peak_db = max(w.energy_db for w in windows)
    return [w for w in windows if w.energy_db >= peak_db + threshold_db]
