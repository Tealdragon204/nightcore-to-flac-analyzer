"""
Loudness / clipping detection and adjustment for audio files.

Two adjustment methods are provided:

  True Peak Limiter  (preferred)
      Uses ffmpeg's ``alimiter`` filter.  Only attenuates samples that exceed
      the ceiling — everything below the threshold is completely untouched.
      This is surgical: a 0.2 dBFS over-peak does not require pulling the
      whole track down 0.2 dB; the limiter shaves only those peaks off.
      Preserves dynamic range; prevents crackling on playback.

  Gain Reduction  (brute force / fallback)
      Uniformly shifts the entire signal down by N dB via ``sox gain``.
      Simple and reliable, but reduces loudness across the board — the quiet
      parts get quieter even though they were never the problem.

Public API
----------
detect_peak(path)                  → (peak_dbfs: float, is_clipping: bool)
apply_true_peak_limiter(src, dst)  → None
apply_gain_reduction(src, dst, gain_db)  → None
make_adj_path(src, version)        → Path
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# ── peak detection ─────────────────────────────────────────────────────────────

def detect_peak(path: str | Path) -> tuple[float, bool]:
    """
    Return *(peak_dbfs, is_clipping)* for the audio file at *path*.

    *peak_dbfs* is the maximum absolute sample value in dBFS (full-scale).
    A value >= 0.0 means the file has digital clipping (samples at or above
    full-scale), which can cause crackling during playback.

    Parameters
    ----------
    path : str or Path
        Any audio file that soundfile can open (FLAC, WAV, AIFF, …).

    Returns
    -------
    peak_dbfs : float
        Peak level in dBFS.  -inf if the file is silent.
    is_clipping : bool
        True when peak_dbfs >= 0.0.
    """
    data, _sr = sf.read(str(path), always_2d=True)
    peak_linear = float(np.max(np.abs(data)))
    if peak_linear == 0.0:
        return (-math.inf, False)
    peak_dbfs = 20.0 * math.log10(peak_linear)
    return (peak_dbfs, peak_dbfs >= 0.0)


# ── path helper ───────────────────────────────────────────────────────────────

def make_adj_path(src: Path, version: int) -> Path:
    """
    Return a versioned ADJ path next to *src*.

    Examples
    --------
    make_adj_path(Path("Song [Nightcore].flac"), 1)
        → Path("Song [Nightcore] ADJ1.flac")
    make_adj_path(Path("Song.flac"), 2)
        → Path("Song ADJ2.flac")
    """
    return src.with_name(src.stem + f" ADJ{version}" + src.suffix)


# ── true peak limiter ─────────────────────────────────────────────────────────

def apply_true_peak_limiter(
    src: Path,
    dst: Path,
    limit_db: float = -0.1,
) -> None:
    """
    Apply a true peak limiter to *src* and write the result to *dst*.

    Uses ffmpeg's ``alimiter`` filter.  Only samples that exceed *limit_db*
    are attenuated; everything below is left completely unchanged.

    Parameters
    ----------
    src : Path
        Input audio file.
    dst : Path
        Output audio file (same format/extension is recommended).
    limit_db : float
        Ceiling in dBFS.  Default -0.1 dBFS (just below 0) — prevents
        inter-sample peaks while making no audible difference.

    Raises
    ------
    SystemExit
        If ffmpeg is not found on PATH.
    subprocess.CalledProcessError
        If ffmpeg exits with a non-zero status.
    """
    if not shutil.which("ffmpeg"):
        print(
            "\n  ERROR: ffmpeg not found on PATH.\n"
            "  Install it:  sudo apt install ffmpeg   (Debian/Ubuntu)\n"
            "               brew install ffmpeg        (macOS)\n"
        )
        raise SystemExit(1)

    # alimiter expects a linear limit value (1.0 = 0 dBFS)
    limit_linear = 10 ** (limit_db / 20.0)

    # attack/release in milliseconds; gentle values to preserve transients
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-af", f"alimiter=limit={limit_linear:.6f}:attack=5:release=50:level=disabled",
        str(dst),
    ]
    print(f"\n  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Created: {dst}")


# ── gain reduction ────────────────────────────────────────────────────────────

def apply_gain_reduction(
    src: Path,
    dst: Path,
    gain_db: float,
) -> None:
    """
    Apply a uniform gain reduction to *src* and write the result to *dst*.

    Tries ``sox`` first; falls back to ``ffmpeg -af volume`` if sox is not
    available.

    Parameters
    ----------
    src : Path
        Input audio file.
    dst : Path
        Output audio file.
    gain_db : float
        Gain change in dB.  Should be negative to reduce level (e.g. -1.0).

    Raises
    ------
    SystemExit
        If neither sox nor ffmpeg is found on PATH.
    subprocess.CalledProcessError
        If the chosen tool exits with a non-zero status.
    """
    if shutil.which("sox"):
        cmd = ["sox", str(src), str(dst), "gain", f"{gain_db:.2f}"]
        print(f"\n  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"  Created: {dst}")
        return

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-af", f"volume={gain_db:.2f}dB",
            str(dst),
        ]
        print(f"\n  Running (ffmpeg fallback): {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"  Created: {dst}")
        return

    print(
        "\n  ERROR: neither sox nor ffmpeg found on PATH.\n"
        "  Install sox:   sudo apt install sox      (Debian/Ubuntu)\n"
        "                 brew install sox           (macOS)\n"
        "  Install ffmpeg: sudo apt install ffmpeg\n"
    )
    raise SystemExit(1)
