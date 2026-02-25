"""
Spectral comparison between two audio files.

Compares brightness, high-frequency content, dynamic range, frequency-band
balance, and reverb tail to surface plain-English differences between a
reference file and another file (typically HQNC vs NCOG).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa


# ── data container ────────────────────────────────────────────────────────────

@dataclass
class SpectralStats:
    centroid: float     # Hz — average spectral centroid (brightness)
    rolloff: float      # Hz — 85th-percentile spectral rolloff
    rms_mean: float     # average RMS energy
    rms_variance: float # variance of RMS frames (dynamics)
    sub_bass: float     # 20–80 Hz mean STFT magnitude
    bass: float         # 80–250 Hz
    midrange: float     # 250–2000 Hz
    presence: float     # 2000–6000 Hz
    brilliance: float   # 6000–20000 Hz
    decay_rate: float   # mean diff of loud RMS frames (reverb proxy)
    duration: float     # seconds


# ── analysis ──────────────────────────────────────────────────────────────────

def analyze(path: str, label: Optional[str] = None) -> SpectralStats:
    """
    Load *path* and return its spectral statistics.

    Parameters
    ----------
    path : str
        Path to any audio file that librosa can open.
    label : str, optional
        Printed while loading (e.g. "HQNC").  Pass None to suppress.
    """
    if label:
        print(f"  Loading {label}…")

    y, sr = librosa.load(path, sr=None, mono=True)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff  = float(np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    ))

    rms        = librosa.feature.rms(y=y)[0]
    rms_mean   = float(np.mean(rms))
    rms_var    = float(np.var(rms))

    stft  = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    def _band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.mean(stft[mask, :])) if mask.any() else 0.0

    sub_bass   = _band(20,    80)
    bass       = _band(80,   250)
    midrange   = _band(250,  2000)
    presence   = _band(2000, 6000)
    brilliance = _band(6000, 20000)

    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    loud       = rms_frames[rms_frames > np.percentile(rms_frames, 75)]
    decay_rate = float(np.mean(np.diff(loud))) if len(loud) > 1 else 0.0

    duration = librosa.get_duration(y=y, sr=sr)

    return SpectralStats(
        centroid=centroid, rolloff=rolloff,
        rms_mean=rms_mean, rms_variance=rms_var,
        sub_bass=sub_bass, bass=bass, midrange=midrange,
        presence=presence, brilliance=brilliance,
        decay_rate=decay_rate, duration=duration,
    )


# ── comparison / reporting ────────────────────────────────────────────────────

def _pct(a: float, b: float) -> float:
    """Percentage change from a to b."""
    return ((b - a) / a) * 100 if a != 0 else 0.0


def compare_and_print(
    ref: SpectralStats,
    other: SpectralStats,
    label_ref: str = "REFERENCE",
    label_other: str = "OTHER",
    ref_path: Optional[str] = None,
    other_path: Optional[str] = None,
) -> None:
    """
    Print a plain-English spectral comparison report to stdout.

    Parameters
    ----------
    ref, other : SpectralStats
        Statistics for the two files.
    label_ref, label_other : str
        Human-readable names used in the report.
    ref_path, other_path : str, optional
        File paths — used only to infer format (FLAC/MP3) for the quality note.
    """
    W = 57  # report width

    print()
    print("=" * W)
    print("SPECTRAL COMPARISON RESULTS")
    print(f"  Reference : {label_ref}")
    print(f"  Other     : {label_other}")
    print("=" * W)

    # ── brightness ────────────────────────────────────────────────────────────
    bd = _pct(ref.centroid, other.centroid)
    print(f"\nBRIGHTNESS (Spectral Centroid)")
    print(f"  {label_ref}: {ref.centroid:.1f} Hz  |  {label_other}: {other.centroid:.1f} Hz")
    if bd < -10:
        print(f"  ! {label_other} is {abs(bd):.1f}% DARKER  -> likely low-pass filter applied")
    elif bd > 10:
        print(f"  ! {label_other} is {bd:.1f}% BRIGHTER  -> likely high-pass or treble boost")
    else:
        print(f"  OK  Similar brightness ({bd:+.1f}%)")

    # ── high-frequency rolloff ─────────────────────────────────────────────────
    rd = _pct(ref.rolloff, other.rolloff)
    print(f"\nHIGH FREQUENCY ROLLOFF")
    print(f"  {label_ref}: {ref.rolloff:.1f} Hz  |  {label_other}: {other.rolloff:.1f} Hz")
    if rd < -10:
        print(f"  ! {label_other} has {abs(rd):.1f}% less high-frequency energy  -> treble cut confirmed")
    elif rd > 10:
        print(f"  ! {label_other} has {rd:.1f}% more high-frequency energy  -> treble boost")
    else:
        print(f"  OK  Similar high-frequency content ({rd:+.1f}%)")

    # ── dynamic range ──────────────────────────────────────────────────────────
    vd = _pct(ref.rms_variance, other.rms_variance)
    print(f"\nDYNAMIC RANGE (Compression)")
    print(f"  {label_ref} variance: {ref.rms_variance:.6f}  |  {label_other}: {other.rms_variance:.6f}")
    if vd < -30:
        print(f"  ! {label_other} is {abs(vd):.1f}% more compressed  -> heavy limiting/compression")
    elif vd < -10:
        print(f"  ! {label_other} is {abs(vd):.1f}% more compressed  -> moderate compression")
    elif vd > 30:
        print(f"  ! {label_other} has {vd:.1f}% MORE dynamic range  -> less compressed than reference")
    else:
        print(f"  OK  Similar dynamic range ({vd:+.1f}%)")

    # ── frequency bands ────────────────────────────────────────────────────────
    print(f"\nFREQUENCY BAND BREAKDOWN")
    bands = [
        ("Sub-bass  (20–80 Hz)",   ref.sub_bass,   other.sub_bass),
        ("Bass      (80–250 Hz)",  ref.bass,        other.bass),
        ("Midrange  (250–2 kHz)",  ref.midrange,    other.midrange),
        ("Presence  (2–6 kHz)",    ref.presence,    other.presence),
        ("Brilliance (6–20 kHz)",  ref.brilliance,  other.brilliance),
    ]
    for name, rv, ov in bands:
        diff = _pct(rv, ov)
        ok   = abs(diff) < 10
        tag  = "OK" if ok else "! "
        more = "more" if diff > 0 else "less"
        print(f"  {tag}  {name}: {diff:+.1f}% ({more} in {label_other})")

    # ── reverb ────────────────────────────────────────────────────────────────
    dd = _pct(ref.decay_rate, other.decay_rate)
    print(f"\nREVERB / DECAY")
    if other.decay_rate > ref.decay_rate * 0.8 and abs(dd) > 20:
        print(f"  ! {label_other} decays more slowly ({dd:+.1f}%)  -> possible reverb added")
    else:
        print(f"  OK  Similar decay characteristics ({dd:+.1f}%)")

    # ── duration note ──────────────────────────────────────────────────────────
    dur_diff = abs(other.duration - ref.duration)
    if dur_diff > 1.0:
        print(f"\nDURATION NOTE")
        print(f"  {label_ref}: {ref.duration:.1f} s  |  {label_other}: {other.duration:.1f} s")
        print(f"  ! Files differ by {dur_diff:.1f} s  -> different edits, fade-in/out, or intro/outro")

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("SUMMARY")
    print("=" * W)

    issues = []
    if bd < -10:
        issues.append(f"low-pass filter ({abs(bd):.0f}% darker)")
    elif bd > 10:
        issues.append(f"treble boost ({bd:.0f}% brighter)")
    if rd < -10:
        issues.append(f"treble cut ({abs(rd):.0f}% rolloff reduction)")
    if vd < -30:
        issues.append(f"heavy compression ({abs(vd):.0f}% less dynamic range)")
    elif vd < -10:
        issues.append(f"moderate compression ({abs(vd):.0f}% less dynamic range)")
    brill_diff = _pct(ref.brilliance, other.brilliance)
    if brill_diff < -20:
        issues.append(
            f"reduced high-frequency content ({abs(brill_diff):.0f}% less brilliance"
            " — consistent with MP3 compression)"
        )
    if other.decay_rate > ref.decay_rate * 0.8 and abs(dd) > 20:
        issues.append("slower decay (possible reverb)")
    if dur_diff > 1.0:
        issues.append(f"duration mismatch ({dur_diff:.1f} s — different edits)")

    if issues:
        print(f"Detected differences in {label_other}:")
        for item in issues:
            print(f"  - {item}")
    else:
        print("No significant spectral differences detected.")

    # ── format / quality note ──────────────────────────────────────────────────
    _format_quality_note(ref_path, other_path, ref.brilliance, other.brilliance,
                         label_ref, label_other)


def _format_quality_note(
    ref_path: Optional[str],
    other_path: Optional[str],
    ref_brilliance: float,
    other_brilliance: float,
    label_ref: str,
    label_other: str,
) -> None:
    """Print a short quality/format note if paths are available."""
    if not ref_path or not other_path:
        return

    def _fmt(p: str) -> str:
        return str(p).rsplit(".", 1)[-1].lower() if "." in str(p) else "?"

    fmt_ref   = _fmt(ref_path)
    fmt_other = _fmt(other_path)
    lossless  = {"flac", "wav", "aiff", "aif", "pcm"}

    ref_lossless   = fmt_ref   in lossless
    other_lossless = fmt_other in lossless

    print()
    print("FORMAT / QUALITY NOTE")
    if ref_lossless and not other_lossless:
        print(
            f"  {label_ref} is lossless ({fmt_ref.upper()}) and {label_other} is lossy "
            f"({fmt_other.upper()}).  {label_ref} is the higher-quality source."
        )
    elif other_lossless and not ref_lossless:
        print(
            f"  {label_other} is lossless ({fmt_other.upper()}) but {label_ref} is lossy "
            f"({fmt_ref.upper()}).  Check that files are in the correct order."
        )
    elif not ref_lossless and not other_lossless:
        print(f"  Both files appear lossy ({fmt_ref.upper()} / {fmt_other.upper()}).")
    else:
        print(f"  Both files are lossless ({fmt_ref.upper()} / {fmt_other.upper()}).")

    # Warn if the "lossless" file has less brilliance (likely the files are swapped)
    brill_diff = _pct(ref_brilliance, other_brilliance)
    if ref_lossless and not other_lossless and brill_diff > 20:
        print(
            f"  Warning: {label_other} (lossy) has more high-frequency content than "
            f"{label_ref} (lossless).  The files may be in the wrong order."
        )
