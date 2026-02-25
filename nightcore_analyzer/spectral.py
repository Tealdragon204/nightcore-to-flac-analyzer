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
    effective_bandwidth_hz: float  # highest freq with significant energy (lossy-transcode indicator)


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

    # Effective bandwidth — highest frequency bin with significant energy.
    # MP3 128k hard-cuts at ~16 kHz, 192k at ~18 kHz, 320k at ~20 kHz.
    # A "lossless" container (FLAC/WAV) with a sharp cutoff below ~18 kHz
    # was almost certainly transcoded from a lossy source — the container
    # format alone does not guarantee actual lossless content.
    stft_db     = librosa.amplitude_to_db(stft, ref=np.max)
    freq_avg_db = np.mean(stft_db, axis=1)          # per-bin average dB
    # 60 dB below the loudest bin ≈ effectively silent
    significant = freq_avg_db > (np.max(freq_avg_db) - 60.0)
    if significant.any():
        effective_bw = float(freqs[np.where(significant)[0][-1]])
    else:
        effective_bw = float(freqs[-1])

    return SpectralStats(
        centroid=centroid, rolloff=rolloff,
        rms_mean=rms_mean, rms_variance=rms_var,
        sub_bass=sub_bass, bass=bass, midrange=midrange,
        presence=presence, brilliance=brilliance,
        decay_rate=decay_rate, duration=duration,
        effective_bandwidth_hz=effective_bw,
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
    _format_quality_note(
        ref_path, other_path, ref.brilliance, other.brilliance,
        label_ref, label_other,
        ref_bandwidth=ref.effective_bandwidth_hz,
        other_bandwidth=other.effective_bandwidth_hz,
    )


def _format_quality_note(
    ref_path: Optional[str],
    other_path: Optional[str],
    ref_brilliance: float,
    other_brilliance: float,
    label_ref: str,
    label_other: str,
    ref_bandwidth: Optional[float] = None,
    other_bandwidth: Optional[float] = None,
) -> None:
    """
    Print a quality/format note.

    Uses the measured effective bandwidth (highest frequency with significant
    energy) rather than the file extension to assess actual audio quality.
    A FLAC that cuts off at ~16 kHz was almost certainly converted from MP3
    128k — its container is lossless but its content is not.
    """
    if not ref_path or not other_path:
        return

    def _fmt(p: str) -> str:
        return str(p).rsplit(".", 1)[-1].lower() if "." in str(p) else "?"

    fmt_ref   = _fmt(ref_path)
    fmt_other = _fmt(other_path)
    lossless  = {"flac", "wav", "aiff", "aif", "pcm"}

    ref_container_lossless   = fmt_ref   in lossless
    other_container_lossless = fmt_other in lossless

    # ── Lossy-transcode detection thresholds ──────────────────────────────────
    # MP3 128k → cutoff ~16 kHz; 192k → ~18 kHz; 320k → ~20 kHz.
    # We flag anything below 18.5 kHz in a "lossless" container as suspect.
    _TRANSCODE_THRESHOLD_HZ = 18_500

    def _transcode_grade(bw: Optional[float]) -> Optional[str]:
        """Return a human-readable guess of lossy source bitrate, or None."""
        if bw is None:
            return None
        if bw < 16_500:
            return "MP3 ~128 kbps"
        if bw < 18_500:
            return "MP3 ~192 kbps"
        if bw < 20_000:
            return "MP3 ~320 kbps"
        return None   # looks genuinely lossless

    ref_transcode   = _transcode_grade(ref_bandwidth)   if ref_container_lossless   else None
    other_transcode = _transcode_grade(other_bandwidth) if other_container_lossless else None

    # "True lossless" = lossless container AND no transcode signature
    ref_truly_lossless   = ref_container_lossless   and ref_transcode   is None
    other_truly_lossless = other_container_lossless and other_transcode is None

    print()
    print("FORMAT / QUALITY NOTE")

    # Container formats
    fmt_line = (
        f"  Container: {label_ref} → {fmt_ref.upper()}   |   "
        f"{label_other} → {fmt_other.upper()}"
    )
    print(fmt_line)

    # Bandwidth measurements
    if ref_bandwidth and other_bandwidth:
        print(
            f"  Effective bandwidth: {label_ref} → {ref_bandwidth/1000:.1f} kHz   |   "
            f"{label_other} → {other_bandwidth/1000:.1f} kHz"
        )

    # Transcode warnings (appear even if the extension says FLAC)
    for label, container_lossless, transcode, bw in [
        (label_ref,   ref_container_lossless,   ref_transcode,   ref_bandwidth),
        (label_other, other_container_lossless, other_transcode, other_bandwidth),
    ]:
        if container_lossless and transcode and bw:
            print(
                f"  ! {label} ({fmt_ref.upper() if label == label_ref else fmt_other.upper()})"
                f" — spectral content cuts off at ~{bw/1000:.1f} kHz, consistent with "
                f"{transcode} encoding. This file appears to be a lossy-to-lossless "
                f"transcode; the lossless container does NOT guarantee lossless audio."
            )

    # Verdict
    if ref_truly_lossless and not other_truly_lossless:
        print(
            f"  Verdict: {label_ref} is genuinely lossless — "
            f"{label_other} is lower quality."
        )
    elif other_truly_lossless and not ref_truly_lossless:
        print(
            f"  Verdict: {label_other} is genuinely lossless but {label_ref} is not — "
            f"check that files are in the correct order."
        )
    elif not ref_truly_lossless and not other_truly_lossless:
        print("  Verdict: Neither file appears to be a genuine lossless master.")
    else:
        print("  Verdict: Both files appear to be genuinely lossless.")

    # Swap-order check: if the "lossless" file has notably less HF content
    brill_diff = _pct(ref_brilliance, other_brilliance)
    if ref_truly_lossless and not other_truly_lossless and brill_diff > 20:
        print(
            f"  Warning: {label_other} (lower quality by format) has more high-frequency "
            f"content than {label_ref}. The files may be in the wrong order."
        )
