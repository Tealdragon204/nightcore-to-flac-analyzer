"""
Pitch-shift detection via chromagram cross-correlation (primary) and
optional MELODIA refinement (essentia, if installed).

Primary method — chromagram cross-correlation (librosa)
--------------------------------------------------------
Splits both audio signals into ~20-second chunks and computes a
high-resolution CQT chroma vector (36 bins/octave = 1/3 semitone per bin)
for each chunk.  A cyclic cross-correlation over all 36 lags finds the
rotation that best aligns the two chroma vectors; the lag index (converted
to semitones) is the coarse pitch shift for that chunk.

A bootstrap over chunk-level shifts gives a 95 % confidence interval.
Because the method works on harmonic energy across all pitch classes it is
robust to polyphonic, fully-mixed music — the main limitation of the
previous CREPE-based approach.

Output format
-------------
Both functions return ``List[Optional[float]]`` in Hz, compatible with
``consensus._bootstrap_ratio``.  For the chroma path the src list is all
440 Hz entries and the nc list is ``440 × 2^(shift_i/12)`` per chunk, so
``median(nc) / median(src)`` equals the correct pitch ratio.  For the
MELODIA path the lists contain per-frame voiced F0 values directly.

Optional refinement — MELODIA (essentia)
-----------------------------------------
If ``essentia`` is installed, ``PredominantPitchMelodia`` is run on both
files to extract the dominant melody pitch track.  If the MELODIA estimate
agrees with the chroma result within ±1.5 semitones, the MELODIA Hz lists
(higher resolution) are used instead.  If essentia is absent or MELODIA
disagrees, the chroma result is used unchanged.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Callable, List, Optional, Tuple

# ── tunables ──────────────────────────────────────────────────────────────────
CHROMA_BINS_PER_OCTAVE: int = 36      # 3 bins/semitone → 1/3 st resolution
CHROMA_HOP_LENGTH: int      = 512     # CQT hop, ~23 ms at sr=22050
CHUNK_SEC: float            = 20.0    # chunk duration for per-chunk shifts
MIN_CHUNKS: int             = 3       # minimum chunks required for a CI
MELODIA_AGREE_ST: float     = 1.5    # MELODIA accepted only within this range of chroma
MAX_MELODIA_FRAMES: int     = 2000   # subsample MELODIA output to this many frames

# Reference pitch for chroma → Hz conversion (A4)
_REF_HZ: float = 440.0


# ── chroma helpers ────────────────────────────────────────────────────────────

def _mean_chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return the time-averaged CQT chroma vector (shape: [CHROMA_BINS_PER_OCTAVE])."""
    import librosa  # already a project dependency
    C = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        bins_per_octave=CHROMA_BINS_PER_OCTAVE,
        hop_length=CHROMA_HOP_LENGTH,
    )
    return C.mean(axis=1)  # shape (36,)


def _cyclic_xcorr_peak(src_chroma: np.ndarray, nc_chroma: np.ndarray) -> int:
    """
    Return the cyclic cross-correlation lag (in chroma bins) at which
    *nc_chroma* best aligns with *src_chroma*.

    A positive lag means nc is shifted UP relative to src.
    Result is wrapped to the range [-(n//2), n//2] so the sign is meaningful.
    Each bin equals 1/3 semitone when CHROMA_BINS_PER_OCTAVE=36.
    """
    n = len(src_chroma)
    xcorr = np.array([
        float(np.dot(src_chroma, np.roll(nc_chroma, -k)))
        for k in range(n)
    ])
    raw_lag = int(np.argmax(xcorr))
    # Wrap to symmetric range
    if raw_lag > n // 2:
        raw_lag -= n
    return raw_lag


def _chroma_shift_for_chunk(
    src_chunk: np.ndarray,
    nc_chunk: np.ndarray,
    sr: int,
) -> float:
    """Return the pitch shift in semitones (nc relative to src) for one chunk pair."""
    lag = _cyclic_xcorr_peak(_mean_chroma(src_chunk, sr), _mean_chroma(nc_chunk, sr))
    return lag / 3.0  # bins → semitones (3 bins per semitone)


# ── stage 1: chroma cross-correlation ────────────────────────────────────────

def estimate_pitch_chroma(
    src_audio: np.ndarray,
    nc_audio: np.ndarray,
    sr: int,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[
    List[Optional[float]],   # src implied Hz per chunk
    List[Optional[float]],   # nc  implied Hz per chunk
    float,                   # point estimate: shift in semitones
    Tuple[float, float],     # bootstrap 95 % CI in semitones
    int,                     # number of chunks used
]:
    """
    Estimate the pitch shift between two audio files using CQT chromagram
    cross-correlation.

    Returns per-chunk implied Hz lists compatible with
    ``consensus._bootstrap_ratio``.  src values are fixed at _REF_HZ (440 Hz);
    nc values are ``_REF_HZ × 2^(shift_i/12)`` so that
    ``median(nc_hz) / median(src_hz)`` equals the correct pitch ratio.
    """
    chunk_n = int(CHUNK_SEC * sr)
    n_src_chunks = len(src_audio) // chunk_n
    n_nc_chunks  = len(nc_audio)  // chunk_n
    n_chunks = min(n_src_chunks, n_nc_chunks)

    if n_chunks < 1:
        # Fallback: use whole file as a single chunk
        shift_sts = np.array([_chroma_shift_for_chunk(src_audio, nc_audio, sr)])
        n_chunks = 1
    else:
        shift_sts = np.array([
            _chroma_shift_for_chunk(
                src_audio[i * chunk_n : (i + 1) * chunk_n],
                nc_audio[i  * chunk_n : (i + 1) * chunk_n],
                sr,
            )
            for i in range(n_chunks)
        ])

    point_st = float(np.median(shift_sts))

    # Bootstrap CI over chunk shifts
    if n_chunks >= MIN_CHUNKS:
        rng = np.random.default_rng(0)
        boots = np.array([
            float(np.median(rng.choice(shift_sts, size=n_chunks, replace=True)))
            for _ in range(2000)
        ])
        ci_lo_st = float(np.percentile(boots, 2.5))
        ci_hi_st = float(np.percentile(boots, 97.5))
    else:
        # Not enough chunks for a meaningful CI
        ci_lo_st = ci_hi_st = point_st
        if log:
            log(
                f"    Only {n_chunks} chunk(s) available (need ≥ {MIN_CHUNKS}) — "
                "pitch CI is degenerate; estimate may be less reliable."
            )

    # Convert semitone shifts to implied Hz pairs for consensus compatibility
    src_hz: List[Optional[float]] = [_REF_HZ] * n_chunks
    nc_hz: List[Optional[float]] = [
        _REF_HZ * (2.0 ** (st / 12.0)) for st in shift_sts
    ]

    if log:
        log(
            f"    Chroma xcorr: {point_st:+.3f} st"
            f"  95% CI [{ci_lo_st:+.3f}, {ci_hi_st:+.3f}] st"
            f"  ({n_chunks} chunk{'s' if n_chunks != 1 else ''})"
        )

    return src_hz, nc_hz, point_st, (ci_lo_st, ci_hi_st), n_chunks


# ── stage 2: MELODIA refinement (optional) ────────────────────────────────────

def _try_import_essentia():
    """Return essentia.standard if installed, else None — never raises."""
    try:
        import essentia.standard as es  # type: ignore[import]
        return es
    except Exception:
        return None


def estimate_pitch_melodia(
    src_audio: np.ndarray,
    nc_audio: np.ndarray,
    sr: int,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Tuple[List[Optional[float]], List[Optional[float]]]]:
    """
    Run essentia PredominantPitchMelodia on both files and return voiced
    F0 lists (Hz).  Returns None if essentia is not installed, MELODIA
    fails, or either file yields no voiced frames.

    Output is subsampled to at most MAX_MELODIA_FRAMES entries per file
    to keep memory usage and histogram rendering fast.
    """
    es = _try_import_essentia()
    if es is None:
        if log:
            log("    essentia not available — skipping MELODIA refinement")
        return None

    def _extract(audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            extractor = es.PredominantPitchMelodia(
                frameSize=2048,
                hopSize=128,
                sampleRate=float(sr),
            )
            pitch_hz, _ = extractor(audio.astype(np.float32))
            voiced = pitch_hz[pitch_hz > 0.0]
            if len(voiced) == 0:
                return None
            # Subsample if needed
            if len(voiced) > MAX_MELODIA_FRAMES:
                step = len(voiced) // MAX_MELODIA_FRAMES
                voiced = voiced[::step]
            return voiced
        except Exception as exc:
            if log:
                log(f"    MELODIA extraction failed: {exc}")
            return None

    src_voiced = _extract(src_audio)
    nc_voiced  = _extract(nc_audio)

    if src_voiced is None or nc_voiced is None:
        return None

    src_hz: List[Optional[float]] = [float(v) for v in src_voiced]
    nc_hz:  List[Optional[float]] = [float(v) for v in nc_voiced]

    melodia_st = 12.0 * math.log2(float(np.median(nc_voiced)) / float(np.median(src_voiced)))
    if log:
        log(f"    MELODIA: {melodia_st:+.6f} st  ({len(src_voiced)} src / {len(nc_voiced)} nc voiced frames)")

    return src_hz, nc_hz


# ── public API ─────────────────────────────────────────────────────────────────

def estimate_pitch_combined(
    src_audio: np.ndarray,
    nc_audio: np.ndarray,
    sr: int,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[
    List[Optional[float]],   # src_pitches (for consensus + histogram)
    List[Optional[float]],   # nc_pitches  (for consensus + histogram)
    str,                     # method: "chroma_xcorr" | "chroma+melodia"
]:
    """
    Estimate the pitch shift between src and nc audio.

    Runs chromagram cross-correlation (always) then attempts MELODIA
    refinement (if essentia is installed).  MELODIA is accepted only when
    its shift estimate agrees with chroma within ±MELODIA_AGREE_ST semitones.

    Returns per-sample Hz lists compatible with ``consensus.build_result``
    and the histogram widget, plus a method string for reporting.
    """
    # Stage 1 — chroma (always runs)
    src_chroma_hz, nc_chroma_hz, chroma_st, _, _ = estimate_pitch_chroma(
        src_audio, nc_audio, sr, log=log,
    )

    # Stage 2 — MELODIA (optional)
    melodia_result = estimate_pitch_melodia(src_audio, nc_audio, sr, log=log)

    if melodia_result is not None:
        src_mel_hz, nc_mel_hz = melodia_result
        src_med = float(np.median([v for v in src_mel_hz if v is not None]))
        nc_med  = float(np.median([v for v in nc_mel_hz  if v is not None]))
        if src_med > 0 and nc_med > 0:
            melodia_st = 12.0 * math.log2(nc_med / src_med)
            if abs(melodia_st - chroma_st) <= MELODIA_AGREE_ST:
                return src_mel_hz, nc_mel_hz, "chroma+melodia"
            else:
                if log:
                    log(
                        f"    MELODIA ({melodia_st:+.3f} st) disagrees with chroma"
                        f" ({chroma_st:+.3f} st) by"
                        f" {abs(melodia_st - chroma_st):.2f} st"
                        f" > {MELODIA_AGREE_ST} st threshold — using chroma only"
                    )

    return src_chroma_hz, nc_chroma_hz, "chroma_xcorr"
