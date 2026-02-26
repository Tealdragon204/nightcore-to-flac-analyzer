"""
Full windowed consensus pipeline.

Orchestrates audio loading → windowing → energy gating → pitch estimation →
tempo estimation → consensus → classification → Rubber Band parameters.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from .io import (
    load_audio, strip_silence, slice_windows, energy_gate,
    WINDOW_SEC, HOP_SEC, ENERGY_GATE_DB, SILENCE_STRIP_DB,
)
from .pitch import batch_estimate_pitch
from .tempo import batch_estimate_tempo, estimate_ibis_global
from .consensus import build_result, compute_ibi_ratio, AnalysisResult
from .xcorr import find_content_offset, ALIGN_MIN_OFFSET


def run(
    nightcore_path: str,
    source_path: str,
    *,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
    energy_gate_db: float = ENERGY_GATE_DB,
    silence_strip_db: Optional[float] = SILENCE_STRIP_DB,
    auto_align: bool = True,
    log: Optional[Callable[[str], None]] = print,
) -> AnalysisResult:
    """
    Analyse the tempo and pitch relationship between a nightcore track and its
    FLAC source.

    Parameters
    ----------
    nightcore_path : str
        Path to the nightcore audio file (any format librosa can open).
    source_path : str
        Path to the original source audio (typically a FLAC).
    window_sec : float
        Duration of each analysis window in seconds (default 10 s).
    hop_sec : float
        Hop between consecutive windows in seconds (default 5 s = 50% overlap).
    energy_gate_db : float
        Windows whose RMS energy is below peak − |energy_gate_db| dB are
        discarded as silence / low-energy artefacts (default −40 dB).
    silence_strip_db : float or None
        Top-dB threshold for trimming leading/trailing silence from each
        audio file before windowing.  Uses librosa.effects.trim: a frame
        is silent when its power is more than *silence_strip_db* dB below
        the peak frame (default 60 dB).  Pass ``None`` to disable.
    log : callable or None
        Called with progress messages.  Pass ``None`` to suppress output.

    Returns
    -------
    AnalysisResult
        Tempo ratio, pitch ratio, 95 % confidence intervals, classification,
        and Rubber Band reconstruction parameters.
    """
    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    # ── 1. load ───────────────────────────────────────────────────────────────
    _log("Loading nightcore audio…")
    nc_audio, sr = load_audio(nightcore_path)
    _log(f"  {len(nc_audio) / sr:.1f} s  ({len(nc_audio):,} samples @ {sr} Hz)")

    _log("Loading source audio…")
    src_audio, _ = load_audio(source_path, sr=sr)
    _log(f"  {len(src_audio) / sr:.1f} s  ({len(src_audio):,} samples @ {sr} Hz)")

    # ── 1b. strip leading/trailing silence ────────────────────────────────────
    if silence_strip_db is not None:
        _log(f"Stripping silence (top_db={silence_strip_db} dB)…")
        nc_audio,  nc_lead,  nc_trail  = strip_silence(nc_audio,  sr, silence_strip_db)
        src_audio, src_lead, src_trail = strip_silence(src_audio, sr, silence_strip_db)
        _log(
            f"  nightcore: −{nc_lead:.2f}s leading, −{nc_trail:.2f}s trailing"
            f"  →  {len(nc_audio)/sr:.1f} s"
        )
        _log(
            f"  source:    −{src_lead:.2f}s leading, −{src_trail:.2f}s trailing"
            f"  →  {len(src_audio)/sr:.1f} s"
        )

    # ── 1c. content-alignment: detect and skip musical intro in src ───────────
    intro_offset_sec: Optional[float] = None
    if auto_align:
        _log("Detecting intro offset (RMS envelope alignment)…")
        raw_offset, align_speed = find_content_offset(src_audio, nc_audio, sr)
        if raw_offset >= ALIGN_MIN_OFFSET:
            src_audio = src_audio[int(raw_offset * sr):]
            intro_offset_sec = raw_offset
            _log(
                f"  Intro detected — trimming {raw_offset:.2f}s from source start"
                f"  (speed hint: {align_speed:.4f}×)"
            )
        else:
            _log(
                f"  No significant intro offset detected"
                f"  (raw: {raw_offset:.2f}s < {ALIGN_MIN_OFFSET:.1f}s threshold)"
            )

    # ── 2. window ─────────────────────────────────────────────────────────────
    _log(f"Slicing into {window_sec:.0f} s windows (hop {hop_sec:.0f} s)…")
    nc_windows  = slice_windows(nc_audio,  sr, window_sec, hop_sec)
    src_windows = slice_windows(src_audio, sr, window_sec, hop_sec)
    _log(f"  nightcore: {len(nc_windows)} windows  |  source: {len(src_windows)} windows")

    # ── 3. energy gate ────────────────────────────────────────────────────────
    _log(f"Energy gating (threshold {energy_gate_db} dB below peak)…")
    nc_windows  = energy_gate(nc_windows,  energy_gate_db)
    src_windows = energy_gate(src_windows, energy_gate_db)
    _log(
        f"  after gating — nightcore: {len(nc_windows)} windows"
        f"  |  source: {len(src_windows)} windows"
    )

    if not nc_windows or not src_windows:
        raise RuntimeError(
            "All windows were discarded by the energy gate.  "
            "Try raising --energy-gate (e.g. --energy-gate -60)."
        )

    # ── 4. pitch estimation (CREPE + GPU) ─────────────────────────────────────
    _log("Estimating pitch (CREPE)…")
    _log("  ← nightcore →")
    nc_pitches  = batch_estimate_pitch(nc_windows,  log=_log)
    _log("  ← source →")
    src_pitches = batch_estimate_pitch(src_windows, log=_log)

    # ── 5. tempo estimation (librosa) ─────────────────────────────────────────
    # Detect source tempos first (default prior), then use the source median
    # BPM × duration ratio to steer librosa toward the correct harmonic for
    # the nightcore.  Without this hint the beat tracker's prior (120 BPM)
    # can snap to a sub-beat periodicity (e.g. ~86 BPM) instead of the true
    # nightcore tempo (~128 BPM).
    _log("Estimating tempo (librosa)…")
    _log("  ← source →")
    src_tempos = batch_estimate_tempo(src_windows, log=_log)

    nc_duration  = len(nc_audio)  / sr
    src_duration = len(src_audio) / sr

    nc_start_bpm = 120.0  # librosa default — used when no hint is computable
    valid_src = [t for t in src_tempos if t is not None]
    if valid_src and nc_duration > 0 and src_duration > 0:
        median_src = float(np.median(valid_src))
        nc_start_bpm = median_src * (src_duration / nc_duration)
        _log(
            f"  NC tempo prior: {nc_start_bpm:.1f} BPM  "
            f"(src median {median_src:.1f} BPM × dur ratio "
            f"{src_duration / nc_duration:.4f})"
        )

    _log("  ← nightcore →")
    nc_tempos  = batch_estimate_tempo(nc_windows,  log=_log, start_bpm=nc_start_bpm)

    # ── 6. consensus + classification ─────────────────────────────────────────
    _log("Computing consensus…")
    result = build_result(
        src_pitches, nc_pitches, src_tempos, nc_tempos,
        nc_duration=nc_duration, src_duration=src_duration,
    )

    result.intro_offset_sec = intro_offset_sec

    # ── 7. IBI ratio pass (full-signal, high-resolution beat timestamps) ───────
    # Runs beat tracking at hop_length=64 (≈1.45 ms) over the entire signal,
    # giving ~0.01% precision on the speed ratio versus the ~0.1–0.3% from the
    # windowed BPM median.  Attached to the existing AnalysisResult so all
    # callers automatically receive it.
    _log("Computing IBI ratio (high-precision beat timestamps, hop=64)…")
    nc_ibis  = estimate_ibis_global(nc_audio,  sr, start_bpm=nc_start_bpm)
    src_ibis = estimate_ibis_global(src_audio, sr)
    if (nc_ibis  is not None and len(nc_ibis)  >= 4 and
            src_ibis is not None and len(src_ibis) >= 4):
        ibi_r, ibi_c = compute_ibi_ratio(nc_ibis, src_ibis)
        result.ibi_ratio = ibi_r
        result.ibi_ci    = ibi_c
        _log(f"  IBI ratio: {ibi_r:.6f}×  95% CI [{ibi_c[0]:.6f}, {ibi_c[1]:.6f}]")
    else:
        _log("  IBI ratio: insufficient beats — skipped")

    _log("Done.")
    return result
