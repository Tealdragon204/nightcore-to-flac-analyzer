"""
Full windowed consensus pipeline.

Orchestrates audio loading → windowing → energy gating → pitch estimation →
tempo estimation → consensus → classification → Rubber Band parameters.
"""

from __future__ import annotations

from typing import Callable, Optional

from .io import (
    load_audio, slice_windows, energy_gate,
    WINDOW_SEC, HOP_SEC, ENERGY_GATE_DB,
)
from .pitch import batch_estimate_pitch
from .tempo import batch_estimate_tempo
from .consensus import build_result, AnalysisResult


def run(
    nightcore_path: str,
    source_path: str,
    *,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
    energy_gate_db: float = ENERGY_GATE_DB,
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
    _log("Estimating tempo (librosa)…")
    _log("  ← nightcore →")
    nc_tempos  = batch_estimate_tempo(nc_windows,  log=_log)
    _log("  ← source →")
    src_tempos = batch_estimate_tempo(src_windows, log=_log)

    # ── 6. consensus + classification ─────────────────────────────────────────
    _log("Computing consensus…")
    nc_duration  = len(nc_audio)  / sr
    src_duration = len(src_audio) / sr
    result = build_result(
        src_pitches, nc_pitches, src_tempos, nc_tempos,
        nc_duration=nc_duration, src_duration=src_duration,
    )

    _log("Done.")
    return result
