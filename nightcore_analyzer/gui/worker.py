"""
QThread worker that runs the analysis pipeline off the GUI thread.

Emits ``log_line`` for each progress message and ``finished`` when the
pipeline completes (with either an AnalysisResult or an Exception).
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from .. import pipeline
from ..consensus import AnalysisResult


class AnalysisWorker(QThread):
    """Run ``pipeline.run()`` in a background thread."""

    log_line = pyqtSignal(str)
    """Emitted for each progress message produced by the pipeline."""

    finished = pyqtSignal(object)
    """
    Emitted when the pipeline finishes.  The payload is either an
    :class:`~nightcore_analyzer.consensus.AnalysisResult` on success or an
    :class:`Exception` on failure.
    """

    def __init__(
        self,
        nightcore_path: str,
        source_path: str,
        window_sec: float,
        hop_sec: float,
        energy_gate_db: float,
    ) -> None:
        super().__init__()
        self.nightcore_path  = nightcore_path
        self.source_path     = source_path
        self.window_sec      = window_sec
        self.hop_sec         = hop_sec
        self.energy_gate_db  = energy_gate_db

    def run(self) -> None:
        try:
            result: AnalysisResult = pipeline.run(
                self.nightcore_path,
                self.source_path,
                window_sec=self.window_sec,
                hop_sec=self.hop_sec,
                energy_gate_db=self.energy_gate_db,
                log=self.log_line.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(exc)
