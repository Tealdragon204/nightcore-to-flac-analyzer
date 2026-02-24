"""
Main application window (Step 3).

Layout
------
QMainWindow
├── menu bar  (File | Help)
├── central widget  (QSplitter, horizontal)
│   ├── left pane  (~340 px)
│   │   ├── "Input Files" group    — nightcore + source pickers
│   │   ├── "Parameters" group     — window / hop / energy-gate spinboxes
│   │   ├── [Run Analysis] button
│   │   └── Log area               — QPlainTextEdit, read-only
│   └── right pane
│       └── QTabWidget
│           ├── "Results"     tab  — ResultsPanel
│           └── "Histograms"  tab  — HistogramWidget
└── status bar
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QGroupBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSplitter, QStatusBar,
    QTabWidget, QVBoxLayout, QWidget, QFormLayout,
)

from ..consensus import AnalysisResult
from .. import export as export_module
from .. import session
from ..io import WINDOW_SEC, HOP_SEC, ENERGY_GATE_DB
from .worker import AnalysisWorker
from .results_panel import ResultsPanel
from .histogram_widget import HistogramWidget

_AUDIO_FILTER = "Audio files (*.flac *.mp3 *.wav *.ogg *.aac *.m4a);;All files (*)"
_JSON_FILTER  = "JSON (*.json);;All files (*)"
_CSV_FILTER   = "CSV (*.csv);;All files (*)"


class MainWindow(QMainWindow):
    """Top-level application window for the Nightcore Analyzer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Nightcore Analyzer")
        self.resize(1100, 680)

        self._worker: AnalysisWorker | None = None
        self._last_result: AnalysisResult | None = None

        self._build_menu()
        self._build_ui()
        self._restore_session()

    # ── menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu = self.menuBar()

        # File
        file_menu = menu.addMenu("&File")

        save_json_act = QAction("Save results as JSON…", self)
        save_json_act.setShortcut("Ctrl+S")
        save_json_act.triggered.connect(self._save_json)
        file_menu.addAction(save_json_act)

        save_csv_act = QAction("Save results as CSV…", self)
        save_csv_act.triggered.connect(self._save_csv)
        file_menu.addAction(save_csv_act)

        file_menu.addSeparator()

        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Help
        help_menu = menu.addMenu("&Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ── left pane ─────────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(400)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # Input files
        files_group = QGroupBox("Input Files")
        files_form = QFormLayout(files_group)

        self._nc_edit = QLineEdit()
        self._nc_edit.setPlaceholderText("Nightcore audio file…")
        nc_row = _file_row(self._nc_edit, self._browse_nightcore)
        files_form.addRow("Nightcore:", nc_row)

        self._src_edit = QLineEdit()
        self._src_edit.setPlaceholderText("Source FLAC…")
        src_row = _file_row(self._src_edit, self._browse_source)
        files_form.addRow("Source:", src_row)

        left_layout.addWidget(files_group)

        # Parameters
        params_group = QGroupBox("Analysis Parameters")
        params_form = QFormLayout(params_group)

        self._window_spin = QDoubleSpinBox()
        self._window_spin.setRange(1.0, 120.0)
        self._window_spin.setSingleStep(1.0)
        self._window_spin.setSuffix(" s")
        self._window_spin.setValue(WINDOW_SEC)

        self._hop_spin = QDoubleSpinBox()
        self._hop_spin.setRange(0.5, 60.0)
        self._hop_spin.setSingleStep(0.5)
        self._hop_spin.setSuffix(" s")
        self._hop_spin.setValue(HOP_SEC)

        self._gate_spin = QDoubleSpinBox()
        self._gate_spin.setRange(-120.0, 0.0)
        self._gate_spin.setSingleStep(5.0)
        self._gate_spin.setSuffix(" dB")
        self._gate_spin.setValue(ENERGY_GATE_DB)

        params_form.addRow("Window:", self._window_spin)
        params_form.addRow("Hop:", self._hop_spin)
        params_form.addRow("Energy gate:", self._gate_spin)

        left_layout.addWidget(params_group)

        # Run button
        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#2980b9; color:#fff; border-radius:4px; font-weight:bold; }"
            "QPushButton:hover { background:#3498db; }"
            "QPushButton:disabled { background:#7f8c8d; }"
        )
        self._run_btn.clicked.connect(self._run_analysis)
        left_layout.addWidget(self._run_btn)

        # Log area
        log_label = QLabel("Log:")
        left_layout.addWidget(log_label)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(4000)
        self._log.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:monospace; font-size:11px;"
        )
        left_layout.addWidget(self._log, stretch=1)

        splitter.addWidget(left)

        # ── right pane ────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._results_panel  = ResultsPanel()
        self._histogram      = HistogramWidget()
        self._tabs.addTab(self._results_panel, "Results")
        self._tabs.addTab(self._histogram,     "Histograms")
        splitter.addWidget(self._tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ── session restore ───────────────────────────────────────────────────────

    def _restore_session(self) -> None:
        self._nc_edit.setText(session.get("last_nightcore", ""))
        self._src_edit.setText(session.get("last_source", ""))
        self._window_spin.setValue(session.get("window_sec", WINDOW_SEC))
        self._hop_spin.setValue(session.get("hop_sec", HOP_SEC))
        self._gate_spin.setValue(session.get("energy_gate_db", ENERGY_GATE_DB))

    def _save_session(self) -> None:
        session.set_many({
            "last_nightcore":  self._nc_edit.text(),
            "last_source":     self._src_edit.text(),
            "window_sec":      self._window_spin.value(),
            "hop_sec":         self._hop_spin.value(),
            "energy_gate_db":  self._gate_spin.value(),
        })

    # ── file pickers ──────────────────────────────────────────────────────────

    def _browse_nightcore(self) -> None:
        start = _start_dir(self._nc_edit.text())
        path, _ = QFileDialog.getOpenFileName(self, "Select nightcore file", start, _AUDIO_FILTER)
        if path:
            self._nc_edit.setText(path)

    def _browse_source(self) -> None:
        start = _start_dir(self._src_edit.text())
        path, _ = QFileDialog.getOpenFileName(self, "Select source file", start, _AUDIO_FILTER)
        if path:
            self._src_edit.setText(path)

    # ── analysis ──────────────────────────────────────────────────────────────

    def _run_analysis(self) -> None:
        nc_path  = self._nc_edit.text().strip()
        src_path = self._src_edit.text().strip()

        errors = []
        if not nc_path:
            errors.append("No nightcore file selected.")
        elif not Path(nc_path).exists():
            errors.append(f"Nightcore file not found:\n  {nc_path}")
        if not src_path:
            errors.append("No source file selected.")
        elif not Path(src_path).exists():
            errors.append(f"Source file not found:\n  {src_path}")

        hop = self._hop_spin.value()
        win = self._window_spin.value()
        if hop >= win:
            errors.append("Hop must be less than Window.")

        if errors:
            QMessageBox.warning(self, "Input error", "\n\n".join(errors))
            return

        self._save_session()
        self._log.clear()
        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")
        self._status.showMessage("Analysis running…")
        self._results_panel.clear()
        self._histogram.clear()

        self._worker = AnalysisWorker(
            nightcore_path=nc_path,
            source_path=src_path,
            window_sec=win,
            hop_sec=hop,
            energy_gate_db=self._gate_spin.value(),
        )
        self._worker.log_line.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, payload: object) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run Analysis")

        if isinstance(payload, Exception):
            self._status.showMessage("Analysis failed.")
            self._append_log(f"\nERROR: {payload}")
            QMessageBox.critical(self, "Analysis failed", str(payload))
            return

        result: AnalysisResult = payload
        self._last_result = result
        self._status.showMessage("Analysis complete.")
        self._results_panel.update_result(result)
        self._histogram.update_result(result)
        self._tabs.setCurrentIndex(0)

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    # ── save results ──────────────────────────────────────────────────────────

    def _save_json(self) -> None:
        if not self._last_result:
            QMessageBox.information(self, "No results", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", _JSON_FILTER)
        if path:
            export_module.export_json(self._last_result, path)
            self._status.showMessage(f"Saved: {path}")

    def _save_csv(self) -> None:
        if not self._last_result:
            QMessageBox.information(self, "No results", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", _CSV_FILTER)
        if path:
            export_module.export_csv(self._last_result, path)
            self._status.showMessage(f"Saved: {path}")

    # ── about ─────────────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        from .. import __version__
        QMessageBox.about(
            self,
            "About Nightcore Analyzer",
            f"<b>Nightcore Analyzer</b> v{__version__}<br><br>"
            "Algorithmic extraction of the precise tempo ratio and pitch ratio "
            "between a nightcore track and its FLAC source.<br><br>"
            "<a href='https://github.com/Tealdragon204/nightcore-to-flac-analyzer'>"
            "github.com/Tealdragon204/nightcore-to-flac-analyzer</a>",
        )

    # ── close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._save_session()
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
        super().closeEvent(event)


# ── helpers ───────────────────────────────────────────────────────────────────

def _file_row(edit: QLineEdit, browse_fn) -> QWidget:
    """Return a widget containing *edit* and a Browse button."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, stretch=1)
    btn = QPushButton("Browse…")
    btn.setFixedWidth(72)
    btn.clicked.connect(browse_fn)
    h.addWidget(btn)
    return w


def _start_dir(path: str) -> str:
    """Return the parent directory of *path*, or '' if *path* is empty."""
    if path:
        p = Path(path)
        if p.parent.exists():
            return str(p.parent)
    return ""
