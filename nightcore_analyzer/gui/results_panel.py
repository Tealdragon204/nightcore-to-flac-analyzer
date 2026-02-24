"""
Results panel widget (Step 5).

Displays the full AnalysisResult:
  - Classification badge (colour-coded)
  - Tempo ratio + 95 % CI + window counts
  - Pitch ratio + 95 % CI + window counts
  - Rubber Band reconstruction parameters with a copy-to-clipboard button
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QFormLayout, QPushButton, QApplication, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..consensus import AnalysisResult

# Classification → (display text, background colour, text colour)
_CLASSIFICATION_STYLE: dict[str, tuple[str, str, str]] = {
    "pure_nightcore":          ("Pure Nightcore",            "#2ecc71", "#fff"),
    "independent_pitch_shift": ("Independent Pitch Shift",   "#e67e22", "#fff"),
    "time_stretch_only":       ("Time Stretch Only",         "#3498db", "#fff"),
    "ambiguous":               ("Ambiguous",                 "#95a5a6", "#fff"),
}


class ResultsPanel(QWidget):
    """Read-only panel showing the output of one analysis run."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── placeholder ───────────────────────────────────────────────────────
        self._placeholder = QLabel("Run an analysis to see results here.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-style: italic;")
        root.addWidget(self._placeholder)
        root.addStretch()

        # ── results container (hidden until first result) ─────────────────────
        self._results_widget = QWidget()
        self._results_widget.hide()
        results_layout = QVBoxLayout(self._results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)
        root.addWidget(self._results_widget)

        # Classification badge
        self._badge = QLabel()
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_font = QFont()
        badge_font.setBold(True)
        badge_font.setPointSize(11)
        self._badge.setFont(badge_font)
        self._badge.setFixedHeight(36)
        results_layout.addWidget(self._badge)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        results_layout.addWidget(sep)

        # Tempo group
        tempo_group = QGroupBox("Tempo Ratio")
        tempo_form = QFormLayout(tempo_group)
        self._tempo_ratio  = QLabel()
        self._tempo_ci     = QLabel()
        self._tempo_wins   = QLabel()
        tempo_form.addRow("Ratio:",   self._tempo_ratio)
        tempo_form.addRow("95 % CI:", self._tempo_ci)
        tempo_form.addRow("Windows:", self._tempo_wins)
        results_layout.addWidget(tempo_group)

        # Pitch group
        pitch_group = QGroupBox("Pitch Ratio")
        pitch_form = QFormLayout(pitch_group)
        self._pitch_ratio  = QLabel()
        self._pitch_ci     = QLabel()
        self._pitch_wins   = QLabel()
        pitch_form.addRow("Ratio:",   self._pitch_ratio)
        pitch_form.addRow("95 % CI:", self._pitch_ci)
        pitch_form.addRow("Windows:", self._pitch_wins)
        results_layout.addWidget(pitch_group)

        # Rubber Band group
        rb_group = QGroupBox("Rubber Band Parameters")
        rb_layout = QVBoxLayout(rb_group)
        rb_form = QFormLayout()
        self._rb_time    = QLabel()
        self._rb_pitch   = QLabel()
        rb_form.addRow("--time:",  self._rb_time)
        rb_form.addRow("--pitch:", self._rb_pitch)
        rb_layout.addLayout(rb_form)

        cmd_row = QHBoxLayout()
        self._rb_cmd = QLabel()
        self._rb_cmd.setWordWrap(True)
        self._rb_cmd.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._rb_cmd.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:monospace;"
            "padding:4px; border-radius:3px;"
        )
        cmd_row.addWidget(self._rb_cmd, stretch=1)

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(56)
        copy_btn.setToolTip("Copy Rubber Band command to clipboard")
        copy_btn.clicked.connect(self._copy_command)
        cmd_row.addWidget(copy_btn)
        rb_layout.addLayout(cmd_row)

        results_layout.addWidget(rb_group)
        results_layout.addStretch()

    # ── public ────────────────────────────────────────────────────────────────

    def update_result(self, result: AnalysisResult) -> None:
        """Populate all fields from *result* and make the panel visible."""
        self._placeholder.hide()
        self._results_widget.show()

        # Badge
        label, bg, fg = _CLASSIFICATION_STYLE.get(
            result.classification,
            (result.classification, "#95a5a6", "#fff"),
        )
        self._badge.setText(label)
        self._badge.setStyleSheet(
            f"background-color:{bg}; color:{fg}; border-radius:4px;"
        )

        # Tempo
        self._tempo_ratio.setText(f"{result.tempo_ratio:.6f}")
        lo, hi = result.tempo_ci
        self._tempo_ci.setText(f"[{lo:.6f},  {hi:.6f}]")
        self._tempo_wins.setText(
            f"{result.n_source_tempo_windows} source  /  "
            f"{result.n_nc_tempo_windows} nightcore"
        )

        # Pitch
        self._pitch_ratio.setText(f"{result.pitch_ratio:.6f}")
        lo, hi = result.pitch_ci
        self._pitch_ci.setText(f"[{lo:.6f},  {hi:.6f}]")
        self._pitch_wins.setText(
            f"{result.n_source_pitch_windows} source  /  "
            f"{result.n_nc_pitch_windows} nightcore"
        )

        # Rubber Band
        rb = result.rubberband
        self._rb_time.setText(str(rb.get("time_ratio", "")))
        self._rb_pitch.setText(f"{rb.get('pitch_semitones', '')} st")
        self._rb_cmd.setText(rb.get("cli_command", ""))

    def clear(self) -> None:
        """Reset to placeholder state."""
        self._results_widget.hide()
        self._placeholder.show()

    # ── private ───────────────────────────────────────────────────────────────

    def _copy_command(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._rb_cmd.text())
