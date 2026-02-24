"""
Per-window estimate histogram widget (Step 4).

Displays four histograms in a 2×2 grid:
  - Source pitch distribution (Hz)
  - Nightcore pitch distribution (Hz)
  - Source tempo distribution (BPM)
  - Nightcore tempo distribution (BPM)

Each plot marks the median with a dashed vertical line.  When no analysis
result is loaded, a placeholder message is shown instead.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("QtAgg")  # must be set before any other matplotlib import

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from ..consensus import AnalysisResult


class HistogramWidget(QWidget):
    """Four per-window histograms embedded in a PyQt6 widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder shown when no result is loaded
        self._placeholder = QLabel("Run an analysis to see per-window distributions.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._placeholder)

        # Matplotlib figure (hidden until first result)
        self._fig = Figure(figsize=(8, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.hide()
        layout.addWidget(self._canvas)

    # ── public ────────────────────────────────────────────────────────────────

    def update_result(self, result: AnalysisResult) -> None:
        """Redraw all four histograms from *result*'s raw per-window data."""
        self._placeholder.hide()
        self._canvas.show()

        self._fig.clear()
        axes = self._fig.subplots(2, 2)

        _draw_histogram(
            ax=axes[0, 0],
            values=result.src_pitches_raw,
            title="Source — pitch",
            xlabel="Frequency (Hz)",
            color="#4c9be8",
        )
        _draw_histogram(
            ax=axes[0, 1],
            values=result.nc_pitches_raw,
            title="Nightcore — pitch",
            xlabel="Frequency (Hz)",
            color="#e8874c",
        )
        _draw_histogram(
            ax=axes[1, 0],
            values=result.src_tempos_raw,
            title="Source — tempo",
            xlabel="BPM",
            color="#4c9be8",
        )
        _draw_histogram(
            ax=axes[1, 1],
            values=result.nc_tempos_raw,
            title="Nightcore — tempo",
            xlabel="BPM",
            color="#e8874c",
        )

        self._canvas.draw()

    def clear(self) -> None:
        """Reset to the placeholder state (no data)."""
        self._canvas.hide()
        self._placeholder.show()


# ── helpers ───────────────────────────────────────────────────────────────────

def _draw_histogram(
    ax,
    values: Optional[List[Optional[float]]],
    title: str,
    xlabel: str,
    color: str,
) -> None:
    """Draw a histogram of valid values onto *ax*, with a median line."""
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("windows", fontsize=8)
    ax.tick_params(labelsize=7)

    if not values:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontstyle="italic")
        return

    valid = np.array([v for v in values if v is not None and np.isfinite(v) and v > 0],
                     dtype=np.float64)

    if len(valid) == 0:
        ax.text(0.5, 0.5, "no valid windows", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontstyle="italic")
        return

    bins = min(max(len(valid) // 2, 5), 30)
    ax.hist(valid, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

    median = float(np.median(valid))
    ax.axvline(median, color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"median {median:.2f}")
    ax.legend(fontsize=7, framealpha=0.6)

    n_total = len(values)
    n_valid = len(valid)
    ax.set_title(f"{title}  ({n_valid}/{n_total} windows)", fontsize=9)
