"""
Export AnalysisResult to JSON or CSV.

JSON format mirrors the output of the CLI (nightcore_analyzer.cli).
CSV writes a single-row spreadsheet with one column per field.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from .consensus import AnalysisResult

PathLike = Union[str, Path]


def to_dict(result: AnalysisResult) -> dict:
    """Convert *result* to a JSON-serialisable dict (same format as the CLI)."""
    return {
        "classification": result.classification,
        "warnings":       result.warnings,
        "tempo_ratio":    round(result.tempo_ratio, 8),
        "pitch_ratio":    round(result.pitch_ratio, 8),
        "tempo_ci_95":    [round(result.tempo_ci[0], 8), round(result.tempo_ci[1], 8)],
        "pitch_ci_95":    [round(result.pitch_ci[0], 8), round(result.pitch_ci[1], 8)],
        "windows_used": {
            "source_pitch":    result.n_source_pitch_windows,
            "nightcore_pitch": result.n_nc_pitch_windows,
            "source_tempo":    result.n_source_tempo_windows,
            "nightcore_tempo": result.n_nc_tempo_windows,
        },
        "rubberband": result.rubberband,  # includes duration_* keys when durations available
        "durations": {
            "nightcore_sec":  round(result.nc_duration,  3) if result.nc_duration  else None,
            "source_sec":     round(result.src_duration, 3) if result.src_duration else None,
            "duration_ratio": (
                round(result.src_duration / result.nc_duration, 8)
                if result.nc_duration and result.src_duration else None
            ),
        },
        "median_bpms": {
            "nightcore": round(result.nc_median_bpm,  2) if result.nc_median_bpm  else None,
            "source":    round(result.src_median_bpm, 2) if result.src_median_bpm else None,
        },
    }


def export_json(result: AnalysisResult, path: PathLike) -> None:
    """Write *result* as formatted JSON to *path*."""
    Path(path).write_text(json.dumps(to_dict(result), indent=2), encoding="utf-8")


def export_csv(result: AnalysisResult, path: PathLike) -> None:
    """
    Write *result* as a single-row CSV to *path*.

    The CSV has a header row followed by one data row.  Nested fields
    (CI bounds, window counts, Rubber Band parameters) are flattened into
    individual columns.
    """
    rb = result.rubberband
    row = {
        "classification":          result.classification,
        "tempo_ratio":             round(result.tempo_ratio, 8),
        "pitch_ratio":             round(result.pitch_ratio, 8),
        "tempo_ci_95_lo":          round(result.tempo_ci[0], 8),
        "tempo_ci_95_hi":          round(result.tempo_ci[1], 8),
        "pitch_ci_95_lo":          round(result.pitch_ci[0], 8),
        "pitch_ci_95_hi":          round(result.pitch_ci[1], 8),
        "source_pitch_windows":    result.n_source_pitch_windows,
        "nightcore_pitch_windows": result.n_nc_pitch_windows,
        "source_tempo_windows":    result.n_source_tempo_windows,
        "nightcore_tempo_windows": result.n_nc_tempo_windows,
        "rb_time_ratio":           rb.get("time_ratio", ""),
        "rb_pitch_semitones":      rb.get("pitch_semitones", ""),
        "rb_nc_to_source_speed":   rb.get("nc_to_source_speed", ""),
        "rb_cli_command":          rb.get("cli_command", ""),
        "rb_dur_time_ratio":       rb.get("duration_time_ratio", ""),
        "rb_dur_pitch_semitones":  rb.get("duration_pitch_semitones", ""),
        "rb_dur_cli_command":      rb.get("duration_cli_command", ""),
        "nc_median_bpm":           round(result.nc_median_bpm,  2) if result.nc_median_bpm  else "",
        "src_median_bpm":          round(result.src_median_bpm, 2) if result.src_median_bpm else "",
        "nc_duration_sec":         round(result.nc_duration,  3) if result.nc_duration  else "",
        "src_duration_sec":        round(result.src_duration, 3) if result.src_duration else "",
        "duration_ratio": (
            round(result.src_duration / result.nc_duration, 8)
            if result.nc_duration and result.src_duration else ""
        ),
        "warnings":                " | ".join(result.warnings),
    }

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
