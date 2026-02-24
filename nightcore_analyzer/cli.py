"""
Command-line interface for the Nightcore Analyzer.

Usage
-----
python -m nightcore_analyzer.cli \\
    --nightcore /path/to/nightcore.flac \\
    --source    /path/to/original.flac \\
    --output    results.json

python -m nightcore_analyzer.cli --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline
from .io import WINDOW_SEC, HOP_SEC, ENERGY_GATE_DB


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nightcore_analyzer.cli",
        description=(
            "Extract the precise tempo ratio and pitch ratio between a nightcore "
            "track and its FLAC source, then emit the Rubber Band parameters "
            "needed to reconstruct the original."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required inputs ───────────────────────────────────────────────────────
    p.add_argument(
        "--nightcore", "-n",
        required=True,
        metavar="FILE",
        help="Nightcore audio file (any format librosa supports: FLAC, MP3, WAV, …)",
    )
    p.add_argument(
        "--source", "-s",
        required=True,
        metavar="FILE",
        help="Source FLAC (the original from which the nightcore was derived)",
    )

    # ── optional output ───────────────────────────────────────────────────────
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write JSON results to this file (default: print to stdout)",
    )

    # ── pipeline tunables ─────────────────────────────────────────────────────
    p.add_argument(
        "--window",
        type=float,
        default=WINDOW_SEC,
        metavar="SEC",
        help="Analysis window duration in seconds",
    )
    p.add_argument(
        "--hop",
        type=float,
        default=HOP_SEC,
        metavar="SEC",
        help="Hop between consecutive windows in seconds (< --window for overlap)",
    )
    p.add_argument(
        "--energy-gate",
        type=float,
        default=ENERGY_GATE_DB,
        metavar="DB",
        help=(
            "Discard windows whose RMS energy is below peak + ENERGY_GATE dB.  "
            "Use a more negative value (e.g. -60) to keep quieter sections."
        ),
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output (errors still go to stderr)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    nc_path  = Path(args.nightcore)
    src_path = Path(args.source)

    # ── validate inputs ───────────────────────────────────────────────────────
    errors = []
    if not nc_path.exists():
        errors.append(f"Nightcore file not found: {nc_path}")
    if not src_path.exists():
        errors.append(f"Source file not found:    {src_path}")
    if args.hop >= args.window:
        errors.append("--hop must be less than --window for overlapping windows")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2

    log = None if args.quiet else print

    # ── run pipeline ──────────────────────────────────────────────────────────
    try:
        result = pipeline.run(
            str(nc_path),
            str(src_path),
            window_sec=args.window,
            hop_sec=args.hop,
            energy_gate_db=args.energy_gate,
            log=log,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    # ── format output ─────────────────────────────────────────────────────────
    output = {
        "classification": result.classification,
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
        "rubberband": result.rubberband,
    }

    json_text = json.dumps(output, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json_text, encoding="utf-8")
        if not args.quiet:
            print(f"\nResults written to: {out_path}")
    else:
        print()
        print(json_text)

    # Always print the human-readable summary unless --quiet
    if not args.quiet:
        print()
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
