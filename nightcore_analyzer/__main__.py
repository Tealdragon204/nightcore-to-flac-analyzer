"""
Entry point for ``python -m nightcore_analyzer``.

GUI mode is implemented in Step 3.  Until then this redirects to the CLI.
"""
import sys

print(
    "GUI mode is not yet implemented (Step 3 — coming soon).\n"
    "\n"
    "Use the CLI instead:\n"
    "  python -m nightcore_analyzer.cli \\\n"
    "      --nightcore /path/to/nightcore.flac \\\n"
    "      --source    /path/to/original.flac \\\n"
    "      --output    results.json\n"
    "\n"
    "  python -m nightcore_analyzer.cli --help",
    file=sys.stderr,
)
sys.exit(1)
