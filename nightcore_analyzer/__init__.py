"""
Nightcore Audio Analysis Tool
==============================
Extracts tempo ratio and pitch ratio between a nightcore track and its FLAC
source with sub-decimal accuracy, then emits the exact Rubber Band parameters
needed to reconstruct the original.

Quick start (CLI):
    python -m nightcore_analyzer.cli \\
        --nightcore /path/to/nightcore.flac \\
        --source    /path/to/original.flac \\
        --output    results.json

Quick start (Python API):
    from nightcore_analyzer import run
    result = run("nightcore.flac", "source.flac")
    print(result)
"""

from .pipeline import run
from .consensus import AnalysisResult
from . import export
from . import session

__version__ = "0.3.0"
__all__ = ["run", "AnalysisResult", "export", "session"]
