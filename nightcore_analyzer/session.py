"""
Session persistence for the Nightcore Analyzer GUI.

Saves lightweight state (last-used file paths, window geometry, parameter
values) to a JSON file in the user's home directory so they persist across
application launches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SESSION_FILE = Path.home() / ".nightcore_analyzer_session.json"


def _load_raw() -> dict:
    try:
        return json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get(key: str, default: Any = None) -> Any:
    """Return the stored value for *key*, or *default* if absent."""
    return _load_raw().get(key, default)


def set(key: str, value: Any) -> None:
    """Persist *value* under *key*."""
    data = _load_raw()
    data[key] = value
    _SESSION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_many(updates: dict) -> None:
    """Persist all key-value pairs in *updates* atomically."""
    data = _load_raw()
    data.update(updates)
    _SESSION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
