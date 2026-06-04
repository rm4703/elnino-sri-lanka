"""Tiny on-disk cache so we don't re-hit the public APIs on every Streamlit rerun."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _key(name: str, params: dict[str, Any]) -> Path:
    blob = name + "|" + json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.md5(blob.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{name}_{digest}.json"


def get(name: str, params: dict[str, Any], max_age_hours: float) -> Any | None:
    """Return cached payload if present and fresh, else ``None``."""
    path = _key(name, params)
    if not path.exists():
        return None
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    if age_h > max_age_hours:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def put(name: str, params: dict[str, Any], payload: Any) -> None:
    path = _key(name, params)
    try:
        path.write_text(json.dumps(payload))
    except OSError:
        pass  # cache is best-effort
