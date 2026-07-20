from __future__ import annotations

import shlex
from typing import Any


def normalize_command(cmd: str) -> dict[str, Any]:
    raw = str(cmd or "").strip()
    try:
        parts = shlex.split(raw)
    except Exception:
        parts = raw.split()
    if not parts:
        return {"action": "invalid", "raw": raw, "valid": False}

    low = raw.lower()

    # kubectl rollout restart deployment/<svc> ...
    if parts[:3] == ["kubectl", "rollout", "restart"]:
        svc = _deployment(parts[3:])
        return {"action": "restart_service", "service":