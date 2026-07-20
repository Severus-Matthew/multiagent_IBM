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
    if parts[:3] == ["kubectl", "rollout", "restart"]:
        svc = _deployment(parts[3:])
        return {"action": "restart_service", "service": svc, "raw": raw, "valid": bool(svc)}
    if parts[:2] == ["kubectl", "scale"]:
        svc = _deployment(parts[2:])
        return {"action": "scale_service", "service": svc, "raw": raw, "valid": bool(svc)}
    if parts[:2] == ["helm", "rollback"]:
        return {"action": "rollback_config", "service": None, "raw": raw, "valid": len(parts) >= 3}
    if parts[:2] == ["kubectl", "patch"]:
        svc = _deployment(parts[2:]) or _configmap(parts[2:])
        return {"action": "rollback_config", "service": svc, "raw": raw, "valid": bool(svc)}
    if parts[:2] == ["kubectl", "get"] or parts[:3] == ["kubectl", "rollout", "status"]:
        return {"action": "verify", "raw": raw, "valid": True}
    return {"action": "unknown", "raw": raw, "valid": False}


def _deployment(parts: list[str]) -> str | None:
    for p in parts:
        if p.startswith("deployment/"):
            return p.split("/", 1)[1]
    if parts and parts[0] in ("deployment", "deploy") and len(parts) > 1:
        return parts[1]
    return None


def _configmap(parts: list[str]) -> str | None:
    for i, p in enumerate(parts):
        if p.startswith("configmap/"):
            return p.split("/", 1)[1]
        if p in ("configmap", "cm") and i + 1 < len(parts):
            return parts[i + 1]
    return None


def normalize_commands(commands: list[str]) -> list[dict[str, Any]]:
    return [normalize_command(c) for c in commands]
