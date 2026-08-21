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

    if parts[:3] == ["kubectl", "rollout", "restart"]:
        svc = _deployment(parts[3:])
        return {"action": "restart_service", "service": svc, "raw": raw, "valid": bool(svc)}

    if parts[:3] == ["kubectl", "rollout", "status"]:
        svc = _deployment(parts[3:])
        return {"action": "verify", "service": svc, "raw": raw, "valid": True}

    if parts[:2] == ["kubectl", "scale"]:
        svc = _deployment(parts[2:])
        return {"action": "scale_service", "service": svc, "raw": raw, "valid": bool(svc)}

    if parts[:2] == ["helm", "rollback"]:
        return {"action": "rollback_config", "service": None, "raw": raw, "valid": len(parts) >= 3}

    if parts[:2] == ["kubectl", "patch"]:
        svc = _deployment(parts[2:]) or _configmap(parts[2:])
        if _looks_like_scheduling_repair(low):
            return {"action": "fix_infra_scheduling", "service": svc, "raw": raw, "valid": bool(svc)}
        return {"action": "rollback_config", "service": svc, "raw": raw, "valid": bool(svc)}

    if parts[:2] == ["kubectl", "delete"]:
        svc = _pod_owner_hint(parts) or _selector_service_hint(parts) or _deployment(parts[2:])
        return {"action": "recreate_pod", "service": svc, "raw": raw, "valid": bool(svc)}

    if parts[:2] == ["kubectl", "get"]:
        return {"action": "verify", "raw": raw, "valid": True}

    return {"action": "unknown", "raw": raw, "valid": False}


def _looks_like_scheduling_repair(low: str) -> bool:
    return any(x in low for x in ["nodename", "node-name", "node selector", "nodeselector", "affinity", "taint", "toleration"])


def _deployment(parts: list[str]) -> str | None:
    for p in parts:
        if p.startswith("deployment/"):
            return p.split("/", 1)[1]
        if p.startswith("deploy/"):
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


def _pod_owner_hint(parts: list[str]) -> str | None:
    for p in parts:
        if p.startswith("pod/"):
            name = p.split("/", 1)[1]
            return _service_from_pod_name(name)
    if len(parts) > 2 and parts[2] in ("pod", "pods", "po") and len(parts) > 3:
        candidate = parts[3]
        if not candidate.startswith("-"):
            return _service_from_pod_name(candidate)
    return None


def _selector_service_hint(parts: list[str]) -> str | None:
    """Recover a target from selector-scoped pod deletes such as -l app=svc."""
    selector = None
    for i, p in enumerate(parts):
        if p in {"-l", "--selector"} and i + 1 < len(parts):
            selector = parts[i + 1]
            break
        if p.startswith("--selector="):
            selector = p.split("=", 1)[1]
            break
        if p.startswith("-l="):
            selector = p.split("=", 1)[1]
            break
        if p.startswith("-l") and len(p) > 2:
            selector = p[2:].lstrip("=")
            break
    if not selector:
        return None
    for clause in str(selector).split(","):
        if "=" not in clause:
            continue
        key, value = clause.split("=", 1)
        if key.strip().lower() in {"app", "app.kubernetes.io/name", "service"} and value.strip():
            return value.strip()
    return None


def _service_from_pod_name(pod: str) -> str | None:
    chunks = pod.split("-")
    if len(chunks) >= 3 and chunks[-1].isalnum():
        return "-".join(chunks[:-2]) or pod
    return pod or None


def normalize_commands(commands: list[str]) -> list[dict[str, Any]]:
    return [normalize_command(c) for c in commands]
