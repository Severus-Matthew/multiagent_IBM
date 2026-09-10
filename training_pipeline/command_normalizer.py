from __future__ import annotations

import shlex
from typing import Any

from .kubectl_command_shape import positional_args, positional_indices, resource_target


def normalize_command(cmd: str) -> dict[str, Any]:
    """Classify a single command's remediation action.

    Dispatches on the verb's actual position (``positional_indices``) rather
    than a fixed ``parts[:N]`` prefix — a global flag placed before the verb
    (``kubectl -n ns patch ...``, valid kubectl syntax) previously made every
    branch below fall through to "unknown".
    """
    raw = str(cmd or "").strip()
    try:
        parts = shlex.split(raw)
    except Exception:
        parts = raw.split()
    if not parts:
        return {"action": "invalid", "raw": raw, "valid": False}

    low = raw.lower()
    program = parts[0]
    indices = positional_indices(parts, 1)

    if program == "kubectl" and indices:
        verb = parts[indices[0]]
        if verb == "rollout" and len(indices) > 1:
            subverb = parts[indices[1]]
            rest = parts[indices[1] + 1:]
            if subverb == "restart":
                svc = _deployment(rest)
                return {"action": "restart_service", "service": svc, "raw": raw, "valid": bool(svc)}
            if subverb == "undo":
                svc = _deployment(rest)
                return {"action": "rollback_config", "service": svc, "raw": raw, "valid": bool(svc)}
            if subverb == "status":
                svc = _deployment(rest)
                return {"action": "verify", "service": svc, "raw": raw, "valid": True}
        rest = parts[indices[0] + 1:]
        if verb == "scale":
            svc = _deployment(rest)
            return {"action": "scale_service", "service": svc, "raw": raw, "valid": bool(svc)}
        if verb == "patch":
            svc = _deployment(rest) or _service(rest) or _configmap(rest)
            if _looks_like_scheduling_repair(low):
                return {"action": "fix_infra_scheduling", "service": svc, "raw": raw, "valid": bool(svc)}
            return {"action": "rollback_config", "service": svc, "raw": raw, "valid": bool(svc)}
        if verb == "delete":
            target = resource_target([verb, *rest])
            if target and target[0] in {
                "networkchaos", "podchaos", "stresschaos",
            }:
                return {
                    "action": "remove_fault_resource",
                    "service": None,
                    "resource_kind": target[0],
                    "resource_name": target[1],
                    "raw": raw,
                    "valid": bool(target[1]),
                }
            svc = _pod_owner_hint(rest) or _selector_service_hint(parts) or _deployment(rest)
            return {"action": "recreate_pod", "service": svc, "raw": raw, "valid": bool(svc)}
        if verb == "get":
            return {"action": "verify", "raw": raw, "valid": True}

    if program == "helm" and indices:
        verb = parts[indices[0]]
        if verb == "rollback":
            return {
                "action": "rollback_config", "service": None, "raw": raw,
                "valid": len(parts) >= indices[0] + 2,
            }

    return {"action": "unknown", "raw": raw, "valid": False}


def _looks_like_scheduling_repair(low: str) -> bool:
    return any(x in low for x in ["nodename", "node-name", "node selector", "nodeselector", "affinity", "taint", "toleration"])


def _deployment(parts: list[str]) -> str | None:
    for p in parts:
        if p.startswith("deployment/"):
            return p.split("/", 1)[1]
        if p.startswith("deploy/"):
            return p.split("/", 1)[1]
    # A flag (e.g. -n ns) may precede "deployment"/"deploy" in this slice too.
    positional = positional_args(parts, 0)
    if len(positional) >= 2 and positional[0] in ("deployment", "deploy"):
        return positional[1]
    return None


def _configmap(parts: list[str]) -> str | None:
    for i, p in enumerate(parts):
        if p.startswith("configmap/"):
            return p.split("/", 1)[1]
        if p in ("configmap", "cm") and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _service(parts: list[str]) -> str | None:
    for i, p in enumerate(parts):
        if p.startswith("service/") or p.startswith("svc/"):
            return p.split("/", 1)[1]
        if p in ("service", "svc") and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _pod_owner_hint(rest: list[str]) -> str | None:
    """``rest`` is the tokens after the "delete" verb (see call site)."""
    for p in rest:
        if p.startswith("pod/"):
            name = p.split("/", 1)[1]
            return _service_from_pod_name(name)
    positional = positional_args(rest, 0)
    if len(positional) >= 2 and positional[0] in ("pod", "pods", "po"):
        return _service_from_pod_name(positional[1])
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
