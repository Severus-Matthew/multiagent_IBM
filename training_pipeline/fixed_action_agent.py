from __future__ import annotations

import json
import re
from typing import Any


_ACTION_PLAN_RE = re.compile(r"ACTION_PLAN_JSON:\s*(\{[\s\S]*\})\s*$", re.MULTILINE)


class FixedActionAgent:
    """Fixed command generator for offline action-policy training.

    This is a real ActionAgent interface, not a debug test class: it consumes the
    structured action prompt contract emitted by StructuredActionPromptPolicy and
    deterministically emits scoped kubectl/helm/mongosh commands. Its weights and
    behavior are fixed; only the prompt policy is meant to be trained/optimized.

    A live LLM ActionAgent can be selected by the training script later, but this
    fixed agent is the safe default for reward development and offline rollouts.
    """

    def __init__(self, max_commands: int = 15):
        self.max_commands = max(1, int(max_commands))

    def get_commands(self, instruction_prompt: str, context: dict[str, Any]) -> list[str]:
        plan = _extract_plan(instruction_prompt) or _fallback_plan(context)
        namespace = str(plan.get("namespace") or context.get("namespace") or "default")
        commands: list[str] = []
        for item in plan.get("plans", []) or []:
            if not isinstance(item, dict):
                continue
            service = str(item.get("service") or "").strip()
            if not service or service == "unknown":
                continue
            action_family = str(item.get("action_family") or "restart_service")
            commands.extend(_commands_for_action_family(service, namespace, action_family))
            if len(commands) >= self.max_commands:
                break
        return _dedupe(commands)[: self.max_commands]


def _extract_plan(text: str) -> dict[str, Any] | None:
    match = _ACTION_PLAN_RE.search(str(text or ""))
    if not match:
        return None
    blob = match.group(1).strip()
    try:
        parsed = json.loads(blob)
    except Exception:
        return None
    if not isinstance(parsed, dict) or parsed.get("contract") != "ACTION_PLAN_V1":
        return None
    return parsed


def _fallback_plan(context: dict[str, Any]) -> dict[str, Any]:
    rca = context.get("rca_result", {}) or {}
    roots = rca.get("root_causes") or context.get("rca_faults") or []
    plans = []
    for item in roots:
        if not isinstance(item, dict):
            continue
        service = item.get("service") or item.get("root_cause_service")
        fault_type = item.get("fault_type") or item.get("fault_family") or "unknown"
        if service:
            plans.append({
                "service": str(service),
                "fault_type": str(fault_type),
                "action_family": _default_family(str(fault_type)),
            })
    if not plans and rca.get("root_cause_service"):
        ft = str(rca.get("fault_type") or "unknown")
        plans.append({
            "service": str(rca.get("root_cause_service")),
            "fault_type": ft,
            "action_family": _default_family(ft),
        })
    return {
        "contract": "ACTION_PLAN_V1",
        "namespace": context.get("namespace") or "default",
        "plans": plans,
    }


def _default_family(fault_type: str) -> str:
    ft = str(fault_type or "unknown")
    if ft == "infra_failure":
        return "infra_patch_first"
    if ft in {"resource_exhaustion", "latency_degradation"}:
        return "scale_service"
    if ft in {"config_error", "auth_failure", "dependency_failure"}:
        return "rollback_config"
    return "restart_service"


def _commands_for_action_family(service: str, namespace: str, action_family: str) -> list[str]:
    ns = namespace or "default"
    verify = f"kubectl rollout status deployment/{service} -n {ns} --timeout=120s"
    get_deploy = f"kubectl get deployment/{service} -n {ns}"

    if action_family == "infra_patch_first":
        patch_node_name = "'[{\"op\":\"remove\",\"path\":\"/spec/template/spec/nodeName\"}]'"
        return [
            f"kubectl patch deployment/{service} -n {ns} --type=json -p={patch_node_name}",
            f"kubectl rollout restart deployment/{service} -n {ns}",
            verify,
        ]

    if action_family == "recreate_pod":
        # Delete only pods selected by the target app label in the target namespace.
        # The safety gate will reject broad/cluster-wide deletes.
        return [
            f"kubectl delete pod -n {ns} -l app={service}",
            verify,
            f"kubectl get pods -n {ns}",
        ]

    if action_family == "scale_service":
        return [
            f"kubectl scale deployment/{service} -n {ns} --replicas=2",
            verify,
            f"kubectl get pods -n {ns}",
        ]

    if action_family == "rollback_config":
        # Prefer a deployment-scoped restart as the safe offline proxy for config,
        # auth, or dependency repairs when no release/configmap name is known.
        return [
            f"kubectl rollout restart deployment/{service} -n {ns}",
            verify,
            get_deploy,
        ]

    # restart_service and unknown safe default.
    return [
        f"kubectl rollout restart deployment/{service} -n {ns}",
        verify,
        get_deploy,
    ]


def _dedupe(commands: list[str]) -> list[str]:
    seen = set()
    out = []
    for cmd in commands:
        if cmd not in seen:
            seen.add(cmd)
            out.append(cmd)
    return out
