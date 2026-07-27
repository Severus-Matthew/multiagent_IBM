from __future__ import annotations

import json
from typing import Any


CANONICAL_ACTION_STRATEGIES = [
    "auto",
    "minimal",
    "restart_first",
    "rollback_first",
    "scale_first",
    "infra_patch_first",
]


class StructuredActionPromptPolicy:
    """Structured prompt optimizer for the fixed ActionAgent.

    This is the active non-debug action prompt policy. It does not execute
    commands and does not see oracle labels. It turns a verified RCA result,
    redacted state abstraction, current SLA summary, and public attempt history
    into a concrete instruction prompt plus a machine-readable action plan.

    For GRPO rollouts, context["sample_index"] selects a different safe action
    family variant for the same verified RCA state. A trainable Qwen/LoRA policy
    can later replace `generate()` while keeping the same ActionAgent contract and
    reward/verifier path.
    """

    def __init__(self, strategy: str = "auto", max_root_causes: int = 2):
        if strategy not in CANONICAL_ACTION_STRATEGIES:
            raise ValueError(f"unknown action strategy {strategy!r}; valid={CANONICAL_ACTION_STRATEGIES}")
        self.strategy = strategy
        self.max_root_causes = max(1, int(max_root_causes))

    def generate(self, context: dict[str, Any]) -> str:
        namespace = str(context.get("namespace") or "default")
        root_causes = _root_causes(context)[: self.max_root_causes]
        if not root_causes:
            root_causes = [{"service": "unknown", "fault_type": "unknown"}]
        current_sla = context.get("current_sla", {}) or {}
        previous_attempts = context.get("previous_attempts", []) or []
        iteration = int(context.get("iteration", 0) or 0)
        sample_index = int(context.get("sample_index", 0) or 0)

        plans = []
        for root in root_causes:
            service = str(root.get("service") or "unknown")
            fault_type = str(root.get("fault_type") or "unknown")
            action_family = _choose_action_family(fault_type, self.strategy, iteration, previous_attempts, sample_index)
            plans.append({
                "service": service,
                "fault_type": fault_type,
                "action_family": action_family,
                "namespace": namespace,
                "verification": "rollout_status_then_get",
            })

        payload = {
            "contract": "ACTION_PLAN_V1",
            "namespace": namespace,
            "strategy": self.strategy,
            "iteration": iteration,
            "sample_index": sample_index,
            "rca_twin_verified": bool((context.get("rca_twin_gate", {}) or {}).get("rca_twin_verified")),
            "current_sla": {
                "sla_restored": current_sla.get("sla_restored"),
                "hard_violations": current_sla.get("hard_violations"),
                "soft_violations": current_sla.get("soft_violations"),
                "weighted_violations": current_sla.get("weighted_violations"),
            },
            "plans": plans,
            "safety_rules": [
                "Use only scoped namespace commands.",
                "Output only kubectl, helm, or mongosh commands.",
                "Do not use exec, cp, proxy, port-forward, attach, debug, auth, create, apply, replace, edit, label, annotate.",
                "Do not use shell metacharacters, sudo, curl, wget, pipelines, broad deletes, -A, or --all-namespaces.",
                "Include a verification command after every mutation.",
            ],
        }

        human = [
            "You are the action prompt optimizer for a fixed Kubernetes ActionAgent.",
            "The RCA has already passed the RCA twin gate. Generate remediation instructions only for the verified RCA target(s).",
            "The ActionAgent must output executable commands only; no prose, markdown, comments, or code fences.",
            "Use the namespace exactly as given in the action plan.",
            "Prefer the smallest safe action that matches the RCA fault type and can be verified by rollout/status/get commands.",
            "Do not repair downstream victims unless they are explicitly listed as RCA root causes.",
            "",
            "Allowed action families:",
            "- infra_patch_first: patch invalid pod scheduling fields, restart deployment, verify rollout.",
            "- restart_service: rollout restart deployment, verify rollout, get deployment.",
            "- rollback_config: rollback/restart configuration-affecting service, verify status.",
            "- scale_service: scale deployment, verify rollout, get pods.",
            "- recreate_pod: delete only target-owned pods, verify rollout.",
            "",
            "ACTION_PLAN_JSON:",
            json.dumps(payload, sort_keys=True, default=str),
        ]
        return "\n".join(human)


def _root_causes(context: dict[str, Any]) -> list[dict[str, Any]]:
    rca = context.get("rca_result", {}) or {}
    roots = rca.get("root_causes") or context.get("rca_faults") or []
    out: list[dict[str, Any]] = []
    for item in roots:
        if isinstance(item, dict):
            svc = item.get("service") or item.get("root_cause_service")
            ft = item.get("fault_type") or item.get("fault_family") or "unknown"
            if svc:
                out.append({"service": str(svc), "fault_type": str(ft)})
    if not out and rca.get("root_cause_service"):
        out.append({
            "service": str(rca.get("root_cause_service")),
            "fault_type": str(rca.get("fault_type") or "unknown"),
        })
    return out


def _choose_action_family(
    fault_type: str,
    strategy: str,
    iteration: int,
    previous_attempts: list[dict[str, Any]],
    sample_index: int = 0,
) -> str:
    ft = str(fault_type or "unknown")
    if strategy == "infra_patch_first":
        return "infra_patch_first"
    if strategy == "restart_first":
        return "restart_service"
    if strategy == "rollback_first":
        return "rollback_config"
    if strategy == "scale_first":
        return "scale_service"

    retry = iteration > 0 or bool(previous_attempts)
    if ft == "infra_failure":
        families = ["infra_patch_first", "restart_service", "scale_service", "rollback_config"]
        if retry:
            families = ["restart_service", "infra_patch_first", "scale_service", "rollback_config"]
        return families[sample_index % len(families)]
    if ft in {"resource_exhaustion", "latency_degradation"}:
        families = ["scale_service", "restart_service", "rollback_config", "infra_patch_first"]
        if retry:
            families = ["restart_service", "scale_service", "rollback_config", "infra_patch_first"]
        return families[sample_index % len(families)]
    if ft in {"config_error", "auth_failure", "dependency_failure"}:
        families = ["rollback_config", "restart_service", "scale_service", "infra_patch_first"]
        if retry:
            families = ["restart_service", "rollback_config", "scale_service", "infra_patch_first"]
        return families[sample_index % len(families)]
    if ft == "network_failure":
        families = ["restart_service", "rollback_config", "scale_service", "infra_patch_first"]
        if retry:
            families = ["rollback_config", "restart_service", "scale_service", "infra_patch_first"]
        return families[sample_index % len(families)]
    families = ["restart_service", "rollback_config", "scale_service", "infra_patch_first"]
    return families[sample_index % len(families)]
