from __future__ import annotations

from typing import Any

from .kubectl_command_shape import positional_args, resource_target, split_command


def action_reward(
    commands: list[str],
    safety: dict[str, Any],
    verifier_result: dict[str, Any],
    instruction_tokens: int = 0,
    iteration_index: int = 0,
) -> dict[str, Any]:
    """Reward an action prompt/command attempt.

    The reward is now tied to three layers:
      1. command safety and parsability;
      2. behavioral twin outcome for the RCA-targeted service;
      3. SLA-style before/after symptom reduction.
    """
    safe = bool(safety.get("safe"))
    resolved = bool(verifier_result.get("resolved"))
    twin_resolved = bool(verifier_result.get("twin_resolved", resolved))
    sla_restored = bool(verifier_result.get("sla_restored", False))
    target_sla_restored = bool(verifier_result.get("target_sla_restored", False))
    global_reduction = _clamp01(verifier_result.get("global_symptom_reduction", verifier_result.get("symptom_reduction", 0.0)))
    target_reduction = _clamp01(verifier_result.get("target_symptom_reduction", 0.0))
    action_repairs = bool(verifier_result.get("action_repairs_fault_type", False))
    has_verify = any("rollout status" in c or "kubectl get" in c or "helm status" in c for c in commands)
    has_mutation = any(_is_mutating_command(c) for c in commands)

    reward_route = str(verifier_result.get("reward_route") or "unknown")
    telemetry_incomplete = bool(verifier_result.get("telemetry_incomplete"))
    after_state_observed = verifier_result.get("after_state_observed")
    if after_state_observed is None:
        after_state_observed = (verifier_result.get("resolution") or {}).get("after_state_observed")
    observable_improvement = bool(
        target_reduction > 0.0 or global_reduction > 0.0
        or target_sla_restored or sla_restored
    )
    credit_eligible = bool(
        reward_route == "live"
        and not telemetry_incomplete
        and after_state_observed is True
        and safe
        and has_mutation
        and observable_improvement
    )

    reward = 0.0
    reward += -1.25 if not safe else 0.0
    reward += -0.10 if not has_verify else 0.0
    reward += -0.20 if not has_mutation else 0.0
    if credit_eligible:
        reward += 0.75 if action_repairs else 0.0
        reward += 1.25 * target_reduction
        reward += 1.00 * global_reduction
        reward += 1.00 if target_sla_restored else 0.00
        reward += 1.25 if twin_resolved else 0.00
        reward += 1.50 if sla_restored else 0.00
        reward += 1.00 if resolved else 0.00
    reward -= 0.04 * len(commands)
    reward -= 0.001 * max(0, instruction_tokens - 120)
    reward -= 0.10 * iteration_index

    if not commands:
        reward -= 1.00
    if safety.get("unsafe"):
        reward -= 0.25 * len(safety.get("unsafe", []))

    success = bool(credit_eligible and resolved and twin_resolved)

    components = {
        "resolved": resolved,
        "twin_resolved": twin_resolved,
        "sla_restored": sla_restored,
        "target_sla_restored": target_sla_restored,
        "symptom_reduction": global_reduction,
        "global_symptom_reduction": global_reduction,
        "target_symptom_reduction": target_reduction,
        "action_repairs_fault_type": action_repairs,
        "safe": safe,
        "num_commands": len(commands),
        "has_verification_command": has_verify,
        "has_mutating_command": has_mutation,
        "instruction_tokens": instruction_tokens,
        "iteration_index": iteration_index,
        "reward_route": reward_route,
        "telemetry_incomplete": telemetry_incomplete,
        "after_state_observed": after_state_observed,
        "observable_improvement": observable_improvement,
        "positive_credit_eligible": credit_eligible,
        "verifier_reason": verifier_result.get("reason"),
        "before_sla": verifier_result.get("before_sla"),
        "after_sla": verifier_result.get("after_sla"),
        "target_before_sla": verifier_result.get("target_before_sla"),
        "target_after_sla": verifier_result.get("target_after_sla"),
    }
    return {
        "reward": round(float(reward), 4),
        "success": success,
        "components": components,
        "feedback": feedback(safety, verifier_result, commands),
    }


def feedback(safety: dict[str, Any], verifier_result: dict[str, Any], commands: list[str]) -> str:
    if not safety.get("safe"):
        return "One or more commands were unsafe or unsupported. Use scoped kubectl commands only."
    if not commands:
        return "No executable commands were produced."
    if verifier_result.get("resolved") and verifier_result.get("sla_condition_satisfied"):
        return "The independent repair cleared observed symptoms and met the configured SLA."
    if verifier_result.get("resolved") and verifier_result.get("sla_restored"):
        return "Commands repaired the twin target and restored the SLA-style symptom signature."
    if verifier_result.get("resolved") and verifier_result.get("target_sla_restored"):
        return "Commands repaired the RCA target in the twin, but global cascade symptoms remain in the offline abstraction."
    if verifier_result.get("target_symptom_reduction", 0.0) > 0:
        return f"Commands partially reduced target symptoms (target_reduction={float(verifier_result.get('target_symptom_reduction', 0.0)):.2f})."
    if verifier_result.get("global_symptom_reduction", 0.0) > 0:
        return f"Commands partially reduced global symptoms (reduction={float(verifier_result.get('global_symptom_reduction', 0.0)):.2f})."
    return "Commands were valid but did not improve the behavioral twin/SLA symptoms."


def terminal_action_failure_penalty(num_iterations: int = 7) -> dict[str, Any]:
    return {
        "reward": -2.0,
        "success": False,
        "components": {"terminal_failure": True, "num_iterations": num_iterations},
        "feedback": "Action loop failed after the iteration budget. Try a different remediation strategy.",
    }


def _clamp01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _is_mutating_command(command: str) -> bool:
    """True for a command that actually mutates twin state.

    Parses the command rather than substring-matching "kubectl patch" etc. —
    a global flag placed before the verb (``kubectl -n ns patch ...``, valid
    kubectl syntax the agents routinely use) breaks a fixed-prefix substring
    check just as badly as it breaks fixed-index parsing.
    """
    raw = str(command or "")
    if "mongosh" in raw.lower():
        return True
    parts = split_command(raw)
    if not parts:
        return False
    program = parts[0].lower()
    positional = positional_args(parts, 1)
    if not positional:
        return False
    verb = positional[0].lower()
    if program == "kubectl":
        if verb in {"patch", "scale"}:
            return True
        if verb == "rollout" and len(positional) > 1 and positional[1].lower() == "restart":
            return True
        if verb == "delete":
            target = resource_target(positional)
            kind = target[0] if target else ""
            return kind in {
                "pod", "pods", "networkchaos", "podchaos", "stresschaos",
            }
        return False
    if program == "helm" and verb == "rollback":
        return True
    return False
