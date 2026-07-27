from __future__ import annotations

from typing import Any, Protocol

from digital_twin_runtime.twin_preflight import rca_twin_gate as build_rca_twin_gate

from .action_reward import action_reward, terminal_action_failure_penalty
from .command_normalizer import normalize_commands
from .command_safety import check_command_safety
from .schemas import ActionAttempt, FaultLabel, approx_token_count


class ActionPromptPolicy(Protocol):
    def generate(self, context: dict[str, Any]) -> str: ...


class ActionAgentLike(Protocol):
    def get_commands(self, instruction_prompt: str, context: dict[str, Any]) -> list[str]: ...


def _namespace(full_state: dict[str, Any], compressed_state: dict[str, Any]) -> str:
    return (
        compressed_state.get("namespace")
        or compressed_state.get("target_namespace")
        or (full_state.get("fault_context", {}) or {}).get("target_namespace")
        or (full_state.get("fault_context", {}) or {}).get("namespace")
        or "default"
    )


def _first_valid_mitigation_action(normalized: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in normalized:
        if not item.get("valid"):
            continue
        if item.get("action") in ("verify", "unknown", "invalid"):
            continue
        return item
    return None


def _scenario_id(full_state: dict[str, Any], compressed_state: dict[str, Any]) -> str:
    return str(full_state.get("scenario_id") or compressed_state.get("scenario_id") or "unknown")


def _derive_rca_twin_gate(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    rca_faults: list[FaultLabel],
    twin_verifier,
    provided_gate: dict[str, Any] | None,
    min_reproduction_score: float,
) -> dict[str, Any]:
    if isinstance(provided_gate, dict) and provided_gate:
        return dict(provided_gate)
    if twin_verifier is None:
        return {
            "rca_twin_verified": False,
            "reason": "missing_twin_verifier",
            "min_reproduction_score": float(min_reproduction_score),
            "reproduction_score": 0.0,
        }
    if not rca_faults:
        return {
            "rca_twin_verified": False,
            "reason": "missing_rca_faults",
            "min_reproduction_score": float(min_reproduction_score),
            "reproduction_score": 0.0,
        }
    twin_result = twin_verifier.validate_rca_prediction(full_state, compressed_state, rca_faults)
    gate = build_rca_twin_gate(twin_result, min_reproduction_score)
    gate["computed_inside_action_loop"] = True
    return gate


def _blocked_by_rca_gate_result(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    rca_result: dict[str, Any],
    rca_faults: list[FaultLabel],
    gate: dict[str, Any],
    max_iterations: int,
) -> dict[str, Any]:
    return {
        "scenario_id": _scenario_id(full_state, compressed_state),
        "success": False,
        "attempts": [],
        "terminal": {
            "reward": 0.0,
            "success": False,
            "components": {
                "action_skipped": True,
                "skip_reason": "rca_not_twin_verified",
                "num_iterations": max_iterations,
            },
            "feedback": "Action stage skipped because RCA failed the twin-verification gate.",
        },
        "skipped_action": True,
        "skip_reason": "rca_not_twin_verified",
        "rca_result": rca_result,
        "rca_faults": [f.to_dict() for f in rca_faults],
        "rca_twin_gate": gate,
    }


def run_action_prompt_optimizer_loop(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    rca_result: dict[str, Any],
    rca_faults: list[FaultLabel],
    prompt_policy: ActionPromptPolicy,
    action_agent: ActionAgentLike,
    twin_verifier,
    max_iterations: int = 5,
    require_rca_twin_verification: bool = False,
    skip_action_if_rca_unverified: bool = True,
    min_twin_reproduction_score: float = 0.0,
    rca_twin_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the action-prompt loop after RCA.

    The action stage can now be explicitly gated by RCA twin verification. When
    require_rca_twin_verification=True and the final RCA is not verified by the
    twin, no remediation commands are generated or scored.
    """
    gate = _derive_rca_twin_gate(
        full_state,
        compressed_state,
        rca_faults,
        twin_verifier,
        provided_gate=rca_twin_gate or rca_result.get("rca_twin_gate"),
        min_reproduction_score=min_twin_reproduction_score,
    )
    if require_rca_twin_verification and skip_action_if_rca_unverified and not gate.get("rca_twin_verified"):
        return _blocked_by_rca_gate_result(full_state, compressed_state, rca_result, rca_faults, gate, max_iterations)

    attempts: list[ActionAttempt] = []
    history: list[dict[str, Any]] = []
    namespace = _namespace(full_state, compressed_state)
    for iteration in range(max_iterations):
        context = {
            "scenario_id": compressed_state.get("scenario_id") or full_state.get("scenario_id"),
            "namespace": namespace,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "rca_result": rca_result,
            "rca_faults": [f.to_dict() for f in rca_faults],
            "rca_twin_gate": gate,
            "require_rca_twin_verification": require_rca_twin_verification,
            "redacted_state": compressed_state,
            "previous_attempts": history,
            "task_instruction": "Generate instructions for a fixed ActionAgent that outputs only kubectl/helm commands.",
        }
        instruction = prompt_policy.generate(context)
        commands = action_agent.get_commands(instruction, context)
        safety = check_command_safety(commands)
        normalized = normalize_commands(commands)
        action = _first_valid_mitigation_action(normalized)
        if safety.get("safe") and action:
            verifier = twin_verifier.apply_action_and_score(full_state, rca_faults, action)
        else:
            verifier = {
                "resolved": False,
                "symptom_reduction": 0.0,
                "reason": "no_safe_valid_mitigation_action",
                "has_safe_commands": bool(safety.get("safe")),
                "has_valid_normalized_action": bool(action),
            }
        reward_obj = action_reward(commands, safety, verifier, approx_token_count(instruction), iteration)
        attempt = ActionAttempt(
            iteration=iteration,
            instruction_prompt=instruction,
            commands=commands,
            reward=reward_obj["reward"],
            reward_components=reward_obj["components"],
            success=reward_obj["success"],
            feedback=reward_obj["feedback"],
            execution_result={"normalized_commands": normalized, "safety": safety},
            verifier_result=verifier,
            token_counts={"instruction_tokens": approx_token_count(instruction)},
        )
        attempts.append(attempt)
        history.append(attempt.to_dict())
        if attempt.success:
            break
    terminal = None if attempts and attempts[-1].success else terminal_action_failure_penalty(max_iterations)
    return {
        "scenario_id": _scenario_id(full_state, compressed_state),
        "success": bool(attempts and attempts[-1].success),
        "attempts": [a.to_dict() for a in attempts],
        "terminal": terminal,
        "skipped_action": False,
        "rca_result": rca_result,
        "rca_faults": [f.to_dict() for f in rca_faults],
        "rca_twin_gate": gate,
    }
