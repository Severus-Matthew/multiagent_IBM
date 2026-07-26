from __future__ import annotations

from typing import Any, Protocol
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


def run_action_prompt_optimizer_loop(full_state: dict[str, Any], compressed_state: dict[str, Any],
                                     rca_result: dict[str, Any], rca_faults: list[FaultLabel],
                                     prompt_policy: ActionPromptPolicy, action_agent: ActionAgentLike,
                                     twin_verifier, max_iterations: int = 5) -> dict[str, Any]:
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
        attempt = ActionAttempt(iteration=iteration, instruction_prompt=instruction, commands=commands,
                                reward=reward_obj["reward"], reward_components=reward_obj["components"],
                                success=reward_obj["success"], feedback=reward_obj["feedback"],
                                execution_result={"normalized_commands": normalized, "safety": safety},
                                verifier_result=verifier,
                                token_counts={"instruction_tokens": approx_token_count(instruction)})
        attempts.append(attempt); history.append(attempt.to_dict())
        if attempt.success:
            break
    terminal = None if attempts and attempts[-1].success else terminal_action_failure_penalty(max_iterations)
    return {"scenario_id": full_state.get("scenario_id") or compressed_state.get("scenario_id"),
            "success": bool(attempts and attempts[-1].success),
            "attempts": [a.to_dict() for a in attempts], "terminal": terminal}
