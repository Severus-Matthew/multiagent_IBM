from __future__ import annotations

import hashlib
import json
from statistics import mean, pstdev
from typing import Any, Protocol

from digital_twin_runtime.sla_verifier import sla_verdict_from_state
from digital_twin_runtime.twin_preflight import rca_twin_gate as build_rca_twin_gate

from .action_reward import action_reward, terminal_action_failure_penalty
from .command_normalizer import normalize_commands
from .command_safety import check_command_safety
from .schemas import ActionAttempt, FaultLabel, GRPORolloutSample, approx_token_count


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
    upstream_rca_success: bool | None = None,
) -> dict[str, Any]:
    if isinstance(provided_gate, dict) and provided_gate:
        gate = dict(provided_gate)
        if upstream_rca_success is not None:
            gate["upstream_rca_success"] = bool(upstream_rca_success)
            if not upstream_rca_success:
                gate["rca_twin_verified"] = False
                gate["reason"] = "upstream_rca_failed"
        return gate
    if twin_verifier is None:
        return {
            "rca_twin_verified": False,
            "reason": "missing_twin_verifier",
            "min_reproduction_score": float(min_reproduction_score),
            "reproduction_score": 0.0,
            "upstream_rca_success": upstream_rca_success,
        }
    if not rca_faults:
        return {
            "rca_twin_verified": False,
            "reason": "missing_rca_faults",
            "min_reproduction_score": float(min_reproduction_score),
            "reproduction_score": 0.0,
            "upstream_rca_success": upstream_rca_success,
        }
    twin_result = twin_verifier.validate_rca_prediction(full_state, compressed_state, rca_faults)
    gate = build_rca_twin_gate(twin_result, min_reproduction_score, rca_success=upstream_rca_success)
    gate["computed_inside_action_loop"] = True
    gate["upstream_rca_success"] = upstream_rca_success
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
        "grpo_samples": [],
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
    group_size: int = 4,
    selection_strategy: str = "best",
    policy_model_name: str = "structured-action-policy",
    policy_version: str = "v0",
) -> dict[str, Any]:
    """Run the action-prompt loop after RCA and emit GRPO-ready samples.

    At each action iteration, the prompt policy emits `group_size` candidate
    instruction prompts for the same verified RCA and public history. The fixed
    ActionAgent converts each instruction into commands, then the safety checker,
    command normalizer, behavioral twin, and SLA verifier score each candidate.
    Rewards are normalized within the group to produce GRPO advantages. Only the
    selected candidate is appended to episode history.
    """
    upstream_success = rca_result.get("upstream_success")
    if upstream_success is not None:
        upstream_success = bool(upstream_success)
    gate = _derive_rca_twin_gate(
        full_state,
        compressed_state,
        rca_faults,
        twin_verifier,
        provided_gate=rca_twin_gate or rca_result.get("rca_twin_gate"),
        min_reproduction_score=min_twin_reproduction_score,
        upstream_rca_success=upstream_success,
    )
    if require_rca_twin_verification and skip_action_if_rca_unverified and not gate.get("rca_twin_verified"):
        return _blocked_by_rca_gate_result(full_state, compressed_state, rca_result, rca_faults, gate, max_iterations)

    attempts: list[ActionAttempt] = []
    grpo_samples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    namespace = _namespace(full_state, compressed_state)
    current_sla = sla_verdict_from_state(compressed_state)
    scenario_id = _scenario_id(full_state, compressed_state)

    for iteration in range(max_iterations):
        group_id = f"action:{scenario_id}:iter{iteration}"
        policy_prompt = _build_action_policy_prompt(
            compressed_state=compressed_state,
            rca_result=rca_result,
            rca_faults=rca_faults,
            rca_twin_gate=gate,
            current_sla=current_sla,
            history=history,
            iteration=iteration,
            max_iterations=max_iterations,
        )
        group_pairs: list[tuple[GRPORolloutSample, ActionAttempt]] = []

        for sample_index in range(max(1, int(group_size))):
            context = {
                "scenario_id": scenario_id,
                "namespace": namespace,
                "iteration": iteration,
                "sample_index": sample_index,
                "group_id": group_id,
                "max_iterations": max_iterations,
                "rca_result": rca_result,
                "rca_faults": [f.to_dict() for f in rca_faults],
                "rca_twin_gate": gate,
                "require_rca_twin_verification": require_rca_twin_verification,
                "current_sla": current_sla,
                "redacted_state": compressed_state,
                "previous_attempts": history,
                "task_instruction": "Generate instructions for a fixed ActionAgent that outputs only kubectl/helm/mongosh commands.",
                "action_requirements": [
                    "Use only scoped namespace commands.",
                    "Prefer the minimal remediation matching the RCA fault type.",
                    "Include at least one verification command such as kubectl rollout status, kubectl get, or helm status.",
                    "Do not use exec, apply, replace, shell pipelines, broad deletes, or cluster-wide flags.",
                ],
            }
            instruction = prompt_policy.generate(context)
            commands = action_agent.get_commands(instruction, context)
            safety = check_command_safety(commands)
            normalized = normalize_commands(commands)
            action = _first_valid_mitigation_action(normalized)
            if safety.get("safe") and action and twin_verifier is not None:
                verifier = twin_verifier.apply_action_and_score(
                    full_state,
                    rca_faults,
                    action,
                    compressed_state=compressed_state,
                )
            else:
                verifier = {
                    "resolved": False,
                    "twin_resolved": False,
                    "sla_restored": False,
                    "target_sla_restored": False,
                    "symptom_reduction": 0.0,
                    "global_symptom_reduction": 0.0,
                    "target_symptom_reduction": 0.0,
                    "reason": "no_safe_valid_mitigation_action",
                    "has_safe_commands": bool(safety.get("safe")),
                    "has_valid_normalized_action": bool(action),
                }
            reward_obj = action_reward(commands, safety, verifier, approx_token_count(instruction), iteration)
            sample_id = f"{group_id}:sample{sample_index}"
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
            sample = GRPORolloutSample(
                stage="action",
                scenario_id=str(scenario_id),
                group_id=group_id,
                sample_id=sample_id,
                sample_index=sample_index,
                iteration=iteration,
                policy_role="action_prompt_policy",
                policy_prompt=policy_prompt,
                completion=instruction,
                completion_tokens=approx_token_count(instruction),
                old_logprob_sum=None,
                old_logprobs=None,
                reward=reward_obj["reward"],
                reward_components=reward_obj["components"],
                advantage=None,
                group_reward_mean=None,
                group_reward_std=None,
                solver_prediction="\n".join(commands),
                parsed_prediction=normalized,
                success=bool(reward_obj["success"]),
                terminal=False,
                model_name=policy_model_name,
                policy_version=policy_version,
                metadata={
                    "observation_hash": _stable_hash({"scenario_id": scenario_id, "iteration": iteration, "history": history}),
                    "redacted_state_hash": _stable_hash(compressed_state),
                    "selection_strategy": selection_strategy,
                    "rca_twin_verified": bool(gate.get("rca_twin_verified")),
                    "rca_twin_gate_reason": gate.get("reason"),
                    "action_family": _action_family_from_instruction(instruction),
                    "normalized_commands": normalized,
                    "safety": safety,
                    "verifier_result": verifier,
                },
            )
            group_pairs.append((sample, attempt))

        _compute_group_advantages([s for s, _ in group_pairs])
        selected_sample, selected_attempt = _select_action_sample(group_pairs, selection_strategy)
        selected_sample.metadata["selected_for_episode_history"] = True
        attempts.append(selected_attempt)
        history.append(_safe_action_history_entry(selected_attempt))
        for s, _ in group_pairs:
            if "selected_for_episode_history" not in s.metadata:
                s.metadata["selected_for_episode_history"] = False
            grpo_samples.append(s.to_dict())
        if selected_attempt.success:
            break

    terminal = None if attempts and attempts[-1].success else terminal_action_failure_penalty(max_iterations)
    if terminal is not None:
        _apply_terminal_action_penalty(attempts, grpo_samples, terminal)

    return {
        "scenario_id": scenario_id,
        "success": bool(attempts and attempts[-1].success),
        "attempts": [a.to_dict() for a in attempts],
        "terminal": terminal,
        "skipped_action": False,
        "rca_result": rca_result,
        "rca_faults": [f.to_dict() for f in rca_faults],
        "rca_twin_gate": gate,
        "initial_sla": current_sla,
        "grpo_samples": grpo_samples,
        "grpo_metadata": {
            "group_size": max(1, int(group_size)),
            "max_iterations": max_iterations,
            "selection_strategy": selection_strategy,
            "policy_model_name": policy_model_name,
            "policy_version": policy_version,
        },
    }


def _build_action_policy_prompt(
    compressed_state: dict[str, Any],
    rca_result: dict[str, Any],
    rca_faults: list[FaultLabel],
    rca_twin_gate: dict[str, Any],
    current_sla: dict[str, Any],
    history: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> str:
    payload = {
        "task": "Generate an action instruction prompt for a fixed ActionAgent.",
        "agent_output_contract": "ActionAgent must output kubectl/helm/mongosh commands only, one per line.",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "verified_rca": {
            "rca_result": rca_result,
            "rca_faults": [f.to_dict() for f in rca_faults],
            "rca_twin_gate": {
                "rca_twin_verified": rca_twin_gate.get("rca_twin_verified"),
                "reason": rca_twin_gate.get("reason"),
                "same_error_pattern_score": rca_twin_gate.get("same_error_pattern_score"),
            },
        },
        "current_sla": current_sla,
        "redacted_state_hash": _stable_hash(compressed_state),
        "previous_attempts_non_leaking": history,
        "instruction_requirements": [
            "Use only the verified RCA targets, not downstream victims.",
            "Choose a safe minimal remediation family that matches the RCA fault type.",
            "Include verification commands.",
            "Avoid unsafe, broad, cluster-wide, or shell-executing commands.",
        ],
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _select_action_sample(
    pairs: list[tuple[GRPORolloutSample, ActionAttempt]],
    strategy: str,
) -> tuple[GRPORolloutSample, ActionAttempt]:
    if not pairs:
        raise ValueError("cannot select from empty action sample group")
    if strategy == "sample0":
        return pairs[0]
    if strategy != "best":
        raise ValueError(f"unknown action selection_strategy={strategy!r}; use best or sample0")
    return max(pairs, key=lambda x: (float(x[0].reward), int(x[0].success)))


def _compute_group_advantages(samples: list[GRPORolloutSample]) -> None:
    rewards = [float(s.reward) for s in samples]
    if not rewards:
        return
    mu = mean(rewards)
    sigma = pstdev(rewards) if len(rewards) > 1 else 0.0
    denom = sigma if sigma > 1e-8 else 1.0
    for s in samples:
        s.group_reward_mean = round(mu, 6)
        s.group_reward_std = round(sigma, 6)
        s.advantage = round((float(s.reward) - mu) / denom, 6)


def _recompute_dict_group_advantages(samples: list[dict[str, Any]], group_id: str) -> None:
    group = [s for s in samples if s.get("group_id") == group_id]
    rewards = [float(s.get("reward", 0.0)) for s in group]
    if not rewards:
        return
    mu = mean(rewards)
    sigma = pstdev(rewards) if len(rewards) > 1 else 0.0
    denom = sigma if sigma > 1e-8 else 1.0
    for s in group:
        s["group_reward_mean"] = round(mu, 6)
        s["group_reward_std"] = round(sigma, 6)
        s["advantage"] = round((float(s.get("reward", 0.0)) - mu) / denom, 6)


def _apply_terminal_action_penalty(
    attempts: list[ActionAttempt],
    grpo_samples: list[dict[str, Any]],
    terminal: dict[str, Any],
) -> None:
    if not attempts:
        return
    final_iter = attempts[-1].iteration
    penalty = float(terminal.get("reward", 0.0) or 0.0)
    attempts[-1].reward = round(float(attempts[-1].reward) + penalty, 4)
    attempts[-1].reward_components = {
        **attempts[-1].reward_components,
        "terminal_failure": True,
        "terminal_failure_penalty": penalty,
        "reward_after_terminal_penalty": attempts[-1].reward,
    }
    attempts[-1].feedback = f"{attempts[-1].feedback} {terminal.get('feedback', '')}".strip()

    affected_group_ids: set[str] = set()
    for s in grpo_samples:
        if s.get("iteration") == final_iter:
            s["terminal"] = True
            s["reward"] = round(float(s.get("reward", 0.0)) + penalty, 4)
            comps = dict(s.get("reward_components", {}) or {})
            comps["terminal_failure"] = True
            comps["terminal_failure_penalty"] = penalty
            comps["reward_after_terminal_penalty"] = s["reward"]
            s["reward_components"] = comps
            if s.get("group_id"):
                affected_group_ids.add(str(s["group_id"]))
    for gid in affected_group_ids:
        _recompute_dict_group_advantages(grpo_samples, gid)


def _safe_action_history_entry(attempt: ActionAttempt) -> dict[str, Any]:
    c = attempt.reward_components
    return {
        "iteration": attempt.iteration,
        "commands": attempt.commands,
        "reward": attempt.reward,
        "success": attempt.success,
        "feedback": attempt.feedback,
        "public_reward_summary": {
            "resolved": c.get("resolved"),
            "twin_resolved": c.get("twin_resolved"),
            "target_sla_restored": c.get("target_sla_restored"),
            "sla_restored": c.get("sla_restored"),
            "global_symptom_reduction": c.get("global_symptom_reduction"),
            "target_symptom_reduction": c.get("target_symptom_reduction"),
            "safe": c.get("safe"),
            "terminal_failure": c.get("terminal_failure", False),
        },
    }


def _action_family_from_instruction(instruction: str) -> str | None:
    marker = "ACTION_PLAN_JSON:"
    text = str(instruction or "")
    if marker not in text:
        return None
    try:
        payload = json.loads(text.split(marker, 1)[1].strip())
    except Exception:
        return None
    plans = payload.get("plans", []) if isinstance(payload, dict) else []
    if plans and isinstance(plans[0], dict):
        return plans[0].get("action_family")
    return None


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
