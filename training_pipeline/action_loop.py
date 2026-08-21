from __future__ import annotations

import hashlib
import json
import math
from statistics import mean, pstdev
from typing import Any, Protocol

from digital_twin_runtime.sla_verifier import sla_verdict_from_state
from digital_twin_runtime.twin_preflight import rca_twin_gate as build_rca_twin_gate

from .action_reward import action_reward, terminal_action_failure_penalty
from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .command_normalizer import normalize_commands
from .command_safety import check_command_safety
from .schemas import ActionAttempt, FaultLabel, GRPORolloutSample, approx_token_count, normalize_fault_type


class ActionPromptPolicy(Protocol):
    def generate(self, context: dict[str, Any]) -> str: ...


class ActionAgentLike(Protocol):
    def get_commands(self, instruction_prompt: str, context: dict[str, Any]) -> list[str]: ...


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _policy_info_from_policy(policy: ActionPromptPolicy) -> dict[str, Any]:
    info = getattr(policy, "last_policy_info", None)
    return _json_safe(info) if isinstance(info, dict) else {}


def _finite_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for x in value:
        try:
            f = float(x)
        except Exception:
            return None
        if not math.isfinite(f):
            return None
        out.append(f)
    return out


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for x in value:
        try:
            i = int(x)
        except Exception:
            return None
        if i < 0:
            return None
        out.append(i)
    return out


def _rollout_token_info(policy_info: dict[str, Any]) -> tuple[float | None, list[float] | None, list[int] | None, list[float] | None]:
    old_logprobs = _finite_float_list(policy_info.get("old_logprobs"))
    completion_token_ids = _int_list(policy_info.get("completion_token_ids"))
    ref_logprobs = _finite_float_list(policy_info.get("ref_logprobs"))

    old_logprob_sum = policy_info.get("old_logprob_sum")
    try:
        old_logprob_sum = float(old_logprob_sum) if old_logprob_sum is not None else None
    except Exception:
        old_logprob_sum = None
    if old_logprob_sum is not None and not math.isfinite(old_logprob_sum):
        old_logprob_sum = None
    if old_logprob_sum is None and old_logprobs is not None:
        old_logprob_sum = float(sum(old_logprobs))

    return old_logprob_sum, old_logprobs, completion_token_ids, ref_logprobs


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


def _public_faults(faults: list[FaultLabel]) -> list[dict[str, str]]:
    """Expose only the RCA prediction itself, never oracle variant metadata."""
    return [
        {
            "service": str(f.service),
            "fault_type": normalize_fault_type(f.fault_type or f.fault_family),
        }
        for f in faults
        if str(f.service or "").strip()
    ]


def _public_rca_result(rca_result: dict[str, Any], rca_faults: list[FaultLabel]) -> dict[str, Any]:
    """Action-agent view of RCA output with private evaluator fields removed."""
    return {
        "root_causes": _public_faults(rca_faults),
        "num_root_causes": len(rca_faults),
        "final_prediction": str(rca_result.get("final_prediction") or ""),
        "source": "upstream_rca_prediction",
    }


def _public_rca_gate(gate: dict[str, Any]) -> dict[str, Any]:
    """Expose only prediction-derived twin feedback to the Action policy."""
    return {
        "mode": gate.get("mode"),
        "reproduction_score": gate.get("reproduction_score"),
        "same_error_pattern_score": gate.get("same_error_pattern_score"),
        "counterfactual_replay_checked": gate.get("counterfactual_replay_checked"),
        "predicted_fault_injection_checked": gate.get("predicted_fault_injection_checked"),
        "same_error_pattern_verified": gate.get("same_error_pattern_verified"),
    }


def _derive_rca_twin_gate(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    rca_faults: list[FaultLabel],
    twin_verifier,
    provided_gate: dict[str, Any] | None,
    min_reproduction_score: float,
    upstream_rca_success: bool | None = None,
    require_upstream_label_success: bool = True,
) -> dict[str, Any]:
    if not require_upstream_label_success:
        provided_gate = None
        upstream_rca_success = None

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
    gate = build_rca_twin_gate(
        twin_result,
        min_reproduction_score,
        rca_success=upstream_rca_success if require_upstream_label_success else None,
    )
    gate["computed_inside_action_loop"] = True
    gate["upstream_rca_success"] = upstream_rca_success if require_upstream_label_success else None
    gate["requires_upstream_label_success"] = bool(require_upstream_label_success)
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
            "feedback": "Action stage skipped because RCA failed the configured twin-verification gate.",
        },
        "skipped_action": True,
        "skip_reason": "rca_not_twin_verified",
        "rca_result": rca_result,
        "public_rca_result": _public_rca_result(rca_result, rca_faults),
        "rca_faults": _public_faults(rca_faults),
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
    agent_state: dict[str, Any] | None = None,
    agent_input_mode: str = "training_safe",
    agent_input_safety: dict[str, Any] | None = None,
    sample_index_offset: int = 0,
    require_upstream_label_success_for_gate: bool = True,
) -> dict[str, Any]:
    """Run the action-prompt loop and emit GRPO-ready samples."""
    if agent_state is None:
        agent_state = sanitize_agent_state(compressed_state, mode=agent_input_mode) if agent_input_mode == "training_safe" else compressed_state
    if agent_input_safety is None:
        agent_input_safety = agent_input_safety_report(agent_state) if agent_input_mode == "training_safe" else {"safe_for_training_agent": None}

    upstream_success = rca_result.get("upstream_success") if require_upstream_label_success_for_gate else None
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
        require_upstream_label_success=require_upstream_label_success_for_gate,
    )
    public_rca = _public_rca_result(rca_result, rca_faults)
    public_gate = _public_rca_gate(gate)

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
            agent_state=agent_state,
            public_rca_result=public_rca,
            public_rca_twin_gate=public_gate,
            current_sla=current_sla,
            history=history,
            iteration=iteration,
            max_iterations=max_iterations,
        )
        group_pairs: list[tuple[GRPORolloutSample, ActionAttempt]] = []

        for local_sample_index in range(max(1, int(group_size))):
            sample_index = int(sample_index_offset) + local_sample_index
            context = {
                "namespace": namespace,
                "iteration": iteration,
                "sample_index": sample_index,
                "max_iterations": max_iterations,
                "rca_result": public_rca,
                "rca_faults": public_rca["root_causes"],
                "rca_twin_gate": public_gate,
                "require_rca_twin_verification": require_rca_twin_verification,
                "current_sla": current_sla,
                "redacted_state": agent_state,
                "previous_attempts": history,
                "task_instruction": "Generate instructions for a fixed ActionAgent that outputs only kubectl/helm/mongosh commands.",
                "action_requirements": [
                    "Use only scoped namespace commands.",
                    "Prefer the minimal remediation matching the predicted RCA fault type.",
                    "Include at least one verification command such as kubectl rollout status, kubectl get, or helm status.",
                    "Do not use exec, apply, replace, shell pipelines, broad deletes, or cluster-wide flags.",
                ],
            }
            instruction = prompt_policy.generate(context)
            policy_info = _policy_info_from_policy(prompt_policy)
            old_logprob_sum, old_logprobs, completion_token_ids, ref_logprobs = _rollout_token_info(policy_info)

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
                old_logprob_sum=old_logprob_sum,
                old_logprobs=old_logprobs,
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
                    "agent_state_hash": _stable_hash(agent_state),
                    "agent_input_mode": agent_input_mode,
                    "agent_input_safety": agent_input_safety,
                    "selection_strategy": selection_strategy,
                    "rca_counterfactual_reproduction_score": public_gate.get("reproduction_score"),
                    "action_family": _action_family_from_instruction(instruction),
                    "normalized_commands": normalized,
                    "safety": safety,
                    "verifier_result": verifier,
                    "sample_index_offset": int(sample_index_offset),
                    "policy_info": policy_info,
                    "old_logprobs_contract": "per_generated_completion_token_sum_matches_old_logprob_sum",
                },
                completion_token_ids=completion_token_ids,
                ref_logprobs=ref_logprobs,
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
        "public_rca_result": public_rca,
        "rca_faults": _public_faults(rca_faults),
        "rca_twin_gate": gate,
        "public_rca_twin_gate": public_gate,
        "initial_sla": current_sla,
        "grpo_samples": grpo_samples,
        "agent_input_mode": agent_input_mode,
        "agent_input_safety": agent_input_safety,
        "grpo_metadata": {
            "group_size": max(1, int(group_size)),
            "max_iterations": max_iterations,
            "selection_strategy": selection_strategy,
            "policy_model_name": policy_model_name,
            "policy_version": policy_version,
            "sample_index_offset": int(sample_index_offset),
            "require_upstream_label_success_for_gate": bool(require_upstream_label_success_for_gate),
        },
    }


def _build_action_policy_prompt(
    agent_state: dict[str, Any],
    public_rca_result: dict[str, Any],
    public_rca_twin_gate: dict[str, Any],
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
        "predicted_rca": public_rca_result,
        "counterfactual_twin_feedback": public_rca_twin_gate,
        "current_sla": current_sla,
        "redacted_state": agent_state,
        "previous_attempts_non_leaking": history,
        "instruction_requirements": [
            "Use only the predicted RCA targets, not downstream victims.",
            "Choose a safe minimal remediation family that matches the predicted RCA fault type.",
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
