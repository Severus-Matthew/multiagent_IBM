from __future__ import annotations

import hashlib
import json
import math
from statistics import mean, pstdev
from typing import Any, Protocol

from .ground_truth import ground_truth_summary, labels_from_full_state
from .rca_reward import rca_reward, terminal_rca_failure_penalty
from .schemas import GRPORolloutSample, RCAAttempt, approx_token_count, parse_fault_lines


class RCAInstructionPolicy(Protocol):
    def generate_instruction(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int = 0,
        group_id: str | None = None,
    ) -> str: ...


class RCASolver(Protocol):
    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str: ...


class HeuristicRCAInstructionPolicy:
    """Debug baseline. Replace with trainable Qwen/LoRA policy."""

    def generate_instruction(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int = 0,
        group_id: str | None = None,
    ) -> str:
        retry = " Avoid repeating previous unsuccessful guesses." if history else ""
        variants = [
            "Prioritize direct Kubernetes service-health signals such as Pending pods, no ready endpoints, crashloops, and failed scheduling; then use logs/traces to separate root cause from cascade.",
            "Start from the first degraded service and dependency neighborhood that can explain downstream trace/log symptoms; avoid blaming downstream victims.",
            "Prioritize resource metrics, restart patterns, unavailable endpoints, and pod readiness before using noisy logs.",
            "Look for datastore, cache, auth, network, and config symptoms that create a fan-out cascade across services.",
            "Use the smallest root-cause set that explains the observed health, metrics, logs, and traces.",
        ]
        strategy = variants[sample_index % len(variants)]
        return (
            "Read only the redacted telemetry. "
            + strategy
            + " Output only service::fault_type::injectible_mechanism, one root cause per line."
            + retry
        )


class HeuristicRCASolver:
    """No-LLM smoke-test baseline using redacted state only."""

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        candidates = []
        for svc, info in (compressed_state.get("system", {}) or {}).items():
            health = (info.get("health", {}) if isinstance(info, dict) else {}) or {}
            score = 0.0
            if health.get("infra_issue_flag"):
                score += 2.0
            if health.get("pods_unready", 0) > 0:
                score += 1.0
            status = str(health.get("status", "")).lower()
            if "no_ready" in status or "pending" in status or "crash" in status:
                score += 1.0
            if score > 0:
                candidates.append((score, svc, "infra_failure"))
        if candidates:
            candidates.sort(reverse=True)
            return f"{candidates[0][1]}::{candidates[0][2]}"
        top = ((compressed_state.get("llm_view", {}) or {}).get("top_log_error_services") or [{}])[0]
        if top.get("service"):
            return f"{top['service']}::dependency_failure"
        services = compressed_state.get("services", []) or []
        return f"{services[0] if services else 'unknown'}::unknown"


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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


def _policy_info_from_policy(policy: RCAInstructionPolicy) -> dict[str, Any]:
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

    # Exact token/log-prob alignment is validated later by grpo_dataset.py. Do not
    # silently truncate one list to fit the other here.
    return old_logprob_sum, old_logprobs, completion_token_ids, ref_logprobs


def build_rca_policy_prompt(
    compressed_state: dict[str, Any],
    history: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> str:
    """Prompt text for the trainable instruction policy.

    `compressed_state` must be the agent-facing state, not the evaluator/private
    state. The training-safe runner passes a candidate/oracle-stripped state here.
    """
    payload = {
        "task": "Generate an RCA instruction prompt for a fixed RCA solver.",
        "solver_output_contract": "The solver must output one component::fault_mechanism line per root cause.",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "redacted_state": compressed_state,
        "previous_attempts_non_leaking": history,
        "instruction_requirements": [
            "Use only redacted telemetry.",
            "Do not ask for ground truth.",
            "Do not mention oracle labels, injected fault families, or candidate menus.",
            "Tell the solver how to distinguish root cause from downstream cascade.",
            "Keep the instruction concise.",
        ],
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _safe_history_entry(attempt: RCAAttempt) -> dict[str, Any]:
    """Return retry history containing no oracle-derived correctness hints."""
    c = attempt.reward_components
    return {
        "iteration": attempt.iteration,
        "prediction": attempt.prediction_text,
        "parsed_prediction": [x.to_dict() for x in attempt.predicted_faults],
        "feedback": attempt.feedback,
        "public_verifier_summary": {
            "twin_reproduction_score": c.get("twin_reproduction_score"),
            "invalid_format": c.get("invalid_format"),
            "repeated_wrong_guess": c.get("repeated_wrong_guess"),
            "terminal_failure": c.get("terminal_failure", False),
        },
    }


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


def _select_sample(samples: list[tuple[GRPORolloutSample, RCAAttempt]], strategy: str) -> tuple[GRPORolloutSample, RCAAttempt]:
    if strategy == "sample0":
        return samples[0]
    if strategy != "best":
        raise ValueError(f"unknown RCA selection_strategy={strategy!r}; use best or sample0")
    return max(samples, key=lambda x: (x[0].reward, int(x[0].success)))


def _apply_terminal_failure_penalty(
    attempts: list[RCAAttempt],
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


def run_rca_grpo_episode(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    instruction_policy: RCAInstructionPolicy,
    solver: RCASolver,
    twin_validator=None,
    max_iterations: int = 5,
    group_size: int = 4,
    selection_strategy: str = "best",
    policy_model_name: str = "debug-heuristic-policy",
    policy_version: str = "v0",
    agent_state: dict[str, Any] | None = None,
    agent_input_mode: str = "legacy",
    agent_input_safety: dict[str, Any] | None = None,
    sample_index_offset: int = 0,
    stop_on_local_success: bool = True,
) -> dict[str, Any]:
    """Run one RCA episode and produce GRPO-ready samples."""
    gt_labels = labels_from_full_state(full_state)
    scenario_id = full_state.get("scenario_id") or compressed_state.get("scenario_id") or "unknown_scenario"
    agent_state = agent_state if agent_state is not None else compressed_state
    attempts: list[RCAAttempt] = []
    grpo_samples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    seen: set[str] = set()

    for iteration in range(max_iterations):
        group_id = f"rca:{scenario_id}:iter{iteration}"
        policy_prompt = build_rca_policy_prompt(agent_state, history, iteration, max_iterations)
        group_pairs: list[tuple[GRPORolloutSample, RCAAttempt]] = []

        for local_sample_index in range(max(1, group_size)):
            sample_index = int(sample_index_offset) + local_sample_index
            instruction = instruction_policy.generate_instruction(
                agent_state, history, iteration, sample_index=sample_index, group_id=group_id
            )
            policy_info = _policy_info_from_policy(instruction_policy)
            old_logprob_sum, old_logprobs, completion_token_ids, ref_logprobs = _rollout_token_info(policy_info)

            prediction_text = solver.solve(agent_state, instruction)
            pred_labels = parse_fault_lines(prediction_text)
            pred_key = "\n".join(sorted(x.canonical_key() for x in pred_labels))
            repeated = bool(pred_key) and pred_key in seen

            twin_result = None
            if twin_validator is not None and pred_labels:
                twin_result = twin_validator.validate_rca_prediction(full_state, compressed_state, pred_labels)

            reward_obj = rca_reward(
                full_state,
                gt_labels,
                pred_labels,
                instruction_tokens=approx_token_count(instruction),
                iteration_index=iteration,
                twin_result=twin_result,
                invalid_format=not pred_labels,
                repeated_wrong_guess=repeated,
            )

            sample_id = f"{group_id}:sample{sample_index}"
            attempt = RCAAttempt(
                iteration=iteration,
                instruction=instruction,
                prediction_text=prediction_text,
                predicted_faults=pred_labels,
                reward=reward_obj["reward"],
                reward_components=reward_obj["components"],
                success=reward_obj["success"],
                feedback=reward_obj["feedback"],
                token_counts={
                    "instruction_tokens": approx_token_count(instruction),
                    "prediction_tokens": approx_token_count(prediction_text),
                },
                group_id=group_id,
                selected_sample_id=sample_id,
            )
            sample = GRPORolloutSample(
                stage="rca",
                scenario_id=str(scenario_id),
                group_id=group_id,
                sample_id=sample_id,
                sample_index=sample_index,
                iteration=iteration,
                policy_role="rca_instruction_policy",
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
                solver_prediction=prediction_text,
                parsed_prediction=[x.to_dict() for x in pred_labels],
                success=bool(reward_obj["success"]),
                terminal=False,
                model_name=policy_model_name,
                policy_version=policy_version,
                metadata={
                    "observation_hash": _stable_hash({"scenario_id": scenario_id, "iteration": iteration, "history": history}),
                    "redacted_state_hash": _stable_hash(compressed_state),
                    "agent_state_hash": _stable_hash(agent_state),
                    "agent_input_mode": agent_input_mode,
                    "agent_input_safety": agent_input_safety,
                    "selection_strategy": selection_strategy,
                    "twin_enabled": twin_validator is not None,
                    "policy_info": policy_info,
                    "sample_index_offset": int(sample_index_offset),
                    "old_logprobs_contract": "per_generated_completion_token_sum_matches_old_logprob_sum",
                },
                completion_token_ids=completion_token_ids,
                ref_logprobs=ref_logprobs,
            )
            group_pairs.append((sample, attempt))

        _compute_group_advantages([s for s, _ in group_pairs])
        selected_sample, selected_attempt = _select_sample(group_pairs, selection_strategy)
        selected_sample.metadata["selected_for_episode_history"] = True
        selected_attempt.selected_sample_id = selected_sample.sample_id
        attempts.append(selected_attempt)
        history.append(_safe_history_entry(selected_attempt))
        for s, _ in group_pairs:
            if "selected_for_episode_history" not in s.metadata:
                s.metadata["selected_for_episode_history"] = False
            grpo_samples.append(s.to_dict())

        selected_key = "\n".join(sorted(x.canonical_key() for x in selected_attempt.predicted_faults))
        if selected_key:
            seen.add(selected_key)
        if stop_on_local_success and selected_attempt.success:
            break

    local_success = bool(attempts and attempts[-1].success)
    terminal = None if local_success else terminal_rca_failure_penalty(max_iterations)
    if terminal is not None:
        _apply_terminal_failure_penalty(attempts, grpo_samples, terminal)

    return {
        "scenario_id": scenario_id,
        "success": local_success,
        "attempts": [a.to_dict() for a in attempts],
        "final_prediction": attempts[-1].prediction_text if attempts else "",
        "ground_truth_summary": ground_truth_summary(full_state),
        "terminal": terminal,
        "grpo_samples": grpo_samples,
        "agent_input_mode": agent_input_mode,
        "agent_input_safety": agent_input_safety,
        "grpo_metadata": {
            "group_size": max(1, group_size),
            "max_iterations": max_iterations,
            "selection_strategy": selection_strategy,
            "policy_model_name": policy_model_name,
            "policy_version": policy_version,
            "sample_index_offset": int(sample_index_offset),
            "stop_on_local_success": bool(stop_on_local_success),
        },
    }


def run_rca_self_prompting_loop(
    full_state: dict[str, Any],
    compressed_state: dict[str, Any],
    instruction_policy: RCAInstructionPolicy,
    solver: RCASolver,
    twin_validator=None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    result = run_rca_grpo_episode(
        full_state,
        compressed_state,
        instruction_policy,
        solver,
        twin_validator=twin_validator,
        max_iterations=max_iterations,
        group_size=1,
        selection_strategy="sample0",
    )
    result.pop("grpo_samples", None)
    return result


def main() -> None:
    import argparse
    from .data_loader import iter_scenarios
    ap = argparse.ArgumentParser(description="Smoke-test Stage 1 RCA loop")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--output", default=None)
    ap.add_argument("--group_size", type=int, default=1)
    ap.add_argument("--max_iterations", type=int, default=5)
    args = ap.parse_args()
    rows = []
    policy = HeuristicRCAInstructionPolicy(); solver = HeuristicRCASolver()
    for rec in iter_scenarios(args.processed_states, limit=args.limit):
        row = run_rca_grpo_episode(
            rec.full_state,
            rec.compressed_state,
            policy,
            solver,
            max_iterations=args.max_iterations,
            group_size=args.group_size,
        )
        rows.append(row)
    print(json.dumps({"total": len(rows), "passed": sum(r["success"] for r in rows)}, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
