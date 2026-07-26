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
    """Debug baseline. Replace with trainable Qwen/LoRA policy.

    The different sample_index variants exist only to exercise GRPO grouping.
    They are not meant to be competitive RCA prompts.
    """
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
            + " Output only service::fault_type, one root cause per line."
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


def build_rca_policy_prompt(
    compressed_state: dict[str, Any],
    history: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> str:
    """Prompt text for the trainable instruction policy.

    This is what Qwen/LoRA will later condition on. It contains only redacted
    telemetry and non-leaking feedback history.
    """
    payload = {
        "task": "Generate an RCA instruction prompt for a fixed RCA solver.",
        "solver_output_contract": "The solver must output one service::fault_type line per root cause.",
        "canonical_fault_types": [
            "infra_failure", "auth_failure", "dependency_failure", "resource_exhaustion",
            "latency_degradation", "network_failure", "config_error", "unknown",
        ],
        "iteration": iteration,
        "max_iterations": max_iterations,
        "redacted_state": compressed_state,
        "previous_attempts_non_leaking": history,
        "instruction_requirements": [
            "Use only redacted telemetry.",
            "Do not ask for ground truth.",
            "Do not mention oracle labels.",
            "Tell the solver how to distinguish root cause from downstream cascade.",
            "Keep the instruction concise.",
        ],
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _safe_history_entry(attempt: RCAAttempt) -> dict[str, Any]:
    """History shown to the policy. Must not contain GT labels or match internals."""
    c = attempt.reward_components
    return {
        "iteration": attempt.iteration,
        "prediction": attempt.prediction_text,
        "parsed_prediction": [x.to_dict() for x in attempt.predicted_faults],
        "reward": attempt.reward,
        "success": attempt.success,
        "feedback": attempt.feedback,
        "public_reward_summary": {
            "pair_score": c.get("pair_score"),
            "twin_reproduction_score": c.get("twin_reproduction_score"),
            "count_mismatch": c.get("count_mismatch"),
            "invalid_format": c.get("invalid_format"),
            "repeated_wrong_guess": c.get("repeated_wrong_guess"),
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


def _select_sample(samples: list[tuple[GRPORolloutSample, RCAAttempt]], strategy: str) -> tuple[GRPORolloutSample, RCAAttempt]:
    if strategy == "sample0":
        return samples[0]
    if strategy != "best":
        raise ValueError(f"unknown RCA selection_strategy={strategy!r}; use best or sample0")
    return max(samples, key=lambda x: (x[0].reward, int(x[0].success)))


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
) -> dict[str, Any]:
    """Run one RCA episode and produce GRPO-ready samples.

    At each iteration we generate `group_size` candidate instruction prompts for
    the same observation/history. Each candidate is sent to the fixed RCA solver,
    scored, and converted into a GRPO sample. Advantages are normalized within
    that same group. For the next iteration we append only the selected attempt's
    non-leaking feedback to history.
    """
    gt_labels = labels_from_full_state(full_state)
    scenario_id = full_state.get("scenario_id") or compressed_state.get("scenario_id") or "unknown_scenario"
    attempts: list[RCAAttempt] = []
    grpo_samples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    seen: set[str] = set()

    for iteration in range(max_iterations):
        group_id = f"rca:{scenario_id}:iter{iteration}"
        policy_prompt = build_rca_policy_prompt(compressed_state, history, iteration, max_iterations)
        group_pairs: list[tuple[GRPORolloutSample, RCAAttempt]] = []

        for sample_index in range(max(1, group_size)):
            instruction = instruction_policy.generate_instruction(
                compressed_state, history, iteration, sample_index=sample_index, group_id=group_id
            )
            prediction_text = solver.solve(compressed_state, instruction)
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
                old_logprob_sum=None,
                old_logprobs=None,
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
                    "selection_strategy": selection_strategy,
                    "twin_enabled": twin_validator is not None,
                },
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
        if selected_attempt.success:
            break

    terminal = None if attempts and attempts[-1].success else terminal_rca_failure_penalty(max_iterations)
    if terminal is not None:
        for s in grpo_samples:
            if s.get("iteration") == max_iterations - 1:
                s["terminal"] = True

    return {
        "scenario_id": scenario_id,
        "success": bool(attempts and attempts[-1].success),
        "attempts": [a.to_dict() for a in attempts],
        "final_prediction": attempts[-1].prediction_text if attempts else "",
        "ground_truth_summary": ground_truth_summary(full_state),
        "terminal": terminal,
        "grpo_samples": grpo_samples,
        "grpo_metadata": {
            "group_size": max(1, group_size),
            "max_iterations": max_iterations,
            "selection_strategy": selection_strategy,
            "policy_model_name": policy_model_name,
            "policy_version": policy_version,
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
    """Backward-compatible one-sample episode runner."""
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
