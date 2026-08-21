from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .action_prompt_policy import StructuredActionPromptPolicy
from .data_loader import iter_scenarios
from .end_to_end_loop import run_end_to_end_trajectory_group
from .fixed_action_agent import FixedActionAgent
from .ground_truth import labels_from_full_state
from .prompt_operator_policy import OperatorRCAInstructionPolicy
from .qwen_prompt_policy import QwenRCAInstructionPolicy
from .rca_loop import HeuristicRCAInstructionPolicy, HeuristicRCASolver
from .split_utils import read_scenario_ids
from .training_safe_llm_rca_solver import TrainingSafeLLMRCASolver


REWARD_MODE = "factorized_joint_pipeline_v2_no_double_count"
CREDIT_ASSIGNMENT_MODE = "joint_rollout_factorized_policy_returns_v2"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _build_rca_policy(args):
    if args.rca_instruction_policy == "heuristic":
        return HeuristicRCAInstructionPolicy()
    if args.rca_instruction_policy == "operator":
        return OperatorRCAInstructionPolicy(profile=args.operator_profile, safe_mode=True)
    if args.rca_instruction_policy == "qwen_stub":
        return QwenRCAInstructionPolicy(
            model_name=args.qwen_model,
            dry_run=True,
            temperature=args.qwen_temperature,
            top_p=args.qwen_top_p,
        )
    raise ValueError(args.rca_instruction_policy)


def _build_rca_solver(args):
    if args.rca_solver == "heuristic":
        return HeuristicRCASolver()
    if args.rca_solver == "safe_llm":
        return TrainingSafeLLMRCASolver(
            provider=args.llm_provider,
            model=args.llm_model,
            max_tokens=args.llm_max_tokens,
            temperature=args.llm_temperature,
            state_char_budget=args.llm_state_char_budget,
            cache_path=str(Path(args.output_dir).expanduser() / "llm_rca_cache.jsonl"),
            max_root_causes=args.llm_max_root_causes,
        )
    raise ValueError(args.rca_solver)


def _build_action_agent(args):
    if args.action_agent == "fixed":
        return FixedActionAgent(max_commands=args.max_commands)
    if args.action_agent == "llm":
        from agents.action_agent import ActionAgent
        from agents.llm_client import LLMClient
        return ActionAgent(
            client=LLMClient(provider=args.action_llm_provider, model=args.action_llm_model),
            max_commands=args.max_commands,
        )
    raise ValueError(args.action_agent)


def _stamp_optimizer_buffer_sample(
    sample: dict[str, Any],
    *,
    sync_batch_id: str,
    adapter_id: str,
    optimizer_role: str,
) -> dict[str, Any]:
    row = dict(sample)
    metadata = dict(row.get("metadata", {}) or {})
    metadata.update({
        "sync_batch_id": sync_batch_id,
        "adapter_id": adapter_id,
        "optimizer_role": optimizer_role,
        "update_schedule": "collect_joint_batch_then_update_separate_policies_then_publish_together",
    })
    row["metadata"] = metadata
    row["sync_batch_id"] = sync_batch_id
    row["adapter_id"] = adapter_id
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Joint RCA -> twin -> Action -> recovery rollout driver with separate "
            "RCA and Action policy credit/buffers"
        )
    )
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--trajectory_group_size", type=int, default=4)
    ap.add_argument("--rca_max_iterations", type=int, default=3)
    ap.add_argument("--action_max_iterations", type=int, default=3)
    ap.add_argument("--policy_version", default="factorized-joint-v2")
    ap.add_argument(
        "--update_batch_scenarios",
        type=int,
        default=32,
        help=(
            "Number of incident groups collected under the same RCA/Action policy versions before "
            "the future separate learner updates are synchronized and published together."
        ),
    )

    ap.add_argument("--rca_instruction_policy", choices=["heuristic", "operator", "qwen_stub"], default="operator")
    ap.add_argument("--operator_profile", choices=["auto", "system_first", "trace_first", "log_first", "multifault_first"], default="auto")
    ap.add_argument("--qwen_model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--qwen_temperature", type=float, default=0.7)
    ap.add_argument("--qwen_top_p", type=float, default=0.9)

    ap.add_argument("--rca_solver", choices=["heuristic", "safe_llm"], default="heuristic")
    ap.add_argument("--llm_provider", choices=["openai", "claude"], default="openai")
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--llm_max_tokens", type=int, default=300)
    ap.add_argument("--llm_temperature", type=float, default=0.7)
    ap.add_argument("--llm_state_char_budget", type=int, default=24000)
    ap.add_argument("--llm_max_root_causes", type=int, default=2)

    ap.add_argument("--action_strategy", default="auto")
    ap.add_argument("--action_agent", choices=["fixed", "llm"], default="fixed")
    ap.add_argument("--action_llm_provider", choices=["openai", "claude"], default="openai")
    ap.add_argument("--action_llm_model", default=None)
    ap.add_argument("--max_commands", type=int, default=15)

    ap.add_argument("--min_twin_reproduction_score", type=float, default=0.0)
    ap.add_argument(
        "--rca_downstream_credit_weight",
        type=float,
        default=0.15,
        help="Fraction of the RCA policy return supplied by downstream end-to-end recovery quality.",
    )
    ap.add_argument(
        "--action_system_credit_weight",
        type=float,
        default=0.25,
        help="Fraction of the Action policy return supplied by shared end-to-end recovery quality.",
    )
    args = ap.parse_args()

    if args.update_batch_scenarios < 1:
        raise ValueError("--update_batch_scenarios must be >= 1")
    if args.trajectory_group_size < 2:
        raise ValueError("--trajectory_group_size must be >= 2 for group-relative policy optimization")
    if not (0.0 <= args.rca_downstream_credit_weight <= 1.0):
        raise ValueError("--rca_downstream_credit_weight must be in [0, 1]")
    if not (0.0 <= args.action_system_credit_weight <= 1.0):
        raise ValueError("--action_system_credit_weight must be in [0, 1]")

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = out_dir / "joint_trajectories.jsonl"
    rca_samples_path = out_dir / "rca_policy_samples.jsonl"
    action_samples_path = out_dir / "action_policy_samples.jsonl"
    combined_samples_path = out_dir / "all_policy_samples_diagnostic.jsonl"
    update_manifest_path = out_dir / "policy_update_batches.jsonl"
    summary_path = out_dir / "summary.json"
    for p in (
        trajectories_path,
        rca_samples_path,
        action_samples_path,
        combined_samples_path,
        update_manifest_path,
    ):
        if p.exists():
            p.unlink()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    rca_policy = _build_rca_policy(args)
    rca_solver = _build_rca_solver(args)
    action_policy = StructuredActionPromptPolicy(strategy=args.action_strategy)
    action_agent = _build_action_agent(args)
    twin = BehavioralTwinVerifier()

    scenario_count = 0
    trajectory_count = 0
    successful_trajectories = 0
    rca_sample_count = 0
    action_sample_count = 0
    unsafe_agent_inputs = 0
    rca_zero_variance_groups = 0
    action_zero_variance_groups = 0
    batch_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scenario_count": 0,
            "trajectory_count": 0,
            "rca_samples": 0,
            "action_samples": 0,
            "successful_trajectories": 0,
        }
    )

    for rec in iter_scenarios(args.processed_states):
        if allowed_ids is not None and rec.scenario_id not in allowed_ids:
            continue
        if not labels_from_full_state(rec.full_state):
            continue
        if args.limit is not None and scenario_count >= args.limit:
            break

        batch_index = scenario_count // args.update_batch_scenarios
        sync_batch_id = f"sync-batch-{batch_index:05d}"

        result = run_end_to_end_trajectory_group(
            rec.full_state,
            rec.compressed_state,
            rca_instruction_policy=rca_policy,
            rca_solver=rca_solver,
            action_prompt_policy=action_policy,
            action_agent=action_agent,
            twin_verifier=twin,
            trajectory_group_size=args.trajectory_group_size,
            rca_max_iterations=args.rca_max_iterations,
            action_max_iterations=args.action_max_iterations,
            rca_policy_model_name=(
                args.qwen_model + ":dry-run" if args.rca_instruction_policy == "qwen_stub"
                else f"{args.rca_instruction_policy}-rca-policy"
            ),
            action_policy_model_name=f"structured-action-policy:{args.action_strategy}",
            policy_version=args.policy_version,
            agent_input_mode="training_safe",
            reward_mode=REWARD_MODE,
            min_twin_reproduction_score=args.min_twin_reproduction_score,
            rca_downstream_credit_weight=args.rca_downstream_credit_weight,
            action_system_credit_weight=args.action_system_credit_weight,
        )

        scenario_count += 1
        unsafe_agent_inputs += int(not bool((result.get("agent_input_safety") or {}).get("safe_for_training_agent")))
        rca_zero_variance_groups += int(bool(result.get("rca_group_zero_variance")))
        action_zero_variance_groups += int(bool(result.get("action_group_zero_variance")))
        trajectories = result.get("trajectories", []) or []
        rca_samples = result.get("rca_grpo_samples", []) or []
        action_samples = result.get("action_grpo_samples", []) or []
        trajectory_count += len(trajectories)
        num_success = sum(1 for t in trajectories if t.get("trajectory_success"))
        successful_trajectories += num_success
        rca_sample_count += len(rca_samples)
        action_sample_count += len(action_samples)

        _append_jsonl(trajectories_path, {
            "scenario_id": result.get("scenario_id"),
            "trajectory_group_id": result.get("trajectory_group_id"),
            "sync_batch_id": sync_batch_id,
            "reward_mode": result.get("reward_mode"),
            "credit_assignment_mode": result.get("credit_assignment_mode"),
            "update_schedule": result.get("update_schedule"),
            "system_group_reward_mean": result.get("system_group_reward_mean"),
            "system_group_reward_std": result.get("system_group_reward_std"),
            "rca_group_return_mean": result.get("rca_group_return_mean"),
            "rca_group_return_std": result.get("rca_group_return_std"),
            "action_group_return_mean": result.get("action_group_return_mean"),
            "action_group_return_std": result.get("action_group_return_std"),
            "rca_group_zero_variance": result.get("rca_group_zero_variance"),
            "action_group_zero_variance": result.get("action_group_zero_variance"),
            "trajectories": trajectories,
            "agent_input_safety": result.get("agent_input_safety"),
            "policy_credit_contract": result.get("policy_credit_contract"),
        })

        for sample in rca_samples:
            row = _stamp_optimizer_buffer_sample(
                sample,
                sync_batch_id=sync_batch_id,
                adapter_id="lora_rca",
                optimizer_role="rca_policy",
            )
            _append_jsonl(rca_samples_path, row)
            _append_jsonl(combined_samples_path, row)
        for sample in action_samples:
            row = _stamp_optimizer_buffer_sample(
                sample,
                sync_batch_id=sync_batch_id,
                adapter_id="lora_action",
                optimizer_role="action_policy",
            )
            _append_jsonl(action_samples_path, row)
            _append_jsonl(combined_samples_path, row)

        stats = batch_stats[sync_batch_id]
        stats["scenario_count"] += 1
        stats["trajectory_count"] += len(trajectories)
        stats["rca_samples"] += len(rca_samples)
        stats["action_samples"] += len(action_samples)
        stats["successful_trajectories"] += num_success

        print(
            f"[E2E] scenario={scenario_count} batch={sync_batch_id} id={rec.scenario_id} "
            f"trajectories={len(trajectories)} successful={num_success} "
            f"rca_samples={len(rca_samples)} action_samples={len(action_samples)}"
        )

    for sync_batch_id in sorted(batch_stats):
        stats = batch_stats[sync_batch_id]
        _append_jsonl(update_manifest_path, {
            "sync_batch_id": sync_batch_id,
            **stats,
            "rca_adapter_id": "lora_rca",
            "action_adapter_id": "lora_action",
            "shared_base_model": args.qwen_model,
            "update_contract": (
                "Freeze rollout policy versions for this batch; update RCA and Action adapters separately using "
                "their precomputed policy_advantage fields. The future learner uses token-level clipped ratios "
                "and DAPO-style normalization by total active completion tokens within each role update, then "
                "publishes both adapter versions together."
            ),
            "rca_optimizer_advantage_field": "policy_advantage",
            "action_optimizer_advantage_field": "policy_advantage",
            "system_advantage_is_optimizer_signal": False,
            "recompute_advantages_inside_optimizer": False,
        })

    summary = {
        "stage": "end_to_end_factorized_joint",
        "scenario_count": scenario_count,
        "trajectory_count": trajectory_count,
        "successful_trajectories": successful_trajectories,
        "trajectory_success_rate": successful_trajectories / max(trajectory_count, 1),
        "rca_policy_samples": rca_sample_count,
        "action_policy_samples": action_sample_count,
        "unsafe_agent_inputs": unsafe_agent_inputs,
        "trajectory_group_size": args.trajectory_group_size,
        "rca_zero_variance_groups": rca_zero_variance_groups,
        "action_zero_variance_groups": action_zero_variance_groups,
        "rca_max_iterations": args.rca_max_iterations,
        "action_max_iterations": args.action_max_iterations,
        "agent_input_mode": "training_safe",
        "reward_mode": REWARD_MODE,
        "credit_assignment_mode": CREDIT_ASSIGNMENT_MODE,
        "advantage_normalization": "per_incident_complete_trajectory_group_sample_std_plus_1e-4",
        "advantage_std_correction": 1,
        "update_schedule": "batch_synchronized_separate_policy_updates",
        "future_loss_aggregation": "DAPO-style total-active-token normalization per role optimizer update",
        "update_batch_scenarios": args.update_batch_scenarios,
        "num_sync_batches": len(batch_stats),
        "rca_downstream_credit_weight": args.rca_downstream_credit_weight,
        "action_system_credit_weight": args.action_system_credit_weight,
        "rca_instruction_policy": args.rca_instruction_policy,
        "rca_solver": args.rca_solver,
        "action_prompt_policy": "structured",
        "action_agent": args.action_agent,
        "twin_mode": "counterfactual_offline_diagnostic",
        "uses_hidden_rca_success_for_action_transition": False,
        "uses_real_training_update": False,
        "uses_real_trainable_rca_policy": False,
        "uses_real_trainable_action_policy": False,
        "shared_base_model_contract": args.qwen_model,
        "rca_adapter_contract": "shared_frozen_base+lora_rca+separate_optimizer",
        "action_adapter_contract": "shared_frozen_base+lora_action+separate_optimizer",
        "optimizer_must_use_precomputed_policy_advantage": True,
        "optimizer_must_use_exact_completion_token_ids": True,
        "optimizer_must_use_per_token_old_logprobs": True,
        "joint_trajectories_jsonl": str(trajectories_path),
        "rca_policy_samples_jsonl": str(rca_samples_path),
        "action_policy_samples_jsonl": str(action_samples_path),
        "all_policy_samples_diagnostic_jsonl": str(combined_samples_path),
        "policy_update_batches_jsonl": str(update_manifest_path),
        "next_training_backend_requirement": (
            "Enable trainable Qwen sampling for both adapters and record exact completion token IDs plus "
            "per-token old-policy logprobs. The learner consumes stored policy_advantage directly with "
            "token-level importance ratios and must not renormalize duplicated decision rows."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
