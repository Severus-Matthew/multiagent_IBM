from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Joint RCA -> twin -> Action -> recovery GRPO-ready trajectory rollout driver"
    )
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--trajectory_group_size", type=int, default=4)
    ap.add_argument("--rca_max_iterations", type=int, default=3)
    ap.add_argument("--action_max_iterations", type=int, default=3)
    ap.add_argument("--policy_version", default="joint-v0")

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
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = out_dir / "joint_trajectories.jsonl"
    samples_path = out_dir / "joint_grpo_samples.jsonl"
    summary_path = out_dir / "summary.json"
    for p in (trajectories_path, samples_path):
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
    joint_sample_count = 0
    unsafe_agent_inputs = 0

    for rec in iter_scenarios(args.processed_states):
        if allowed_ids is not None and rec.scenario_id not in allowed_ids:
            continue
        if not labels_from_full_state(rec.full_state):
            continue
        if args.limit is not None and scenario_count >= args.limit:
            break

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
            reward_mode="offline_diagnostic_joint_v1",
            min_twin_reproduction_score=args.min_twin_reproduction_score,
        )

        scenario_count += 1
        unsafe_agent_inputs += int(not bool((result.get("agent_input_safety") or {}).get("safe_for_training_agent")))
        trajectories = result.get("trajectories", []) or []
        samples = result.get("joint_grpo_samples", []) or []
        trajectory_count += len(trajectories)
        successful_trajectories += sum(1 for t in trajectories if t.get("trajectory_success"))
        joint_sample_count += len(samples)

        _append_jsonl(trajectories_path, {
            "scenario_id": result.get("scenario_id"),
            "trajectory_group_id": result.get("trajectory_group_id"),
            "reward_mode": result.get("reward_mode"),
            "group_reward_mean": result.get("group_reward_mean"),
            "group_reward_std": result.get("group_reward_std"),
            "trajectories": trajectories,
            "agent_input_safety": result.get("agent_input_safety"),
        })
        for sample in samples:
            _append_jsonl(samples_path, sample)

        print(
            f"[E2E] scenario={scenario_count} id={rec.scenario_id} "
            f"trajectories={len(trajectories)} successful={sum(1 for t in trajectories if t.get('trajectory_success'))} "
            f"samples={len(samples)}"
        )

    summary = {
        "stage": "end_to_end_joint",
        "scenario_count": scenario_count,
        "trajectory_count": trajectory_count,
        "successful_trajectories": successful_trajectories,
        "trajectory_success_rate": successful_trajectories / max(trajectory_count, 1),
        "joint_grpo_samples": joint_sample_count,
        "unsafe_agent_inputs": unsafe_agent_inputs,
        "trajectory_group_size": args.trajectory_group_size,
        "rca_max_iterations": args.rca_max_iterations,
        "action_max_iterations": args.action_max_iterations,
        "agent_input_mode": "training_safe",
        "reward_mode": "offline_diagnostic_joint_v1",
        "rca_instruction_policy": args.rca_instruction_policy,
        "rca_solver": args.rca_solver,
        "action_prompt_policy": "structured",
        "action_agent": args.action_agent,
        "twin_mode": "counterfactual_offline_diagnostic",
        "uses_hidden_rca_success_for_action_transition": False,
        "uses_real_training_update": False,
        "uses_real_trainable_rca_policy": False,
        "uses_real_trainable_action_policy": False,
        "joint_trajectories_jsonl": str(trajectories_path),
        "joint_grpo_samples_jsonl": str(samples_path),
        "next_training_backend_requirement": (
            "Replace dry-run/structured prompt policies with trainable Qwen/LoRA sampling that records old logprobs, "
            "then update all trainable policy decisions with joint_advantage in one optimizer iteration."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
