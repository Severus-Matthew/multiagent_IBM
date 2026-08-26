from __future__ import annotations

"""GPU audit for genuine instruction-sensitive joint policy trajectories.

Unlike ``audit_qwen_exact_e2e_gpu``, this smoke does not use the deterministic
HeuristicRCASolver/FixedActionAgent.  It reuses the same resident Qwen base with
all LoRA adapters disabled as frozen RCA and Action executors.  The only
stochastic/trainable outputs are the ``lora_rca`` and ``lora_action`` policy
instructions.  Frozen downstream decoding is greedy, so within-incident reward
variation can be attributed to policy instructions rather than executor noise.

This remains a mechanics/credit audit against the current BehavioralTwinVerifier;
it is not the final scientific live-Kubernetes-twin training run.
"""

import argparse
import json
from pathlib import Path
from statistics import pstdev
from typing import Any

import torch

from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .audit_qwen_exact_e2e_gpu import _prompt_contract, _replay_ratios, _stamp
from .bounded_agent_state import BoundedAgentStateConfig
from .data_loader import iter_scenarios
from .end_to_end_loop import run_end_to_end_trajectory_group
from .frozen_qwen_agents import (
    FrozenBaseGenerationConfig,
    FrozenBaseQwenGenerator,
    FrozenQwenActionAgent,
    FrozenQwenRCASolver,
)
from .ground_truth import labels_from_full_state
from .grpo_dataset import load_grpo_dataset, summarize_dataset
from .hf_exact_token_sampler import ExactTokenGenerationConfig, HFExactTokenPolicySampler
from .qwen_shared_policy_backend import (
    DEFAULT_QWEN_MODEL,
    QwenSharedPolicyBackendConfig,
    SUPPORTED_QUANTIZATION_MODES,
    load_qwen_shared_policy_backend,
)
from .split_utils import read_scenario_ids
from .trainable_hf_prompt_policies import (
    TrainableHFActionPromptPolicy,
    TrainableHFRCAInstructionPolicy,
)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _variation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completions = [str(row.get("completion") or "") for row in rows]
    downstream = [str(row.get("solver_prediction") or "") for row in rows]
    policy_rewards = [float(row.get("policy_reward", 0.0) or 0.0) for row in rows]
    policy_advantages = [float(row.get("policy_advantage", 0.0) or 0.0) for row in rows]
    return {
        "num_rows": len(rows),
        "unique_policy_completions": len(set(completions)),
        "unique_downstream_outputs": len(set(downstream)),
        "policy_reward_std": pstdev(policy_rewards) if len(policy_rewards) > 1 else 0.0,
        "policy_advantage_std": pstdev(policy_advantages) if len(policy_advantages) > 1 else 0.0,
        "completion_examples": completions[:4],
        "downstream_output_examples": downstream[:4],
        "policy_rewards": policy_rewards,
        "policy_advantages": policy_advantages,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy-sensitive real-Qwen joint GPU smoke")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    ap.add_argument(
        "--quantization",
        choices=sorted(SUPPORTED_QUANTIZATION_MODES),
        default="nf4",
    )
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--trajectory_group_size", type=int, default=4)
    ap.add_argument("--rca_max_iterations", type=int, default=1)
    ap.add_argument("--action_max_iterations", type=int, default=1)
    ap.add_argument("--policy_max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_prompt_tokens", type=int, default=18_000)
    ap.add_argument("--downstream_max_prompt_tokens", type=int, default=22_000)
    ap.add_argument("--downstream_rca_max_new_tokens", type=int, default=64)
    ap.add_argument("--downstream_action_max_new_tokens", type=int, default=128)
    ap.add_argument("--max_serialized_chars", type=int, default=50_000)
    ap.add_argument("--max_system_services", type=int, default=1)
    ap.add_argument("--max_metric_services", type=int, default=64)
    ap.add_argument("--max_log_services", type=int, default=1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.trajectory_group_size < 2:
        raise ValueError("--trajectory_group_size must be >= 2")

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rca_path = out_dir / "rca_policy_samples.jsonl"
    action_path = out_dir / "action_policy_samples.jsonl"
    trajectory_path = out_dir / "joint_trajectories.jsonl"
    summary_path = out_dir / "summary.json"
    for path in (rca_path, action_path, trajectory_path, summary_path):
        if path.exists():
            path.unlink()

    bounded_cfg = BoundedAgentStateConfig(
        max_serialized_chars=args.max_serialized_chars,
        max_system_services=args.max_system_services,
        max_metric_services=args.max_metric_services,
        max_log_services=args.max_log_services,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    backend = load_qwen_shared_policy_backend(
        QwenSharedPolicyBackendConfig(
            model_name=args.model,
            device="cuda:0",
            quantization=args.quantization,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=("q_proj", "v_proj"),
        )
    )
    model = backend.model
    tokenizer = backend.tokenizer

    sampler = HFExactTokenPolicySampler(
        model,
        tokenizer,
        config=ExactTokenGenerationConfig(
            max_new_tokens=args.policy_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            max_prompt_tokens=None,
            seed=240824,
            prompt_format="chat_template",
            chat_role="user",
        ),
        device="cuda:0",
    )
    rca_policy = TrainableHFRCAInstructionPolicy(
        sampler,
        adapter_name="lora_rca",
        max_iterations=args.rca_max_iterations,
    )
    action_policy = TrainableHFActionPromptPolicy(sampler, adapter_name="lora_action")

    frozen_generator = FrozenBaseQwenGenerator(
        model,
        tokenizer,
        config=FrozenBaseGenerationConfig(
            max_prompt_tokens=args.downstream_max_prompt_tokens,
            rca_max_new_tokens=args.downstream_rca_max_new_tokens,
            action_max_new_tokens=args.downstream_action_max_new_tokens,
        ),
        device="cuda:0",
    )
    rca_solver = FrozenQwenRCASolver(frozen_generator)
    action_agent = FrozenQwenActionAgent(frozen_generator, max_commands=15)
    twin = BehavioralTwinVerifier()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    scenario_count = 0
    trajectory_count = 0
    unsafe_agent_inputs = 0
    rca_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []

    for rec in iter_scenarios(args.processed_states):
        if allowed_ids is not None and rec.scenario_id not in allowed_ids:
            continue
        if not labels_from_full_state(rec.full_state):
            continue
        if scenario_count >= args.limit:
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
            rca_policy_model_name=f"{args.model}+{args.quantization}+lora_rca",
            action_policy_model_name=f"{args.model}+{args.quantization}+lora_action",
            policy_version="qwen-policy-sensitive-smoke-v1",
            agent_input_mode="training_safe",
            reward_mode="factorized_joint_pipeline_v2_no_double_count",
            min_twin_reproduction_score=0.0,
            rca_downstream_credit_weight=0.15,
            action_system_credit_weight=0.25,
            bounded_agent_state_config=bounded_cfg,
        )
        scenario_count += 1
        trajectories = list(result.get("trajectories", []) or [])
        trajectory_count += len(trajectories)
        trajectory_records.extend(trajectories)
        unsafe_agent_inputs += int(
            not bool((result.get("agent_input_safety") or {}).get("safe_for_training_agent"))
        )
        if isinstance(result.get("agent_state_projection"), dict):
            projections.append(dict(result["agent_state_projection"]))

        _append_jsonl(
            trajectory_path,
            {
                "scenario_id": result.get("scenario_id"),
                "trajectory_group_id": result.get("trajectory_group_id"),
                "trajectories": trajectories,
                "agent_state_projection": result.get("agent_state_projection"),
                "agent_input_safety": result.get("agent_input_safety"),
                "rca_group_zero_variance": result.get("rca_group_zero_variance"),
                "action_group_zero_variance": result.get("action_group_zero_variance"),
            },
        )

        for raw in result.get("rca_grpo_samples", []) or []:
            row = _stamp(raw, adapter_id="lora_rca", optimizer_role="rca_policy")
            rca_rows.append(row)
            _append_jsonl(rca_path, row)
        for raw in result.get("action_grpo_samples", []) or []:
            row = _stamp(raw, adapter_id="lora_action", optimizer_role="action_policy")
            action_rows.append(row)
            _append_jsonl(action_path, row)

        print(
            f"[POLICY-SENSITIVE-QWEN] scenario={scenario_count} id={rec.scenario_id} "
            f"trajectories={len(trajectories)} rca_rows={len(result.get('rca_grpo_samples', []) or [])} "
            f"action_rows={len(result.get('action_grpo_samples', []) or [])}"
        )

    if scenario_count == 0:
        raise RuntimeError("no labeled scenarios matched the requested inputs")
    if unsafe_agent_inputs:
        raise AssertionError(f"unsafe agent-facing inputs detected: {unsafe_agent_inputs}")
    if not rca_rows or not action_rows:
        raise AssertionError("policy-sensitive smoke must emit both role buffers")

    strict_rca = load_grpo_dataset(
        rca_path, require_policy_credit=True, require_old_logprobs=True
    )
    strict_action = load_grpo_dataset(
        action_path, require_policy_credit=True, require_old_logprobs=True
    )
    rca_prompt_contract = _prompt_contract(
        strict_rca, max_prompt_tokens=args.max_prompt_tokens
    )
    action_prompt_contract = _prompt_contract(
        strict_action, max_prompt_tokens=args.max_prompt_tokens
    )
    rca_replay = _replay_ratios(model, strict_rca, "lora_rca")
    action_replay = _replay_ratios(model, strict_action, "lora_action")

    rca_variation = _variation(strict_rca)
    action_variation = _variation(strict_action)
    rca_signal = rca_variation["policy_advantage_std"] > 0.0
    action_signal = action_variation["policy_advantage_std"] > 0.0
    completion_variation = (
        rca_variation["unique_policy_completions"] > 1
        and action_variation["unique_policy_completions"] > 1
    )

    if rca_signal and action_signal and completion_variation:
        status = "PASS_POLICY_SIGNAL"
    elif rca_signal or action_signal:
        status = "PARTIAL_POLICY_SIGNAL"
    else:
        status = "ZERO_POLICY_SIGNAL"

    summary = {
        "status": status,
        "model": args.model,
        "quantization": backend.quantization_mode,
        "gpu": torch.cuda.get_device_name(0),
        "scenario_count": scenario_count,
        "trajectory_count": trajectory_count,
        "policy_group_size": args.trajectory_group_size,
        "policy_max_new_tokens": args.policy_max_new_tokens,
        "policy_sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "downstream_agents": {
            "shared_frozen_qwen_base": True,
            "adapters_disabled": True,
            "decoder": "greedy",
            "rca_max_new_tokens": args.downstream_rca_max_new_tokens,
            "action_max_new_tokens": args.downstream_action_max_new_tokens,
            "max_prompt_tokens": args.downstream_max_prompt_tokens,
            "uses_hidden_labels": False,
            "uses_candidate_root_cause_menu": False,
            "fallback_remediation": False,
        },
        "bounded_agent_state_enabled": True,
        "agent_state_projections": projections,
        "twin_receives_original_compressed_state": True,
        "twin_mode_for_this_smoke": "BehavioralTwinVerifier_offline_debug_not_final_live_k8s",
        "unsafe_agent_inputs": unsafe_agent_inputs,
        "rca": summarize_dataset(strict_rca),
        "action": summarize_dataset(strict_action),
        "rca_variation": rca_variation,
        "action_variation": action_variation,
        "rca_prompt_contract": rca_prompt_contract,
        "action_prompt_contract": action_prompt_contract,
        "rca_exact_replay": rca_replay,
        "action_exact_replay": action_replay,
        "strict_policy_credit_validation": True,
        "strict_old_logprob_validation": True,
        "uses_real_qwen_policy_generation": True,
        "uses_instruction_sensitive_frozen_qwen_downstream_agents": True,
        "uses_real_training_update": False,
        "ready_for_real_grpo_update_mechanics": bool(rca_signal and action_signal),
        "peak_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    # A nonzero exit distinguishes an informative diagnostic from the gate needed
    # before attempting a real two-role optimizer update.
    if status != "PASS_POLICY_SIGNAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
