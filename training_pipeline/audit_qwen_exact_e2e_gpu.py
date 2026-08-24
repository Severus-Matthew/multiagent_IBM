from __future__ import annotations

"""Single-GPU smoke of the canonical joint trajectory with the real Qwen backend.

This is intentionally a rollout/replay smoke, not a scientific training run. It
loads one Qwen3-Coder shared base using the production single-GPU NF4 QLoRA path,
attaches independent RCA and Action LoRAs, uses Qwen's chat template for exact
policy prompt tokenization, and runs the existing causal pipeline:

    Qwen LoRA_RCA -> RCA solver -> twin -> Qwen LoRA_Action -> action -> verifier

The emitted role buffers must pass strict exact-token GRPO validation and replay
at importance ratio one before any real Qwen optimizer update is attempted.

For this first integration smoke the downstream RCA solver/action executor remain
the deterministic training-safe debug components. Therefore zero within-incident
advantages are allowed and no optimizer step is performed here.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .data_loader import iter_scenarios
from .end_to_end_loop import run_end_to_end_trajectory_group
from .factorized_grpo_learner import FactorizedGRPOConfig, model_decision_loss
from .fixed_action_agent import FixedActionAgent
from .ground_truth import labels_from_full_state
from .grpo_dataset import load_grpo_dataset, summarize_dataset
from .hf_exact_token_sampler import ExactTokenGenerationConfig, HFExactTokenPolicySampler
from .qwen_shared_policy_backend import (
    DEFAULT_QWEN_MODEL,
    QwenSharedPolicyBackendConfig,
    SUPPORTED_QUANTIZATION_MODES,
    load_qwen_shared_policy_backend,
)
from .rca_loop import HeuristicRCASolver
from .split_utils import read_scenario_ids
from .trainable_hf_prompt_policies import (
    TrainableHFActionPromptPolicy,
    TrainableHFRCAInstructionPolicy,
)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _stamp(row: dict[str, Any], *, adapter_id: str, optimizer_role: str) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata", {}) or {})
    metadata.update({
        "adapter_id": adapter_id,
        "optimizer_role": optimizer_role,
        "sync_batch_id": "qwen-gpu-smoke-batch-00000",
    })
    out["metadata"] = metadata
    out["adapter_id"] = adapter_id
    out["sync_batch_id"] = "qwen-gpu-smoke-batch-00000"
    return out


def _replay_ratios(model: Any, rows: list[dict[str, Any]], adapter_name: str) -> dict[str, Any]:
    model.set_adapter(adapter_name)
    model.eval()
    cfg = FactorizedGRPOConfig(kl_coeff=0.0)
    values: list[float] = []
    with torch.no_grad():
        for row in rows:
            d = model_decision_loss(model, row, config=cfg, device="cuda:0")
            values.append(float(d.ratio_mean.detach().cpu()))
    if not values:
        raise AssertionError(f"{adapter_name}: no optimizer rows emitted")
    max_dev = max(abs(x - 1.0) for x in values)
    if max_dev > 2e-4:
        raise AssertionError(f"{adapter_name}: unchanged exact replay ratio deviates from one by {max_dev}")
    return {
        "num_rows": len(values),
        "ratio_min": min(values),
        "ratio_max": max(values),
        "ratio_mean": sum(values) / len(values),
        "max_abs_deviation_from_one": max_dev,
        "all_ratio_one": True,
    }


def _prompt_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    truncated = 0
    chat_template_rows = 0
    for row in rows:
        metadata = row.get("metadata", {}) or {}
        info = metadata.get("policy_info", {}) or {}
        if info.get("chat_template_applied"):
            chat_template_rows += 1
        truncated += int(bool(info.get("prompt_was_truncated")))
        prompt_lengths.append(len(row.get("prompt_token_ids") or []))
        completion_lengths.append(len(row.get("completion_token_ids") or []))
    if chat_template_rows != len(rows):
        raise AssertionError(
            f"Qwen smoke requires chat-template tokenization for every policy row: {chat_template_rows}/{len(rows)}"
        )
    if truncated:
        raise AssertionError(f"Qwen smoke unexpectedly truncated {truncated} policy prompts")
    return {
        "chat_template_rows": chat_template_rows,
        "truncated_rows": truncated,
        "prompt_tokens_min": min(prompt_lengths),
        "prompt_tokens_max": max(prompt_lengths),
        "completion_tokens_min": min(completion_lengths),
        "completion_tokens_max": max(completion_lengths),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Real Qwen two-adapter exact-token end-to-end GPU smoke")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    ap.add_argument(
        "--quantization",
        choices=sorted(SUPPORTED_QUANTIZATION_MODES),
        default="nf4",
        help="Single-GPU production default is NF4 QLoRA; BF16 is retained only for larger-memory hosts.",
    )
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--trajectory_group_size", type=int, default=2)
    ap.add_argument("--rca_max_iterations", type=int, default=1)
    ap.add_argument("--action_max_iterations", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
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
    for path in (rca_path, action_path, trajectory_path):
        if path.exists():
            path.unlink()

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

    allocated_after_load = torch.cuda.memory_allocated() / 1024**3
    reserved_after_load = torch.cuda.memory_reserved() / 1024**3
    print(
        f"[QWEN-GPU-SMOKE] quantization={backend.quantization_mode} "
        f"model_footprint_gib={backend.model_memory_footprint_gib} "
        f"allocated_after_load_gib={allocated_after_load:.3f} "
        f"reserved_after_load_gib={reserved_after_load:.3f}"
    )

    sampler = HFExactTokenPolicySampler(
        model,
        tokenizer,
        config=ExactTokenGenerationConfig(
            max_new_tokens=args.max_new_tokens,
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

    rca_solver = HeuristicRCASolver()
    action_agent = FixedActionAgent(max_commands=15)
    twin = BehavioralTwinVerifier()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    scenario_count = 0
    trajectory_count = 0
    unsafe_agent_inputs = 0
    rca_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

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
            policy_version="qwen-gpu-smoke-v1",
            agent_input_mode="training_safe",
            reward_mode="factorized_joint_pipeline_v2_no_double_count",
            min_twin_reproduction_score=0.0,
            rca_downstream_credit_weight=0.15,
            action_system_credit_weight=0.25,
        )
        scenario_count += 1
        trajectories = list(result.get("trajectories", []) or [])
        trajectory_count += len(trajectories)
        unsafe_agent_inputs += int(
            not bool((result.get("agent_input_safety") or {}).get("safe_for_training_agent"))
        )
        _append_jsonl(
            trajectory_path,
            {
                "scenario_id": result.get("scenario_id"),
                "trajectory_group_id": result.get("trajectory_group_id"),
                "trajectories": trajectories,
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
            f"[QWEN-GPU-SMOKE] scenario={scenario_count} id={rec.scenario_id} "
            f"trajectories={len(trajectories)} rca_rows={len(result.get('rca_grpo_samples', []) or [])} "
            f"action_rows={len(result.get('action_grpo_samples', []) or [])}"
        )

    if scenario_count == 0:
        raise RuntimeError("no labeled scenarios matched the requested smoke inputs")
    if unsafe_agent_inputs:
        raise AssertionError(f"unsafe agent-facing inputs detected: {unsafe_agent_inputs}")
    if not rca_rows or not action_rows:
        raise AssertionError("Qwen smoke must emit both RCA and Action optimizer rows")

    strict_rca = load_grpo_dataset(
        rca_path,
        require_policy_credit=True,
        require_old_logprobs=True,
    )
    strict_action = load_grpo_dataset(
        action_path,
        require_policy_credit=True,
        require_old_logprobs=True,
    )

    rca_prompt_contract = _prompt_contract(strict_rca)
    action_prompt_contract = _prompt_contract(strict_action)
    rca_replay = _replay_ratios(model, strict_rca, "lora_rca")
    action_replay = _replay_ratios(model, strict_action, "lora_action")

    summary = {
        "status": "PASS",
        "model": args.model,
        "quantization": backend.quantization_mode,
        "model_memory_footprint_gib": (
            round(backend.model_memory_footprint_gib, 3)
            if backend.model_memory_footprint_gib is not None
            else None
        ),
        "allocated_after_load_gib": round(allocated_after_load, 3),
        "reserved_after_load_gib": round(reserved_after_load, 3),
        "gpu": torch.cuda.get_device_name(0),
        "scenario_count": scenario_count,
        "trajectory_count": trajectory_count,
        "unsafe_agent_inputs": unsafe_agent_inputs,
        "shared_base_two_role_adapters": True,
        "adapter_parameter_counts": backend.adapter_parameter_counts,
        "qwen_chat_template_exact_prompt_tokens": True,
        "rca": summarize_dataset(strict_rca),
        "action": summarize_dataset(strict_action),
        "rca_prompt_contract": rca_prompt_contract,
        "action_prompt_contract": action_prompt_contract,
        "rca_exact_replay": rca_replay,
        "action_exact_replay": action_replay,
        "strict_policy_credit_validation": True,
        "strict_old_logprob_validation": True,
        "uses_real_qwen_generation": True,
        "uses_real_training_update": False,
        "downstream_components_are_debug_for_this_smoke": True,
        "peak_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
