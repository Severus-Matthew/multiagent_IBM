from __future__ import annotations

"""Real two-role Qwen GRPO optimizer audit using genuine trajectory advantages.

This consumes policy-sensitive RCA/Action JSONL buffers exactly as emitted by the
joint Qwen rollout audit.  Unlike earlier synthetic-credit tests, no reward or
advantage is replaced.  The run is still a *mechanics* audit because those buffers
were scored by the offline BehavioralTwinVerifier rather than the final live
Kubernetes twin; the updated adapters are therefore intentionally not checkpointed
for scientific training.

The update uses row-streaming backward plus non-reentrant gradient checkpointing so
four ~18k-token trajectories do not retain four full transformer graphs at once.
"""

import argparse
import json
from pathlib import Path
from statistics import pstdev
from typing import Any

import torch

from .audit_qwen_exact_e2e_gpu import _memory_efficient_row_logprobs
from .factorized_grpo_learner import FactorizedGRPOConfig
from .grpo_dataset import load_grpo_dataset
from .peft_adapter_control import parameter_belongs_to_adapter
from .qwen_shared_policy_backend import (
    DEFAULT_QWEN_MODEL,
    QwenSharedPolicyBackendConfig,
    load_qwen_shared_policy_backend,
)
from .streaming_synchronized_grpo_trainer import StreamingSynchronizedFactorizedGRPOTrainer
from .synchronized_grpo_trainer import SynchronizedGRPOTrainerConfig


def _enable_nonreentrant_gradient_checkpointing(model: Any) -> str:
    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    fn = getattr(target, "gradient_checkpointing_enable", None)
    if not callable(fn):
        raise RuntimeError("Qwen base does not expose gradient_checkpointing_enable()")
    try:
        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        return "nonreentrant"
    except TypeError:
        fn()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return "legacy_reentrant_fallback"


def _adapter_snapshot(model: Any, adapter_name: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter_belongs_to_adapter(name, adapter_name)
    }


def _changed_adapter_tensors(
    before: dict[str, torch.Tensor], model: Any, adapter_name: str
) -> list[str]:
    changed: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter_belongs_to_adapter(name, adapter_name):
            continue
        old = before.get(name)
        if old is None:
            raise KeyError(f"missing snapshot tensor: {name}")
        if not torch.equal(old, parameter.detach().cpu()):
            changed.append(name)
    return changed


def _ratio_stats(model: Any, rows: list[dict[str, Any]], adapter_name: str) -> dict[str, Any]:
    model.set_adapter(adapter_name)
    model.eval()
    means: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    with torch.no_grad():
        for row in rows:
            new = _memory_efficient_row_logprobs(model, row)
            old = torch.tensor(
                row["old_logprobs"], dtype=new.dtype, device=new.device
            )
            ratios = torch.exp(new - old)
            if not torch.isfinite(ratios).all():
                raise FloatingPointError("importance ratio contains NaN/Inf")
            means.append(float(ratios.mean().detach().cpu()))
            mins.append(float(ratios.min().detach().cpu()))
            maxs.append(float(ratios.max().detach().cpu()))
    max_dev = max(
        max(abs(x - 1.0) for x in means),
        max(abs(x - 1.0) for x in mins),
        max(abs(x - 1.0) for x in maxs),
    )
    return {
        "num_rows": len(rows),
        "ratio_mean": sum(means) / len(means),
        "ratio_min": min(mins),
        "ratio_max": max(maxs),
        "max_abs_deviation_from_one": max_dev,
        "all_ratio_one_within_2e4": bool(max_dev <= 2e-4),
    }


def _signal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_trajectory: dict[str, float] = {}
    for row in rows:
        tid = str(row["trajectory_id"])
        value = float(row["policy_advantage"])
        previous = by_trajectory.get(tid)
        if previous is not None and abs(previous - value) > 1e-8:
            raise AssertionError(f"{tid}: inconsistent trajectory policy advantage")
        by_trajectory[tid] = value
    values = list(by_trajectory.values())
    return {
        "num_trajectories": len(values),
        "advantages": values,
        "advantage_std": pstdev(values) if len(values) > 1 else 0.0,
        "nonzero_advantage_trajectories": sum(abs(x) > 1e-12 for x in values),
    }


def _single_value(rows: list[dict[str, Any]], key: str) -> str:
    values = {str(row.get(key) or "") for row in rows}
    if len(values) != 1 or "" in values:
        raise ValueError(f"buffer must contain exactly one non-empty {key}: {sorted(values)}")
    return next(iter(values))


def main() -> None:
    ap = argparse.ArgumentParser(description="Real Qwen synchronized two-role GRPO update mechanics audit")
    ap.add_argument("--rca_buffer", required=True)
    ap.add_argument("--action_buffer", required=True)
    ap.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--kl_coeff", type=float, default=0.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    rca_rows = load_grpo_dataset(
        Path(args.rca_buffer).expanduser(),
        require_policy_credit=True,
        require_old_logprobs=True,
    )
    action_rows = load_grpo_dataset(
        Path(args.action_buffer).expanduser(),
        require_policy_credit=True,
        require_old_logprobs=True,
    )
    rca_signal = _signal(rca_rows)
    action_signal = _signal(action_rows)
    if rca_signal["advantage_std"] <= 0.0 or action_signal["advantage_std"] <= 0.0:
        raise ValueError(
            f"real update requires genuine nonzero role signals: rca={rca_signal} action={action_signal}"
        )

    policy_version = _single_value(rca_rows, "policy_version")
    if _single_value(action_rows, "policy_version") != policy_version:
        raise ValueError("RCA and Action buffers use different policy versions")

    torch.cuda.empty_cache()
    backend = load_qwen_shared_policy_backend(
        QwenSharedPolicyBackendConfig(
            model_name=args.model,
            device="cuda:0",
            quantization="nf4",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=("q_proj", "v_proj"),
        )
    )
    model = backend.model
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    checkpoint_mode = _enable_nonreentrant_gradient_checkpointing(model)

    pre_rca = _ratio_stats(model, rca_rows, "lora_rca")
    pre_action = _ratio_stats(model, action_rows, "lora_action")
    if not pre_rca["all_ratio_one_within_2e4"] or not pre_action["all_ratio_one_within_2e4"]:
        raise AssertionError(
            "fresh Qwen+LoRA state does not exactly replay the rollout policy; "
            f"rca={pre_rca} action={pre_action}"
        )

    rca_before = _adapter_snapshot(model, "lora_rca")
    action_before = _adapter_snapshot(model, "lora_action")

    trainer = StreamingSynchronizedFactorizedGRPOTrainer(
        model,
        current_policy_version=policy_version,
        config=SynchronizedGRPOTrainerConfig(
            rca_learning_rate=float(args.learning_rate),
            action_learning_rate=float(args.learning_rate),
            weight_decay=0.0,
            signal_epsilon=1e-12,
            require_single_sync_batch=True,
            require_matching_policy_version=True,
            grpo=FactorizedGRPOConfig(
                clip_epsilon_low=0.2,
                clip_epsilon_high=0.2,
                kl_coeff=float(args.kl_coeff),
                max_grad_norm=1.0,
            ),
        ),
        device="cuda:0",
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before_update = torch.cuda.memory_allocated() / 1024**3
    update = trainer.update_joint_batch(rca_rows, action_rows)
    peak_update = torch.cuda.max_memory_allocated() / 1024**3

    rca_changed = _changed_adapter_tensors(rca_before, model, "lora_rca")
    action_changed = _changed_adapter_tensors(action_before, model, "lora_action")
    if not rca_changed:
        raise AssertionError("real RCA optimizer step changed no RCA LoRA tensors")
    if not action_changed:
        raise AssertionError("real Action optimizer step changed no Action LoRA tensors")
    if not update.get("rca", {}).get("updated") or not update.get("action", {}).get("updated"):
        raise AssertionError(f"both role updates were expected: {update}")

    post_rca = _ratio_stats(model, rca_rows, "lora_rca")
    post_action = _ratio_stats(model, action_rows, "lora_action")
    if post_rca["all_ratio_one_within_2e4"]:
        raise AssertionError("RCA update left rollout importance ratios at one")
    if post_action["all_ratio_one_within_2e4"]:
        raise AssertionError("Action update left rollout importance ratios at one")

    print(json.dumps({
        "status": "PASS_REAL_SYNCHRONIZED_UPDATE_MECHANICS",
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "quantization": backend.quantization_mode,
        "rollout_policy_version": policy_version,
        "published_policy_version": trainer.current_policy_version,
        "uses_actual_trajectory_policy_advantages": True,
        "synthetic_credit_used": False,
        "scientific_training_checkpoint_saved": False,
        "reason_not_scientific_training": "buffers were scored by BehavioralTwinVerifier offline debug, not final live Kubernetes twin",
        "gradient_checkpointing": checkpoint_mode,
        "streaming_row_backward": True,
        "retains_all_row_graphs": False,
        "rca_signal": rca_signal,
        "action_signal": action_signal,
        "pre_update_replay": {"rca": pre_rca, "action": pre_action},
        "update": update,
        "rca_changed_tensor_count": len(rca_changed),
        "action_changed_tensor_count": len(action_changed),
        "post_update_ratio": {"rca": post_rca, "action": post_action},
        "allocated_before_update_gib": round(allocated_before_update, 3),
        "peak_update_cuda_memory_gib": round(peak_update, 3),
        "model_memory_footprint_gib": (
            round(backend.model_memory_footprint_gib, 3)
            if backend.model_memory_footprint_gib is not None else None
        ),
        "twin_trainable": False,
        "shared_base_optimizer_membership": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
