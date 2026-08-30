from __future__ import annotations

"""Single-row real-Qwen backward smoke on the bounded RCA policy context.

This is a mechanical GPU-training audit, not a scientific GRPO update. It uses one
real Qwen3-Coder NF4 + RCA-LoRA rollout, exact stored rollout token IDs/logprobs,
and a synthetic non-zero advantage solely to prove that the differentiable
exact-token loss can backpropagate through the real model within single-GPU memory.

The production joint trajectory remains RCA -> Twin -> Action. This audit does not
replace that path and does not use its synthetic advantage as training data.
"""

import argparse
import json
from typing import Any

import torch

from .agent_input_safety import agent_input_safety_report, sanitize_agent_state
from .bounded_agent_state import BoundedAgentStateConfig, build_bounded_agent_state
from .data_loader import iter_scenarios
from .factorized_grpo_learner import FactorizedGRPOConfig, model_decision_loss
from .hf_exact_token_sampler import ExactTokenGenerationConfig, HFExactTokenPolicySampler
from .peft_adapter_control import (
    activate_exclusive_adapter,
    finite_selected_adapter_gradients,
    parameter_belongs_to_adapter,
    trainable_parameter_list,
)
from .qwen_shared_policy_backend import (
    DEFAULT_QWEN_MODEL,
    QwenSharedPolicyBackendConfig,
    load_qwen_shared_policy_backend,
)
from .rca_loop import build_rca_policy_prompt


def _find_scenario(root: str, scenario_id: str | None):
    for rec in iter_scenarios(root):
        if scenario_id is None or rec.scenario_id == scenario_id:
            return rec
    raise RuntimeError(f"scenario not found: {scenario_id!r}")


def _adapter_snapshot(model: Any, adapter_name: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter_belongs_to_adapter(name, adapter_name)
    }


def _changed_names(before: dict[str, torch.Tensor], model: Any, adapter_name: str) -> list[str]:
    changed: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter_belongs_to_adapter(name, adapter_name):
            continue
        old = before.get(name)
        if old is None:
            raise KeyError(f"missing snapshot tensor {name}")
        if not torch.equal(old, parameter.detach().cpu()):
            changed.append(name)
    return changed


def _enable_nonreentrant_gradient_checkpointing(model: Any) -> str:
    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    fn = getattr(target, "gradient_checkpointing_enable", None)
    if not callable(fn):
        raise RuntimeError("Qwen base does not expose gradient_checkpointing_enable()")
    try:
        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        return "nonreentrant"
    except TypeError:
        # Retained only as an explicit compatibility fallback; current production
        # Transformers supports gradient_checkpointing_kwargs.
        fn()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return "legacy_reentrant_fallback"


def main() -> None:
    ap = argparse.ArgumentParser(description="Real Qwen bounded-context one-row backward GPU smoke")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", default=None)
    ap.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    ap.add_argument("--max_prompt_tokens", type=int, default=18_000)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.max_prompt_tokens < 1:
        raise ValueError("--max_prompt_tokens must be positive")

    rec = _find_scenario(args.processed_states, args.scenario_id)
    original = sanitize_agent_state(rec.compressed_state, mode="training_safe")
    bounded = build_bounded_agent_state(
        original,
        config=BoundedAgentStateConfig(
            max_serialized_chars=50_000,
            max_system_services=1,
            max_metric_services=64,
            max_log_services=1,
        ),
    )
    safety = agent_input_safety_report(bounded)
    if not safety.get("safe_for_training_agent"):
        raise AssertionError(f"bounded state is not training safe: {safety}")

    prompt = build_rca_policy_prompt(bounded, [], 0, 1)

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
    tokenizer = backend.tokenizer

    sampler = HFExactTokenPolicySampler(
        model,
        tokenizer,
        config=ExactTokenGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            max_prompt_tokens=None,
            seed=260826,
            prompt_format="chat_template",
            chat_role="user",
        ),
        device="cuda:0",
    )

    _, policy_info = sampler.generate(
        prompt,
        adapter_name="lora_rca",
        sample_index=0,
        group_id="qwen-bounded-backward-smoke",
    )
    prompt_ids = list(policy_info["prompt_token_ids"])
    completion_ids = list(policy_info["completion_token_ids"])
    old_logprobs = list(policy_info["old_logprobs"])
    ref_logprobs = list(policy_info["ref_logprobs"])

    if len(prompt_ids) > args.max_prompt_tokens:
        raise AssertionError(
            f"bounded prompt exceeds smoke budget: {len(prompt_ids)} > {args.max_prompt_tokens}"
        )
    if len(completion_ids) != len(old_logprobs) or len(completion_ids) != len(ref_logprobs):
        raise AssertionError("exact rollout token/logprob alignment failed")
    if policy_info.get("prompt_was_truncated"):
        raise AssertionError("bounded backward smoke must not truncate prompt tokens")

    # Rollout is complete. Release transient inference allocations before measuring
    # the differentiable forward/backward phase.
    torch.cuda.empty_cache()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    checkpoint_mode = _enable_nonreentrant_gradient_checkpointing(model)
    model.train()
    trainable_names = activate_exclusive_adapter(model, "lora_rca")
    parameters = trainable_parameter_list(model)

    rca_before = _adapter_snapshot(model, "lora_rca")
    action_before = _adapter_snapshot(model, "lora_action")
    optimizer = torch.optim.AdamW(parameters, lr=float(args.lr))
    optimizer.zero_grad(set_to_none=True)

    row = {
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        # Synthetic signal for this mechanics-only audit. Real training must use
        # the trajectory-derived rca_policy_advantage.
        "policy_advantage": 1.0,
    }

    torch.cuda.reset_peak_memory_stats()
    allocated_before_backward_gib = torch.cuda.memory_allocated() / 1024**3
    loss_obj = model_decision_loss(
        model,
        row,
        config=FactorizedGRPOConfig(kl_coeff=0.0),
        device="cuda:0",
    )
    forward_peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    loss_obj.loss.backward()
    backward_peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    grad_report = finite_selected_adapter_gradients(model, "lora_rca")
    if not grad_report.get("gradient_isolation_ok"):
        raise AssertionError(f"RCA gradient isolation failed: {grad_report}")

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("gradient norm is non-finite")
    optimizer.step()

    rca_changed = _changed_names(rca_before, model, "lora_rca")
    action_changed = _changed_names(action_before, model, "lora_action")
    if not rca_changed:
        raise AssertionError("real Qwen backward produced no RCA adapter parameter update")
    if action_changed:
        raise AssertionError(f"inactive Action adapter changed during RCA update: {action_changed[:8]}")

    ratio_mean = float(loss_obj.ratio_mean.detach().cpu())
    if abs(ratio_mean - 1.0) > 5e-3:
        raise AssertionError(f"unchanged pre-update policy ratio is not approximately one: {ratio_mean}")

    summary = {
        "status": "PASS",
        "scenario_id": rec.scenario_id,
        "model": args.model,
        "quantization": backend.quantization_mode,
        "gpu": torch.cuda.get_device_name(0),
        "bounded_prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "prompt_was_truncated": False,
        "synthetic_advantage_for_mechanics_only": True,
        "uses_real_scientific_policy_credit": False,
        "gradient_checkpointing": checkpoint_mode,
        "tail_only_training_logits": True,
        "pre_update_ratio_mean": ratio_mean,
        "trainable_rca_parameter_count": sum(int(p.numel()) for p in parameters),
        "trainable_rca_tensor_count": len(trainable_names),
        "rca_changed_tensor_count": len(rca_changed),
        "action_changed_tensor_count": len(action_changed),
        "gradient_report": grad_report,
        "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
        "allocated_before_training_forward_gib": round(allocated_before_backward_gib, 3),
        "forward_peak_cuda_memory_gib": round(forward_peak_gib, 3),
        "backward_peak_cuda_memory_gib": round(backward_peak_gib, 3),
        "model_memory_footprint_gib": (
            round(backend.model_memory_footprint_gib, 3)
            if backend.model_memory_footprint_gib is not None
            else None
        ),
        "bounded_state_projection": bounded.get("projection"),
    }
    print(json.dumps(summary, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
