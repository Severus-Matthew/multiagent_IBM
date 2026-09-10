from __future__ import annotations

"""CPU audit of the real Hugging Face/PEFT two-adapter training path.

This test uses a randomly initialized tiny Llama causal LM created entirely from
configuration, so it downloads no model weights.  It verifies the same mechanics
we will use for Qwen:

* one shared frozen base model;
* two named LoRA adapters: lora_rca and lora_action;
* exact rollout-time prompt/completion token IDs;
* exact stored old-policy token log probabilities;
* ratio=1 before an update;
* reference factorized-GRPO loss/backward;
* only the selected adapter changes after its optimizer step;
* the shared base and the other role adapter remain bitwise unchanged.

A failure is a blocker for wiring the large Qwen model.
"""

import json
from typing import Any

import torch

from .factorized_grpo_learner import (
    FactorizedGRPOConfig,
    completion_logprobs_from_causal_lm_logits,
    model_decision_loss,
    role_buffer_loss,
)
from .peft_adapter_control import (
    activate_exclusive_adapter,
    changed_parameter_names,
    finite_selected_adapter_gradients,
    parameter_belongs_to_adapter,
    snapshot_named_parameters,
    trainable_parameter_list,
)


def _imports():
    try:
        import transformers
        import peft
        from peft import LoraConfig, TaskType
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "This audit requires current Hugging Face Transformers and PEFT. "
            "Install/upgrade them in .venv-aiops312 before rerunning."
        ) from exc
    return transformers, peft, LoraConfig, TaskType, LlamaConfig, LlamaForCausalLM


def _build_model():
    transformers, peft, LoraConfig, TaskType, LlamaConfig, LlamaForCausalLM = _imports()
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model.add_adapter(lora, adapter_name="lora_rca")
    model.add_adapter(lora, adapter_name="lora_action")
    return model, transformers.__version__, peft.__version__


def _exact_logprobs(model: Any, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        gathered = completion_logprobs_from_causal_lm_logits(
            logits,
            ids,
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
        )
    return [float(x) for x in gathered.detach().cpu()]


def _row(
    model: Any,
    *,
    group_id: str,
    trajectory_id: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    advantage: float,
    weight: float,
) -> dict[str, Any]:
    old = _exact_logprobs(model, prompt_ids, completion_ids)
    return {
        "prompt_token_ids": list(prompt_ids),
        "completion_token_ids": list(completion_ids),
        "old_logprobs": old,
        "old_logprob_sum": float(sum(old)),
        "ref_logprobs": list(old),
        "policy_advantage": float(advantage),
        "optimizer_sample_weight": float(weight),
        "optimizer_group_id": group_id,
        "trajectory_id": trajectory_id,
    }


def _role_rows(model: Any, role: str) -> list[dict[str, Any]]:
    # Same initial incident, two complete trajectories.  Use different completion
    # tokens so +A and -A do not trivially cancel at the parameter level.
    group = f"tiny-incident:{role}"
    return [
        _row(
            model,
            group_id=group,
            trajectory_id="traj0",
            prompt_ids=[1, 5, 7],
            completion_ids=[11, 13],
            advantage=1.0,
            weight=1.0,
        ),
        _row(
            model,
            group_id=group,
            trajectory_id="traj1",
            prompt_ids=[1, 5, 7],
            completion_ids=[17, 19],
            advantage=-1.0,
            weight=1.0,
        ),
    ]


def _audit_one_role(model: Any, adapter_name: str) -> dict[str, Any]:
    trainable_names = activate_exclusive_adapter(model, adapter_name)
    model.eval()
    rows = _role_rows(model, adapter_name)
    cfg = FactorizedGRPOConfig(kl_coeff=0.0)

    # Exact replay must start at importance ratio one.
    first = model_decision_loss(model, rows[0], config=cfg)
    ratio_before = float(first.ratio_mean.detach().cpu())
    if abs(ratio_before - 1.0) > 1e-6:
        raise AssertionError(f"{adapter_name}: unchanged rollout policy ratio != 1: {ratio_before}")

    before = snapshot_named_parameters(model)
    optimizer = torch.optim.AdamW(trainable_parameter_list(model), lr=5e-3, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = role_buffer_loss(model, rows, config=cfg, device="cpu")
    if not torch.isfinite(loss):
        raise AssertionError(f"{adapter_name}: non-finite role loss")
    loss.backward()

    grad_report = finite_selected_adapter_gradients(model, adapter_name)
    if not grad_report["gradient_isolation_ok"]:
        raise AssertionError(f"{adapter_name}: gradient isolation failed: {grad_report}")

    optimizer.step()
    changed = changed_parameter_names(before, model)
    if not changed:
        raise AssertionError(f"{adapter_name}: optimizer step changed no parameters")

    wrong_changes = [
        name for name in changed
        if not parameter_belongs_to_adapter(name, adapter_name)
    ]
    if wrong_changes:
        raise AssertionError(
            f"{adapter_name}: base/inactive adapter changed during optimizer step: {wrong_changes[:20]}"
        )

    if not any(parameter_belongs_to_adapter(name, adapter_name) for name in changed):
        raise AssertionError(f"{adapter_name}: selected adapter did not change")

    return {
        "adapter": adapter_name,
        "ratio_before_update": ratio_before,
        "role_loss": float(loss.detach().cpu()),
        "num_trainable_parameter_tensors": len(trainable_names),
        "num_changed_parameter_tensors": len(changed),
        "changed_parameters_only_selected_adapter": True,
        "gradient_report": grad_report,
        "diagnostics": diagnostics,
    }


def main() -> None:
    model, transformers_version, peft_version = _build_model()

    # Snapshot before any role update to verify the first role never modifies the
    # second adapter or base. _audit_one_role checks that directly.  Then perform
    # an Action update on the already RCA-updated shared object and verify the RCA
    # adapter remains frozen during the Action step.
    rca = _audit_one_role(model, "lora_rca")
    action = _audit_one_role(model, "lora_action")

    # Final trainability contract: only whichever role was activated last should
    # remain trainable.
    active_trainables = [name for name, p in model.named_parameters() if p.requires_grad]
    if not active_trainables or any(
        not parameter_belongs_to_adapter(name, "lora_action") for name in active_trainables
    ):
        raise AssertionError("exclusive adapter trainability contract failed after Action activation")

    print(json.dumps({
        "transformers_version": transformers_version,
        "peft_version": peft_version,
        "rca_adapter_update": rca,
        "action_adapter_update": action,
        "shared_base_frozen": True,
        "inactive_adapter_frozen_each_step": True,
        "exact_old_policy_replay_ratio_one": True,
        "status": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
