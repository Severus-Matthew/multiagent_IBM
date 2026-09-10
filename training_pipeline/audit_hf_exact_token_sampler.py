from __future__ import annotations

"""CPU integration audit for the exact-token HF rollout sampler.

This is the bridge test between the already-audited GRPO learner and the future
Qwen rollout backend.  It uses a tiny local Llama model with real PEFT adapters,
performs actual ``generate()``, recomputes exact old/reference token logprobs, and
verifies the stored row replays at importance ratio one.
"""

import json
import warnings
from typing import Any

import torch

from .action_loop import _build_action_policy_prompt
from .factorized_grpo_learner import FactorizedGRPOConfig, model_decision_loss
from .hf_exact_token_sampler import ExactTokenGenerationConfig, HFExactTokenPolicySampler
from .peft_adapter_control import parameter_belongs_to_adapter
from .rca_loop import build_rca_policy_prompt
from .schemas import GRPORolloutSample
from .trainable_hf_prompt_policies import (
    TrainableHFActionPromptPolicy,
    TrainableHFRCAInstructionPolicy,
)


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Deterministic local tokenizer sufficient for exercising exact token
        # capture.  Keep IDs inside the tiny model's 64-token vocabulary.
        body = [3 + (ord(ch) % 61) for ch in str(text)]
        body = body[:24] or [3]
        return ([self.bos_token_id] if add_special_tokens else []) + body

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        kept = [int(x) for x in ids if not (skip_special_tokens and int(x) in special)]
        if not kept:
            kept = [int(x) for x in ids]
        return " ".join(f"tok{x}" for x in kept)


def _build_model():
    try:
        from peft import LoraConfig, TaskType
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as exc:
        raise RuntimeError("audit requires transformers + peft") from exc

    torch.manual_seed(23)
    cfg = LlamaConfig(
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
    model = LlamaForCausalLM(cfg)
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Already found a `peft_config`.*")
        model.add_adapter(lora, adapter_name="lora_rca")
        model.add_adapter(lora, adapter_name="lora_action")

    # PEFT initializes LoRA B to zero, making adapters initially identical to the
    # frozen reference.  Give the two audit adapters small different nonzero B
    # tensors so reference-logprob capture is meaningfully exercised.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" not in name:
                continue
            if parameter_belongs_to_adapter(name, "lora_rca"):
                parameter.fill_(0.04)
            elif parameter_belongs_to_adapter(name, "lora_action"):
                parameter.fill_(-0.03)
    return model


def _assert_ratio_one(model: Any, info: dict[str, Any], completion_text: str) -> float:
    row = {
        "prompt_token_ids": info["prompt_token_ids"],
        "completion_token_ids": info["completion_token_ids"],
        "old_logprobs": info["old_logprobs"],
        "old_logprob_sum": info["old_logprob_sum"],
        "ref_logprobs": info["ref_logprobs"],
        "policy_advantage": 1.0,
        "optimizer_sample_weight": 1.0,
        "optimizer_group_id": "audit:role",
        "trajectory_id": "traj0",
        "completion": completion_text,
    }
    d = model_decision_loss(model, row, config=FactorizedGRPOConfig(kl_coeff=0.01), device="cpu")
    ratio = float(d.ratio_mean.detach().cpu())
    if abs(ratio - 1.0) > 1e-6:
        raise AssertionError(f"exact rollout replay ratio must be 1, got {ratio}")
    return ratio


def _sample_dataclass_row(prompt: str, completion: str, info: dict[str, Any]) -> dict[str, Any]:
    sample = GRPORolloutSample(
        stage="rca",
        scenario_id="audit-scenario",
        group_id="audit-group",
        sample_id="audit-sample",
        sample_index=0,
        iteration=0,
        policy_role="rca_instruction_policy",
        policy_prompt=prompt,
        completion=completion,
        completion_tokens=len(info["completion_token_ids"]),
        old_logprob_sum=None,
        old_logprobs=None,
        reward=0.0,
        reward_components={},
        advantage=None,
        group_reward_mean=None,
        group_reward_std=None,
        solver_prediction="",
        parsed_prediction=[],
        success=False,
        terminal=False,
        model_name="tiny-audit",
        policy_version="audit-v1",
        metadata={"policy_info": info},
    )
    row = sample.to_dict()
    for key in ("prompt_token_ids", "completion_token_ids", "old_logprobs", "old_logprob_sum", "ref_logprobs"):
        if row.get(key) is None:
            raise AssertionError(f"GRPORolloutSample failed to materialize {key}")
    return row


def main() -> None:
    model = _build_model()
    tokenizer = TinyTokenizer()
    sampler = HFExactTokenPolicySampler(
        model,
        tokenizer,
        config=ExactTokenGenerationConfig(
            max_new_tokens=3,
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
            max_prompt_tokens=32,
            seed=101,
        ),
        device="cpu",
    )

    rca_prompt = "RCA exact-token audit prompt"
    rca_text, rca_info = sampler.generate(rca_prompt, adapter_name="lora_rca", sample_index=0, group_id="g0")
    if rca_info["prompt_token_ids"] != tokenizer.encode(rca_prompt, add_special_tokens=True):
        raise AssertionError("stored RCA prompt IDs are not exact rollout tokenization")
    if len(rca_info["old_logprobs"]) != len(rca_info["completion_token_ids"]):
        raise AssertionError("RCA old logprob/token alignment failed")
    if len(rca_info["ref_logprobs"]) != len(rca_info["completion_token_ids"]):
        raise AssertionError("RCA reference logprob/token alignment failed")
    if not any(abs(a - b) > 1e-8 for a, b in zip(rca_info["old_logprobs"], rca_info["ref_logprobs"])):
        raise AssertionError("RCA adapter/reference logprobs unexpectedly identical; reference path not exercised")
    model.set_adapter("lora_rca")
    rca_ratio = _assert_ratio_one(model, rca_info, rca_text)
    materialized = _sample_dataclass_row(rca_prompt, rca_text, rca_info)

    action_prompt = "Action exact-token audit prompt"
    action_text, action_info = sampler.generate(action_prompt, adapter_name="lora_action", sample_index=1, group_id="g1")
    model.set_adapter("lora_action")
    action_ratio = _assert_ratio_one(model, action_info, action_text)

    # Verify stage-specific wrappers bind model input to the canonical loop prompt.
    rca_policy = TrainableHFRCAInstructionPolicy(sampler, max_iterations=3)
    state = {"services": ["svc"], "system": {}, "llm_view": {}}
    rca_policy.generate_instruction(state, [], 0, sample_index=2, group_id="g2")
    expected_rca_prompt = build_rca_policy_prompt(state, [], 0, 3)
    if rca_policy.last_policy_info.get("prompt_text") != expected_rca_prompt:
        raise AssertionError("trainable RCA wrapper used a prompt different from canonical RCA policy_prompt")

    action_policy = TrainableHFActionPromptPolicy(sampler)
    context = {
        "redacted_state": state,
        "rca_result": {"root_causes": [{"service": "svc", "fault_type": "infra_failure"}]},
        "rca_twin_gate": {"reproduction_score": 0.5},
        "current_sla": {"sla_restored": False},
        "previous_attempts": [],
        "iteration": 0,
        "max_iterations": 3,
        "sample_index": 3,
    }
    action_policy.generate(context)
    expected_action_prompt = _build_action_policy_prompt(
        agent_state=state,
        public_rca_result=context["rca_result"],
        public_rca_twin_gate=context["rca_twin_gate"],
        current_sla=context["current_sla"],
        history=[],
        iteration=0,
        max_iterations=3,
    )
    if action_policy.last_policy_info.get("prompt_text") != expected_action_prompt:
        raise AssertionError("trainable Action wrapper used a prompt different from canonical Action policy_prompt")

    # Prompt mismatch must fail at serialization time, before optimizer use.
    bad = GRPORolloutSample(
        stage="rca", scenario_id="s", group_id="g", sample_id="x", sample_index=0, iteration=0,
        policy_role="rca_instruction_policy", policy_prompt="different prompt", completion=rca_text,
        completion_tokens=1, old_logprob_sum=None, old_logprobs=None, reward=0.0, reward_components={},
        advantage=None, group_reward_mean=None, group_reward_std=None, solver_prediction="",
        parsed_prediction=[], success=False, terminal=False, model_name="tiny", policy_version="v",
        metadata={"policy_info": rca_info},
    )
    try:
        bad.to_dict()
    except ValueError:
        mismatch_blocked = True
    else:
        mismatch_blocked = False
    if not mismatch_blocked:
        raise AssertionError("policy-prompt/tokenization mismatch was not blocked")

    print(json.dumps({
        "rca": {
            "ratio_replay": rca_ratio,
            "num_prompt_tokens": len(rca_info["prompt_token_ids"]),
            "num_completion_tokens": len(rca_info["completion_token_ids"]),
            "adapter_reference_logprobs_differ": True,
        },
        "action": {
            "ratio_replay": action_ratio,
            "num_prompt_tokens": len(action_info["prompt_token_ids"]),
            "num_completion_tokens": len(action_info["completion_token_ids"]),
        },
        "canonical_rca_prompt_binding": True,
        "canonical_action_prompt_binding": True,
        "grpo_row_exact_token_fields_materialized": all(materialized.get(k) is not None for k in (
            "prompt_token_ids", "completion_token_ids", "old_logprobs", "old_logprob_sum", "ref_logprobs"
        )),
        "prompt_mismatch_blocked": True,
        "status": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
