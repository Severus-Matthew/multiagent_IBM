from __future__ import annotations

"""Trainable RCA/Action prompt-policy wrappers over one shared HF sampler."""

from typing import Any

from .hf_exact_token_sampler import HFExactTokenPolicySampler
from .rca_loop import build_rca_policy_prompt


class TrainableHFRCAInstructionPolicy:
    def __init__(
        self,
        sampler: HFExactTokenPolicySampler,
        *,
        adapter_name: str = "lora_rca",
        max_iterations: int = 5,
    ) -> None:
        self.sampler = sampler
        self.adapter_name = adapter_name
        self.max_iterations = int(max_iterations)
        self.last_policy_info: dict[str, Any] = {}

    def generate_from_prompt(
        self,
        policy_prompt: str,
        *,
        sample_index: int = 0,
        group_id: str | None = None,
        **_: Any,
    ) -> str:
        text, info = self.sampler.generate(
            policy_prompt,
            adapter_name=self.adapter_name,
            sample_index=sample_index,
            group_id=group_id,
        )
        self.last_policy_info = info
        return text

    def generate_instruction(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int = 0,
        group_id: str | None = None,
    ) -> str:
        # Backward-compatible direct call.  The canonical RCA loop uses
        # generate_from_prompt() so the exact stored policy_prompt is guaranteed to
        # be the exact text presented to the model.
        prompt = build_rca_policy_prompt(
            compressed_state,
            history,
            iteration,
            self.max_iterations,
        )
        return self.generate_from_prompt(
            prompt,
            sample_index=sample_index,
            group_id=group_id,
        )


class TrainableHFActionPromptPolicy:
    def __init__(
        self,
        sampler: HFExactTokenPolicySampler,
        *,
        adapter_name: str = "lora_action",
    ) -> None:
        self.sampler = sampler
        self.adapter_name = adapter_name
        self.last_policy_info: dict[str, Any] = {}

    def generate_from_prompt(
        self,
        policy_prompt: str,
        *,
        sample_index: int = 0,
        group_id: str | None = None,
        context: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        del context
        text, info = self.sampler.generate(
            policy_prompt,
            adapter_name=self.adapter_name,
            sample_index=sample_index,
            group_id=group_id,
        )
        self.last_policy_info = info
        return text

    def generate(self, context: dict[str, Any]) -> str:
        raise RuntimeError(
            "TrainableHFActionPromptPolicy must be called through the canonical action loop's "
            "generate_from_prompt() path so model input and stored policy_prompt are identical."
        )
