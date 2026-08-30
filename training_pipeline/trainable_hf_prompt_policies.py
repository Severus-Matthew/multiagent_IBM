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
        # The canonical RCA loop constructs this same prompt.  The driver must set
        # max_iterations consistently with the episode configuration.
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
        # Backward-compatible with the existing action-loop protocol while still
        # guaranteeing that the model sees exactly the canonical policy_prompt
        # that the loop stores in the rollout row.
        from .action_loop import _build_action_policy_prompt

        prompt = _build_action_policy_prompt(
            agent_state=context.get("redacted_state", {}) or {},
            public_rca_result=context.get("rca_result", {}) or {},
            public_rca_twin_gate=context.get("rca_twin_gate", {}) or {},
            current_sla=context.get("current_sla", {}) or {},
            history=context.get("previous_attempts", []) or [],
            iteration=int(context.get("iteration", 0) or 0),
            max_iterations=int(context.get("max_iterations", 1) or 1),
        )
        return self.generate_from_prompt(
            prompt,
            sample_index=int(context.get("sample_index", 0) or 0),
            group_id=context.get("group_id"),
            context=context,
        )
