from __future__ import annotations

from typing import Any

from .rca_loop import build_rca_policy_prompt


class QwenRCAInstructionPolicy:
    """Qwen free-form RCA instruction policy interface.

    The default mode is dry-run/stub mode so the comparison pipeline can be
    tested without loading a GPU model. Real Qwen inference/training will be
    enabled in a later patch after the policy-selection and logging plumbing is
    stable.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        adapter_path: str | None = None,
        dry_run: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.dry_run = dry_run
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

    def generate_instruction(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int = 0,
        group_id: str | None = None,
    ) -> str:
        if self.dry_run:
            return self._dry_run_instruction(compressed_state, history, iteration, sample_index)
        return self._generate_with_qwen(compressed_state, history, iteration, sample_index)

    def _dry_run_instruction(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int,
    ) -> str:
        variants = [
            "Act as a Qwen-generated RCA prompt. Prioritize explicit pod readiness, endpoint, crashloop, and scheduling signals before noisy logs.",
            "Act as a Qwen-generated RCA prompt. Build a dependency-cascade explanation from suspicious trace edges, then identify the upstream root cause.",
            "Act as a Qwen-generated RCA prompt. Prioritize log error services and datastore/auth/config symptoms, then verify with service health.",
            "Act as a Qwen-generated RCA prompt. Consider whether independent symptoms require multiple root causes; avoid repeating failed guesses.",
        ]
        retry = " Use previous non-leaking feedback to avoid repeated wrong guesses." if history else ""
        return (
            variants[sample_index % len(variants)]
            + " Use only redacted telemetry. Output only service::fault_type, one root cause per line."
            + retry
        )

    def _generate_with_qwen(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int,
    ) -> str:
        raise NotImplementedError(
            "Real Qwen RCA instruction generation is not wired yet. "
            "Use --qwen_dry_run for plumbing tests, then enable this in the Qwen training patch."
        )

    def build_model_input(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        max_iterations: int = 5,
    ) -> str:
        """Expose the exact prompt that real Qwen will later condition on."""
        return build_rca_policy_prompt(compressed_state, history, iteration, max_iterations)
