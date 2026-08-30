from __future__ import annotations

"""Production-memory subclass of the synchronized factorized GRPO trainer.

State validation, rollback, optimizer ownership, checkpointing, and synchronized
bundle publication remain inherited from ``SynchronizedFactorizedGRPOTrainer``.
Only the per-role optimizer implementation is replaced with exact row-streaming
backward so long Qwen prompts do not retain one transformer graph per rollout row.
"""

from typing import Any, Sequence

from .peft_adapter_control import activate_exclusive_adapter
from .streaming_grpo_optimizer import streaming_optimizer_step
from .synchronized_grpo_trainer import (
    ROLE_TO_ADAPTER,
    SynchronizedFactorizedGRPOTrainer,
)


class StreamingSynchronizedFactorizedGRPOTrainer(SynchronizedFactorizedGRPOTrainer):
    """Synchronized trainer using memory-safe row-at-a-time role backward."""

    def _update_role(
        self,
        role: str,
        rows: Sequence[dict[str, Any]],
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        adapter_name = ROLE_TO_ADAPTER[role]
        if not signal["has_policy_gradient_signal"]:
            return {
                "role": role,
                "adapter": adapter_name,
                "updated": False,
                "skip_reason": "zero_policy_advantage_signal",
                "signal": signal,
                "kl_only_update_blocked": bool(self.config.grpo.kl_coeff > 0.0),
                "streaming_row_backward": True,
            }

        activate_exclusive_adapter(self.model, adapter_name)

        # Rollout/replay helpers intentionally leave the shared model in eval()
        # mode. Transformers gradient checkpointing is training-mode dependent;
        # entering a long Qwen backward while still in eval() silently disables
        # checkpointing and retains full-layer activations, which can exhaust a
        # 96-GiB GPU even for a single ~18k-token row. Every real optimizer role
        # update must therefore re-enter train mode after adapter activation and
        # before the differentiable forward. LoRA dropout is configured as zero,
        # so this does not introduce extra stochasticity into the policy update.
        self.model.train()

        optimizer = self.rca_optimizer if role == "rca" else self.action_optimizer
        diagnostics = streaming_optimizer_step(
            self.model,
            optimizer,
            rows,
            config=self.config.grpo,
            device=self.device,
        )
        if role == "rca":
            self.rca_update_step += 1
        else:
            self.action_update_step += 1
        return {
            "role": role,
            "adapter": adapter_name,
            "updated": True,
            "signal": signal,
            "optimizer": diagnostics,
            "streaming_row_backward": True,
            "model_training_mode_during_backward": True,
        }
