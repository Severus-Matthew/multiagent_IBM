"""Legacy GRPO trainer placeholder -- intentionally disabled for the IBM joint pipeline.

DO NOT use this module for the current factorized RCA/Action training architecture.
The previous implementation was an early prototype that:

* reconstructed completion probabilities from text instead of exact rollout token IDs;
* used ``exp(logp - logp.detach())`` as the old/new policy ratio, which is
  numerically 1 and is not a valid stored-old-policy PPO/GRPO ratio for repeated
  optimization epochs;
* formed the ratio from a sequence-mean log probability rather than token-level
  ratios;
* used ``policy_logp - reference_logp`` directly as a KL term, which is not a
  guaranteed non-negative KL estimator; and
* recomputed group advantages over local rows rather than consuming the
  trajectory-level factorized RCA/Action advantages produced by the current
  rollout pipeline.

Those choices are unacceptable for the final experiments.  The canonical rollout
path now writes exact contracts for:

    completion_token_ids
    old_logprobs                 # one value per generated token
    old_logprob_sum              # diagnostic consistency check only
    policy_advantage             # precomputed per complete trajectory group
    trajectory_id
    optimizer_group_id
    adapter_id                   # lora_rca or lora_action

The real GPU trainer must implement token-level GRPO/PPO ratios against the stored
old log probabilities, optional reference-policy KL with a non-negative estimator,
and separate RCA/Action adapter optimizers.  Until that trainer is wired and
validated, this module fails fast rather than silently running mathematically
incorrect updates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GRPOConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    clip_eps: float = 0.2
    kl_coeff: float = 0.01
    max_grad_norm: float = 1.0
    max_prompt_length: int = 1536
    max_completion_length: int = 512


class GRPOTrainer:
    """Disabled legacy entry point.

    A deliberate RuntimeError is safer than allowing an old approximate trainer
    to be mistaken for the final factorized GRPO learner.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "training.grpo_trainer.GRPOTrainer is a quarantined legacy prototype and must not be used. "
            "Use the factorized joint rollout buffers and the forthcoming audited token-level RCA/Action GRPO trainer."
        )
