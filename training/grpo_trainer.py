"""Legacy GRPO trainer placeholder -- intentionally disabled for the IBM joint pipeline.

DO NOT use this module for the current factorized RCA/Action training architecture.
The previous implementation was an early prototype that:

* reconstructed model inputs/completion probabilities from text instead of exact rollout token IDs;
* used ``exp(logp - logp.detach())`` as the old/new policy ratio, which is numerically 1 and is not a valid stored-old-policy PPO/GRPO ratio for repeated optimization epochs;
* formed the ratio from a sequence-mean log probability rather than token-level ratios;
* used ``policy_logp - reference_logp`` directly as a KL term, which is not a guaranteed non-negative KL estimator; and
* recomputed group advantages over local rows rather than consuming the trajectory-level factorized RCA/Action advantages produced by the current rollout pipeline.

Those choices are unacceptable for the final experiments. The canonical rollout
path now requires exact contracts for:

    prompt_token_ids             # exact rollout-time model input tokens
    completion_token_ids         # exact generated tokens
    old_logprobs                 # one value per generated token
    old_logprob_sum              # diagnostic consistency check only
    ref_logprobs                 # required when KL regularization is enabled
    policy_advantage             # precomputed per complete trajectory group
    optimizer_sample_weight      # 1 / number of role decisions in trajectory
    trajectory_id
    optimizer_group_id
    adapter_id                   # lora_rca or lora_action

The audited CPU reference loss is implemented in
``training_pipeline/factorized_grpo_learner.py``. The optimized GPU trainer must
match that reference numerically: token-level clipped old/new ratios, optional
non-negative sampled reverse-KL, exact causal token alignment, equal trajectory
weighting, and separate RCA/Action adapter optimizers.

Until the real Qwen sampler/adapters are wired and validated, this legacy module
fails fast rather than silently running mathematically incorrect updates.
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
    """Disabled legacy entry point."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "training.grpo_trainer.GRPOTrainer is a quarantined legacy prototype and must not be used. "
            "Use training_pipeline.factorized_grpo_learner as the audited reference and the factorized joint rollout buffers."
        )
