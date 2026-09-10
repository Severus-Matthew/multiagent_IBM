"""Training pipeline for RCA and remediation prompt optimization.

Stage 0: dataset loading and redaction checks.
Stage 1: RCA self-prompting loop.
Stage 2: digital-twin construction and RCA validation.
Stage 3: action prompt-optimizer loop.
Stage 4: trainer-facing rollout records for GRPO/RL.
"""

__all__ = [
    "data_loader",
    "ground_truth",
    "rca_loop",
    "rca_reward",
    "action_loop",
    "action_reward",
]
