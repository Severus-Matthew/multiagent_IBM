from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GroupAdvantageResult:
    values: tuple[float, ...]
    mean: float
    std: float
    advantages: tuple[float, ...]
    zero_variance: bool
    scale_by_std: bool
    std_correction: int
    normalization_epsilon: float


def group_relative_advantages(
    values: Iterable[float],
    *,
    scale_by_std: bool = True,
    std_floor: float = 1e-12,
    normalization_epsilon: float = 1e-4,
    std_correction: int = 1,
) -> GroupAdvantageResult:
    """Compute group-relative advantages with explicit standard-deviation semantics.

    For group rewards/returns ``r_1..r_G``:

        centered_i = r_i - mean(r)
        A_i = centered_i                                      if scale_by_std=False
        A_i = centered_i / (std(r) + normalization_epsilon)  otherwise

    ``std_correction=1`` uses the sample standard deviation (denominator ``G-1``),
    matching ``torch.std(..., correction=1)`` used in current common GRPO/TRL
    implementations. The correction is explicit so future code cannot silently
    change the convention. If the group has fewer than two samples or variance is
    effectively zero, all advantages are exactly zero.
    """
    vals = tuple(float(x) for x in values)
    if not vals:
        return GroupAdvantageResult((), 0.0, 0.0, (), True, bool(scale_by_std), std_correction, normalization_epsilon)
    if any(not math.isfinite(x) for x in vals):
        raise ValueError(f"non-finite value in GRPO group: {vals}")
    if std_floor <= 0:
        raise ValueError("std_floor must be > 0")
    if normalization_epsilon < 0:
        raise ValueError("normalization_epsilon must be >= 0")
    if std_correction not in {0, 1}:
        raise ValueError("std_correction must be 0 (population) or 1 (sample)")

    mu = sum(vals) / len(vals)
    centered = tuple(x - mu for x in vals)

    if len(vals) <= std_correction:
        return GroupAdvantageResult(
            vals, mu, 0.0, tuple(0.0 for _ in vals), True,
            bool(scale_by_std), std_correction, normalization_epsilon,
        )

    denom = len(vals) - std_correction
    var = max(0.0, sum(x * x for x in centered) / denom)
    sigma = math.sqrt(var)

    if len(vals) < 2 or sigma < std_floor:
        adv = tuple(0.0 for _ in vals)
        return GroupAdvantageResult(
            vals, mu, sigma, adv, True,
            bool(scale_by_std), std_correction, normalization_epsilon,
        )

    if scale_by_std:
        divisor = sigma + normalization_epsilon
        adv = tuple(x / divisor for x in centered)
    else:
        adv = centered

    if any(not math.isfinite(x) for x in adv):
        raise ValueError(f"non-finite advantage computed from values={vals}")

    # Centering is a mathematical invariant even when an epsilon is added to the
    # denominator. Fail loudly if a future refactor breaks it.
    adv_mean = sum(adv) / len(adv)
    if abs(adv_mean) > 1e-10:
        raise AssertionError(f"group advantages are not centered: mean={adv_mean}")

    return GroupAdvantageResult(
        vals, mu, sigma, adv, False,
        bool(scale_by_std), std_correction, normalization_epsilon,
    )


def clipped_grpo_surrogate(
    new_logprob: float,
    old_logprob: float,
    advantage: float,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> dict[str, float | bool]:
    """Pure-math token-level GRPO/PPO clipped surrogate diagnostic.

        ratio = exp(log pi_theta - log pi_old)
        objective = min(ratio*A, clip(ratio, 1-eps_low, 1+eps_high)*A)
        loss = -objective
    """
    vals = (new_logprob, old_logprob, advantage, epsilon_low, epsilon_high)
    if any(not math.isfinite(float(x)) for x in vals):
        raise ValueError(f"non-finite clipped-surrogate input: {vals}")
    if not (0.0 <= epsilon_low < 1.0):
        raise ValueError("epsilon_low must be in [0, 1)")
    if epsilon_high < 0.0:
        raise ValueError("epsilon_high must be >= 0")

    log_ratio = float(new_logprob) - float(old_logprob)
    ratio = math.exp(max(-60.0, min(60.0, log_ratio)))
    lo = 1.0 - float(epsilon_low)
    hi = 1.0 + float(epsilon_high)
    clipped_ratio = min(hi, max(lo, ratio))
    unclipped = ratio * float(advantage)
    clipped = clipped_ratio * float(advantage)
    objective = min(unclipped, clipped)
    return {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "unclipped_objective": unclipped,
        "clipped_objective": clipped,
        "objective": objective,
        "loss": -objective,
        "was_clipped": abs(ratio - clipped_ratio) > 1e-12,
    }


def schulman_reverse_kl_estimate(policy_logprob: float, reference_logprob: float) -> float:
    """Non-negative sampled-token KL estimator used by modern GRPO code paths.

    Let ``x = log pi_ref - log pi_policy``. Then

        KL_hat = exp(x) - x - 1

    which is non-negative and is zero when the two token probabilities are equal.
    """
    p = float(policy_logprob)
    r = float(reference_logprob)
    if not math.isfinite(p) or not math.isfinite(r):
        raise ValueError("KL log-probabilities must be finite")
    x = r - p
    estimate = math.exp(max(-60.0, min(60.0, x))) - x - 1.0
    if estimate < 0.0 and estimate > -1e-12:
        estimate = 0.0
    if estimate < 0.0:
        raise AssertionError(f"KL estimator became negative: {estimate}")
    return estimate
