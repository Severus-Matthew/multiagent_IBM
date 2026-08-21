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


def group_relative_advantages(
    values: Iterable[float],
    *,
    scale_by_std: bool = True,
    std_floor: float = 1e-8,
) -> GroupAdvantageResult:
    """Compute a GRPO-style group-relative baseline/advantage exactly once.

    For group rewards/returns r_1..r_G:

        A_i = r_i - mean(r)                              if scale_by_std=False
        A_i = (r_i - mean(r)) / std_population(r)        otherwise

    We use population standard deviation (denominator G), matching the common
    group-normalization implementation used in GRPO code paths. If the group has
    fewer than two samples or its variance is below `std_floor`, all advantages
    are set to zero. This is intentional: a constant-reward group contains no
    relative learning signal and must not create numerically amplified noise.
    """
    vals = tuple(float(x) for x in values)
    if not vals:
        return GroupAdvantageResult((), 0.0, 0.0, (), True, bool(scale_by_std))
    if any(not math.isfinite(x) for x in vals):
        raise ValueError(f"non-finite value in GRPO group: {vals}")
    if std_floor <= 0:
        raise ValueError("std_floor must be > 0")

    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / len(vals)
    # Protect against tiny negative roundoff.
    var = max(0.0, var)
    sigma = math.sqrt(var)

    if len(vals) < 2 or sigma < std_floor:
        adv = tuple(0.0 for _ in vals)
        return GroupAdvantageResult(vals, mu, sigma, adv, True, bool(scale_by_std))

    if scale_by_std:
        adv = tuple((x - mu) / sigma for x in vals)
    else:
        adv = tuple(x - mu for x in vals)

    if any(not math.isfinite(x) for x in adv):
        raise ValueError(f"non-finite advantage computed from values={vals}")

    # Centering is a mathematical invariant of both modes; fail loudly if a code
    # change breaks it beyond floating-point tolerance.
    adv_mean = sum(adv) / len(adv)
    if abs(adv_mean) > 1e-10:
        raise AssertionError(f"group advantages are not centered: mean={adv_mean}")

    return GroupAdvantageResult(vals, mu, sigma, adv, False, bool(scale_by_std))


def clipped_grpo_surrogate(
    new_logprob: float,
    old_logprob: float,
    advantage: float,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> dict[str, float | bool]:
    """Pure-math token-level GRPO/PPO clipped surrogate diagnostic.

    This helper is not a trainer. It exists so the eventual torch implementation
    can be unit-tested against an unambiguous scalar reference:

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
    # Clamp only for safe diagnostic exponentiation. A real trainer should monitor
    # extreme ratios and reject pathological batches rather than silently rely on
    # this numerical guard.
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

    Let log_ratio_ref_over_policy = log pi_ref - log pi_policy. Then

        KL_hat = exp(log_ratio_ref_over_policy)
                 - log_ratio_ref_over_policy - 1

    which is >= 0 up to floating-point tolerance and equals zero when the two
    token probabilities are equal.
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
