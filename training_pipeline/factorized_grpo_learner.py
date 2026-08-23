from __future__ import annotations

"""Audited reference implementation of the factorized trajectory-GRPO loss.

This module is intentionally correctness-first rather than throughput-first.  It
implements the exact optimization contract emitted by ``end_to_end_loop.py``:

* exact prompt/completion token IDs from rollout time;
* one stored old-policy log probability per generated completion token;
* token-level clipped importance ratios;
* optional non-negative sampled reverse-KL penalty against a frozen reference;
* one precomputed trajectory-level ``policy_advantage`` per role;
* per-decision weight ``1 / D_role`` so long trajectories do not get more weight;
* equal averaging over complete trajectories, then equal averaging over incident
  optimizer groups.

The optimized GPU trainer may batch forwards later, but it must be numerically
identical to this reference implementation on the same inputs.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FactorizedGRPOConfig:
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    kl_coeff: float = 0.0
    max_grad_norm: float = 1.0
    log_ratio_clip: float = 60.0

    def validate(self) -> None:
        if not (0.0 <= self.clip_epsilon_low < 1.0):
            raise ValueError("clip_epsilon_low must be in [0, 1)")
        if self.clip_epsilon_high < 0.0:
            raise ValueError("clip_epsilon_high must be >= 0")
        if self.kl_coeff < 0.0:
            raise ValueError("kl_coeff must be >= 0")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be > 0")
        if self.log_ratio_clip <= 0.0:
            raise ValueError("log_ratio_clip must be > 0")


@dataclass
class DecisionLoss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_min: torch.Tensor
    ratio_max: torch.Tensor
    clip_fraction: torch.Tensor
    num_tokens: int


@dataclass
class WeightedDecisionLoss:
    decision_loss: torch.Tensor
    optimizer_group_id: str
    trajectory_id: str
    sample_weight: float


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def _as_1d_long(values: Sequence[int], *, device: torch.device | str) -> torch.Tensor:
    if not values:
        raise ValueError("token ID sequence must be non-empty")
    out = torch.tensor(list(values), dtype=torch.long, device=device)
    if (out < 0).any():
        raise ValueError("token IDs must be non-negative")
    return out


def completion_logprobs_from_causal_lm_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prompt_length: int,
    completion_length: int,
) -> torch.Tensor:
    """Gather exact completion-token log probabilities from causal-LM logits.

    ``input_ids`` must be the exact rollout-time token sequence formed by
    ``prompt_token_ids + completion_token_ids``.  For a completion token stored at
    input position ``j``, a causal LM predicts it using logits from position
    ``j-1``.  Retokenizing text is deliberately unsupported.
    """
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("reference implementation expects batch size 1")
        logits = logits[0]
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("reference implementation expects batch size 1")
        input_ids = input_ids[0]
    if logits.ndim != 2 or input_ids.ndim != 1:
        raise ValueError("expected logits [seq,vocab] and input_ids [seq]")
    if prompt_length < 1:
        raise ValueError("prompt_length must be >= 1 so the first completion token has a causal predictor")
    if completion_length < 1:
        raise ValueError("completion_length must be >= 1")
    total = prompt_length + completion_length
    if input_ids.numel() < total or logits.shape[0] < total:
        raise ValueError(
            f"sequence too short: prompt={prompt_length} completion={completion_length} "
            f"input={input_ids.numel()} logits={logits.shape[0]}"
        )

    # Completion tokens occupy input indices [P, P+T).  Their predicting logits
    # occupy [P-1, P+T-1).
    prediction_logits = logits[prompt_length - 1 : total - 1]
    target_ids = input_ids[prompt_length:total]
    if prediction_logits.shape[0] != completion_length or target_ids.numel() != completion_length:
        raise AssertionError("causal completion alignment produced the wrong number of tokens")

    log_probs = F.log_softmax(prediction_logits.float(), dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    _require_finite_tensor("completion_logprobs", gathered)
    return gathered


def sampled_reverse_kl(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    *,
    clamp: float = 60.0,
) -> torch.Tensor:
    """Non-negative sampled reverse-KL estimator per completion token.

    With ``x = log pi_ref - log pi_policy``:

        KL_hat = exp(x) - x - 1 >= 0.

    We clamp ``x`` only for numerical protection.  Reference log probabilities
    are treated as constants.
    """
    if policy_logprobs.shape != reference_logprobs.shape:
        raise ValueError("policy/reference logprob shapes must match")
    x = (reference_logprobs.detach() - policy_logprobs).clamp(-clamp, clamp)
    kl = torch.exp(x) - x - 1.0
    # Floating-point roundoff near x=0 can produce tiny negatives.
    kl = torch.clamp_min(kl, 0.0)
    _require_finite_tensor("sampled_reverse_kl", kl)
    return kl


def decision_grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: float | torch.Tensor,
    *,
    reference_logprobs: torch.Tensor | None = None,
    config: FactorizedGRPOConfig | None = None,
) -> DecisionLoss:
    """Compute the mean token-level clipped GRPO loss for one policy decision."""
    cfg = config or FactorizedGRPOConfig()
    cfg.validate()

    if new_logprobs.ndim != 1 or old_logprobs.ndim != 1:
        raise ValueError("new_logprobs and old_logprobs must be 1-D token vectors")
    if new_logprobs.numel() == 0:
        raise ValueError("decision must contain at least one completion token")
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError("new/old logprob shapes must match")
    _require_finite_tensor("new_logprobs", new_logprobs)
    _require_finite_tensor("old_logprobs", old_logprobs)

    old = old_logprobs.detach().to(device=new_logprobs.device, dtype=new_logprobs.dtype)
    adv = torch.as_tensor(advantage, dtype=new_logprobs.dtype, device=new_logprobs.device).detach()
    if adv.numel() != 1 or not torch.isfinite(adv):
        raise ValueError("advantage must be one finite scalar")

    log_ratio = (new_logprobs - old).clamp(-cfg.log_ratio_clip, cfg.log_ratio_clip)
    ratio = torch.exp(log_ratio)
    lo = 1.0 - cfg.clip_epsilon_low
    hi = 1.0 + cfg.clip_epsilon_high
    clipped_ratio = ratio.clamp(lo, hi)

    unclipped_objective = ratio * adv
    clipped_objective = clipped_ratio * adv
    surrogate_objective = torch.minimum(unclipped_objective, clipped_objective)
    token_policy_loss = -surrogate_objective

    if reference_logprobs is None or cfg.kl_coeff == 0.0:
        token_kl = torch.zeros_like(token_policy_loss)
    else:
        ref = reference_logprobs.detach().to(device=new_logprobs.device, dtype=new_logprobs.dtype)
        if ref.shape != new_logprobs.shape:
            raise ValueError("reference/new logprob shapes must match")
        token_kl = sampled_reverse_kl(new_logprobs, ref, clamp=cfg.log_ratio_clip)

    token_total_loss = token_policy_loss + cfg.kl_coeff * token_kl
    _require_finite_tensor("token_total_loss", token_total_loss)

    was_clipped = (ratio - clipped_ratio).abs() > 1e-12
    return DecisionLoss(
        loss=token_total_loss.mean(),
        policy_loss=token_policy_loss.mean(),
        kl_loss=token_kl.mean(),
        ratio_mean=ratio.mean().detach(),
        ratio_min=ratio.min().detach(),
        ratio_max=ratio.max().detach(),
        clip_fraction=was_clipped.float().mean().detach(),
        num_tokens=int(new_logprobs.numel()),
    )


def model_decision_loss(
    model: Any,
    row: dict[str, Any],
    *,
    config: FactorizedGRPOConfig | None = None,
    device: torch.device | str | None = None,
) -> DecisionLoss:
    """Compute one rollout-row loss using exact stored tokenization/logprobs.

    This is a slow reference path used for audits and small-model tests.  A future
    batched GPU implementation must match it numerically.
    """
    cfg = config or FactorizedGRPOConfig()
    cfg.validate()

    prompt_ids = row.get("prompt_token_ids")
    completion_ids = row.get("completion_token_ids")
    old_values = row.get("old_logprobs")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("row is missing exact prompt_token_ids")
    if not isinstance(completion_ids, list) or not completion_ids:
        raise ValueError("row is missing exact completion_token_ids")
    if not isinstance(old_values, list) or len(old_values) != len(completion_ids):
        raise ValueError("row old_logprobs must match completion_token_ids exactly")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    prompt = _as_1d_long(prompt_ids, device=device)
    completion = _as_1d_long(completion_ids, device=device)
    input_ids = torch.cat([prompt, completion], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    new_logprobs = completion_logprobs_from_causal_lm_logits(
        logits,
        input_ids,
        prompt_length=int(prompt.numel()),
        completion_length=int(completion.numel()),
    )

    old = torch.tensor(old_values, dtype=new_logprobs.dtype, device=device)
    ref_values = row.get("ref_logprobs")
    reference = None
    if ref_values is not None:
        if not isinstance(ref_values, list) or len(ref_values) != len(completion_ids):
            raise ValueError("row ref_logprobs must match completion_token_ids")
        reference = torch.tensor(ref_values, dtype=new_logprobs.dtype, device=device)
    elif cfg.kl_coeff > 0.0:
        raise ValueError("kl_coeff > 0 requires ref_logprobs in every training row")

    return decision_grpo_loss(
        new_logprobs,
        old,
        float(row["policy_advantage"]),
        reference_logprobs=reference,
        config=cfg,
    )


def aggregate_role_loss(records: Iterable[WeightedDecisionLoss]) -> torch.Tensor:
    """Aggregate decision losses without trajectory-length bias.

    For every optimizer group (one incident + one role):

        L_group = (1/G) sum_i sum_d w_{i,d} L_{i,d}

    where ``sum_d w_{i,d} = 1`` for each trajectory.  Multiple incident groups in
    one synchronized learner batch are then averaged equally.
    """
    groups: dict[str, dict[str, list[WeightedDecisionLoss]]] = {}
    for record in records:
        if not record.optimizer_group_id or not record.trajectory_id:
            raise ValueError("optimizer_group_id and trajectory_id are required")
        if not (0.0 < float(record.sample_weight) <= 1.0):
            raise ValueError("sample_weight must be in (0, 1]")
        groups.setdefault(record.optimizer_group_id, {}).setdefault(record.trajectory_id, []).append(record)

    if not groups:
        raise ValueError("cannot aggregate an empty role batch")

    group_losses: list[torch.Tensor] = []
    for group_id, trajectories in groups.items():
        if len(trajectories) < 2:
            raise ValueError(f"{group_id}: GRPO optimizer group must contain at least two trajectories")
        trajectory_losses: list[torch.Tensor] = []
        for trajectory_id, decisions in trajectories.items():
            weight_sum = sum(float(x.sample_weight) for x in decisions)
            if abs(weight_sum - 1.0) > 1e-8:
                raise ValueError(
                    f"{group_id}/{trajectory_id}: optimizer sample weights must sum to 1; got {weight_sum}"
                )
            loss = sum((float(x.sample_weight) * x.decision_loss for x in decisions), start=decisions[0].decision_loss * 0.0)
            _require_finite_tensor(f"trajectory_loss:{group_id}:{trajectory_id}", loss)
            trajectory_losses.append(loss)
        group_loss = torch.stack(trajectory_losses).mean()
        _require_finite_tensor(f"group_loss:{group_id}", group_loss)
        group_losses.append(group_loss)

    total = torch.stack(group_losses).mean()
    _require_finite_tensor("factorized_role_loss", total)
    return total


def role_buffer_loss(
    model: Any,
    rows: Sequence[dict[str, Any]],
    *,
    config: FactorizedGRPOConfig | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Correctness-first loss for one role buffer under the currently active adapter."""
    cfg = config or FactorizedGRPOConfig()
    cfg.validate()
    if not rows:
        raise ValueError("role buffer is empty")

    weighted: list[WeightedDecisionLoss] = []
    token_count = 0
    clip_fractions: list[float] = []
    ratio_means: list[float] = []
    kl_means: list[float] = []

    for row in rows:
        d = model_decision_loss(model, row, config=cfg, device=device)
        weighted.append(
            WeightedDecisionLoss(
                decision_loss=d.loss,
                optimizer_group_id=str(row["optimizer_group_id"]),
                trajectory_id=str(row["trajectory_id"]),
                sample_weight=float(row["optimizer_sample_weight"]),
            )
        )
        token_count += d.num_tokens
        clip_fractions.append(float(d.clip_fraction.cpu()))
        ratio_means.append(float(d.ratio_mean.cpu()))
        kl_means.append(float(d.kl_loss.detach().cpu()))

    loss = aggregate_role_loss(weighted)
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "num_rows": float(len(rows)),
        "num_completion_tokens": float(token_count),
        "mean_clip_fraction": sum(clip_fractions) / len(clip_fractions),
        "mean_ratio": sum(ratio_means) / len(ratio_means),
        "mean_sampled_kl": sum(kl_means) / len(kl_means),
    }
    return loss, diagnostics


def finite_trainable_gradient_report(model: Any) -> dict[str, Any]:
    trainable = 0
    with_grad = 0
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable += int(parameter.numel())
        if parameter.grad is not None:
            with_grad += int(parameter.numel())
            if not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
    return {
        "trainable_parameters": trainable,
        "trainable_parameters_with_grad": with_grad,
        "nonfinite_gradient_parameters": nonfinite,
        "all_finite": not nonfinite,
    }


def optimizer_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    rows: Sequence[dict[str, Any]],
    *,
    config: FactorizedGRPOConfig | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """One audited role-specific optimizer step.

    The caller is responsible for activating exactly one role adapter and for
    ensuring that only that adapter's parameters have ``requires_grad=True``.
    """
    cfg = config or FactorizedGRPOConfig()
    cfg.validate()
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = role_buffer_loss(model, rows, config=cfg, device=device)
    loss.backward()

    grad_report = finite_trainable_gradient_report(model)
    if not grad_report["all_finite"]:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"non-finite gradients: {grad_report['nonfinite_gradient_parameters']}")
    if grad_report["trainable_parameters_with_grad"] == 0:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("no trainable parameter received a gradient")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg.max_grad_norm)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("gradient norm is NaN/Inf")
    optimizer.step()

    return {
        **diagnostics,
        "grad_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
        "gradient_report": grad_report,
    }
