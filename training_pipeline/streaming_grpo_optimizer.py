from __future__ import annotations

"""Memory-safe row-streaming optimizer for factorized trajectory GRPO.

The reference learner intentionally materializes every decision graph before
aggregating the role loss.  That is useful for mathematical audits but is not
viable for 30B Qwen with ~18k-token prompts.  This module preserves the exact
same objective while backpropagating one decision at a time, so each row's
transformer activations are released immediately after backward.

For optimizer group g with G_g trajectories and M incident groups, row d from
trajectory i is scaled by

    (1 / M) * (1 / G_g) * optimizer_sample_weight_{i,d}.

Since each trajectory's decision weights sum to one, this is exactly the same as
``aggregate_role_loss`` in ``factorized_grpo_learner.py``.
"""

from collections import defaultdict
from typing import Any, Sequence

import torch

from .factorized_grpo_learner import (
    FactorizedGRPOConfig,
    finite_trainable_gradient_report,
    model_decision_loss,
)


def _row_backward_scales(rows: Sequence[dict[str, Any]]) -> list[float]:
    if not rows:
        raise ValueError("role buffer is empty")

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        gid = str(row.get("optimizer_group_id") or "")
        tid = str(row.get("trajectory_id") or "")
        if not gid or not tid:
            raise ValueError("optimizer_group_id and trajectory_id are required")
        weight = float(row.get("optimizer_sample_weight", 0.0) or 0.0)
        if not (0.0 < weight <= 1.0):
            raise ValueError("optimizer_sample_weight must be in (0, 1]")
        grouped[gid][tid].append(index)

    num_groups = len(grouped)
    if num_groups < 1:
        raise ValueError("role buffer contains no optimizer groups")

    scales = [0.0] * len(rows)
    for gid, trajectories in grouped.items():
        if len(trajectories) < 2:
            raise ValueError(f"{gid}: GRPO optimizer group must contain at least two trajectories")
        trajectory_factor = 1.0 / float(len(trajectories))
        group_factor = 1.0 / float(num_groups)
        for tid, indices in trajectories.items():
            weight_sum = sum(float(rows[i]["optimizer_sample_weight"]) for i in indices)
            if abs(weight_sum - 1.0) > 1e-8:
                raise ValueError(
                    f"{gid}/{tid}: optimizer sample weights must sum to 1; got {weight_sum}"
                )
            for i in indices:
                scales[i] = (
                    group_factor
                    * trajectory_factor
                    * float(rows[i]["optimizer_sample_weight"])
                )

    if any(scale <= 0.0 for scale in scales):
        raise AssertionError("failed to assign a positive backward scale to every row")
    return scales


def streaming_role_backward(
    model: Any,
    rows: Sequence[dict[str, Any]],
    *,
    config: FactorizedGRPOConfig | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Backpropagate the exact aggregate role objective one decision graph at a time.

    The caller must zero gradients before this function.  No optimizer step is
    performed here, which makes the helper directly auditable against the
    reference aggregate loss.
    """
    cfg = config or FactorizedGRPOConfig()
    cfg.validate()
    scales = _row_backward_scales(rows)

    total_loss = 0.0
    token_count = 0
    clip_sum = 0.0
    ratio_sum = 0.0
    kl_sum = 0.0

    for row_index, (row, scale) in enumerate(zip(rows, scales)):
        decision = model_decision_loss(model, row, config=cfg, device=device)
        scaled_loss = decision.loss * float(scale)
        if not torch.isfinite(scaled_loss).all():
            raise FloatingPointError(f"scaled decision loss is non-finite at row {row_index}")

        total_loss += float(scaled_loss.detach().cpu())
        token_count += int(decision.num_tokens)
        clip_sum += float(decision.clip_fraction.detach().cpu())
        ratio_sum += float(decision.ratio_mean.detach().cpu())
        kl_sum += float(decision.kl_loss.detach().cpu())

        # Crucially, do not retain this graph until the end of the role buffer.
        scaled_loss.backward()
        del scaled_loss, decision

    n = float(len(rows))
    return {
        "loss": float(total_loss),
        "num_rows": float(len(rows)),
        "num_completion_tokens": float(token_count),
        "mean_clip_fraction": clip_sum / n,
        "mean_ratio": ratio_sum / n,
        "mean_sampled_kl": kl_sum / n,
        "streaming_row_backward": True,
        "retains_all_row_graphs": False,
        "loss_aggregation": "equal_incident_groups_equal_trajectories_1_over_D_role",
    }


def streaming_optimizer_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    rows: Sequence[dict[str, Any]],
    *,
    config: FactorizedGRPOConfig | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """One memory-safe exact factorized-GRPO optimizer step for an active role."""
    cfg = config or FactorizedGRPOConfig()
    cfg.validate()
    optimizer.zero_grad(set_to_none=True)

    diagnostics = streaming_role_backward(
        model,
        rows,
        config=cfg,
        device=device,
    )

    grad_report = finite_trainable_gradient_report(model)
    if not grad_report["all_finite"]:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"non-finite gradients: {grad_report['nonfinite_gradient_parameters']}"
        )
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
