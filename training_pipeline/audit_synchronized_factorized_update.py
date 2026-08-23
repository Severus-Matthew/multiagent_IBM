from __future__ import annotations

"""CPU audit of one synchronized factorized RCA/Action GRPO update.

This audit consumes *real exact-token rollout buffers* emitted by
``audit_end_to_end_hf_exact_rollout``.  The tiny end-to-end audit intentionally
uses deterministic downstream debug components, so its within-incident policy
returns can be identical and therefore its genuine GRPO advantages can be zero.
That is a property of the debug environment, not a reason to fabricate training
signal in production.

To test only the optimizer mechanics, this audit makes an in-memory copy of each
strictly validated buffer and replaces ``policy_reward``/``policy_advantage`` with
a controlled group-relative signal that is mathematically recomputed over the
same complete trajectories.  Exact prompt/completion token IDs, old logprobs,
reference logprobs, trajectory IDs, optimizer groups, and 1/D decision weights are
left untouched.

It then performs the intended synchronized update sequence:

    frozen rollout batch
      -> LoRA_RCA update only
      -> LoRA_Action update only
      -> verify base/inactive adapter isolation

The controlled credit is audit-only and is never written back to the canonical
rollout files.
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .audit_hf_exact_token_sampler import _build_model
from .factorized_grpo_learner import (
    FactorizedGRPOConfig,
    model_decision_loss,
    optimizer_step,
)
from .grpo_dataset import load_grpo_dataset
from .grpo_math import group_relative_advantages
from .peft_adapter_control import (
    activate_exclusive_adapter,
    changed_parameter_names,
    parameter_belongs_to_adapter,
    snapshot_named_parameters,
    trainable_parameter_list,
)


def _original_signal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        gid = str(row["optimizer_group_id"])
        tid = str(row["trajectory_id"])
        groups[gid][tid] = float(row["policy_advantage"])
    nonzero_groups = 0
    zero_groups = 0
    max_abs = 0.0
    for trajectories in groups.values():
        vals = list(trajectories.values())
        local_max = max((abs(x) for x in vals), default=0.0)
        max_abs = max(max_abs, local_max)
        if local_max > 1e-12:
            nonzero_groups += 1
        else:
            zero_groups += 1
    return {
        "num_optimizer_groups": len(groups),
        "nonzero_advantage_groups": nonzero_groups,
        "zero_advantage_groups": zero_groups,
        "max_abs_policy_advantage": max_abs,
    }


def _controlled_credit_copy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inject a valid, zero-mean/sample-std-normalized audit signal in memory."""
    out = copy.deepcopy(rows)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in out:
        groups[str(row["optimizer_group_id"])][str(row["trajectory_id"])].append(row)

    for gid, trajectories in groups.items():
        tids = sorted(trajectories)
        if len(tids) < 2:
            raise AssertionError(f"{gid}: need at least two trajectories for GRPO audit")

        # Strictly increasing synthetic returns ensure a non-zero group-relative
        # signal while preserving the real trajectory grouping.  Advantages are
        # computed by the exact same helper used by the rollout pipeline.
        synthetic_returns = [float(i) for i in range(len(tids))]
        normalized = group_relative_advantages(synthetic_returns, scale_by_std=True)
        if normalized.zero_variance:
            raise AssertionError(f"{gid}: controlled audit credit unexpectedly has zero variance")

        for tid, reward, advantage in zip(tids, synthetic_returns, normalized.advantages):
            trows = trajectories[tid]
            expected_weight = 1.0 / float(len(trows))
            if abs(sum(float(r["optimizer_sample_weight"]) for r in trows) - 1.0) > 1e-10:
                raise AssertionError(f"{gid}/{tid}: decision weights do not sum to one")
            if any(abs(float(r["optimizer_sample_weight"]) - expected_weight) > 1e-10 for r in trows):
                raise AssertionError(f"{gid}/{tid}: decision weights are not 1/D_role")
            for row in trows:
                row["policy_reward"] = float(reward)
                row["policy_advantage"] = float(advantage)
                metadata = dict(row.get("metadata", {}) or {})
                metadata["controlled_credit_for_optimizer_audit_only"] = True
                metadata["original_policy_reward"] = row.get("policy_reward")
                row["metadata"] = metadata
    return out


def _ratio_stats(model: Any, rows: list[dict[str, Any]], adapter_name: str) -> dict[str, float | bool]:
    model.set_adapter(adapter_name)
    model.eval()
    cfg = FactorizedGRPOConfig(kl_coeff=0.0)
    means: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    for row in rows:
        d = model_decision_loss(model, row, config=cfg, device="cpu")
        means.append(float(d.ratio_mean.detach().cpu()))
        mins.append(float(d.ratio_min.detach().cpu()))
        maxs.append(float(d.ratio_max.detach().cpu()))
    if not means:
        raise AssertionError("cannot compute ratio statistics for empty role buffer")
    max_deviation = max(
        max(abs(x - 1.0) for x in means),
        max(abs(x - 1.0) for x in mins),
        max(abs(x - 1.0) for x in maxs),
    )
    return {
        "ratio_mean": sum(means) / len(means),
        "ratio_min": min(mins),
        "ratio_max": max(maxs),
        "max_abs_deviation_from_one": max_deviation,
        "all_ratio_one": bool(max_deviation <= 2e-6),
    }


def _assert_only_adapter_changed(changed: list[str], adapter_name: str) -> None:
    if not changed:
        raise AssertionError(f"{adapter_name}: optimizer step changed no parameters")
    wrong = [name for name in changed if not parameter_belongs_to_adapter(name, adapter_name)]
    if wrong:
        raise AssertionError(f"{adapter_name}: non-selected parameters changed: {wrong[:20]}")


def _assert_snapshot_equal_for_non_adapter(
    reference: dict[str, torch.Tensor],
    model: Any,
    *,
    allowed_adapter: str | None,
) -> None:
    changed = changed_parameter_names(reference, model)
    wrong = [
        name for name in changed
        if allowed_adapter is None or not parameter_belongs_to_adapter(name, allowed_adapter)
    ]
    if wrong:
        raise AssertionError(f"unexpected parameter changes outside {allowed_adapter}: {wrong[:20]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit one synchronized RCA/Action exact-token GRPO update on CPU")
    ap.add_argument("--rca_buffer", required=True)
    ap.add_argument("--action_buffer", required=True)
    ap.add_argument("--learning_rate", type=float, default=5e-3)
    args = ap.parse_args()

    rca_path = Path(args.rca_buffer).expanduser()
    action_path = Path(args.action_buffer).expanduser()

    # First validate the untouched canonical rollout records.  This ensures the
    # controlled audit signal is applied only after exact-token/factorized buffer
    # correctness has been established.
    rca_original = load_grpo_dataset(
        rca_path,
        require_policy_credit=True,
        require_old_logprobs=True,
    )
    action_original = load_grpo_dataset(
        action_path,
        require_policy_credit=True,
        require_old_logprobs=True,
    )

    original_rca_signal = _original_signal_summary(rca_original)
    original_action_signal = _original_signal_summary(action_original)

    rca_rows = _controlled_credit_copy(rca_original)
    action_rows = _controlled_credit_copy(action_original)

    # Recreate the deterministic tiny HF+PEFT model used by the rollout audit.
    # Exact rollout replay must start at ratio one for both roles.
    model = _build_model()
    initial_snapshot = snapshot_named_parameters(model)
    rca_pre = _ratio_stats(model, rca_rows, "lora_rca")
    action_pre = _ratio_stats(model, action_rows, "lora_action")
    if not rca_pre["all_ratio_one"] or not action_pre["all_ratio_one"]:
        raise AssertionError(f"rollout replay must start at ratio one: rca={rca_pre}, action={action_pre}")

    cfg = FactorizedGRPOConfig(
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        kl_coeff=0.0,
        max_grad_norm=1.0,
    )

    # RCA update.  Only lora_rca may change.
    activate_exclusive_adapter(model, "lora_rca")
    before_rca = snapshot_named_parameters(model)
    rca_optimizer = torch.optim.AdamW(
        trainable_parameter_list(model),
        lr=float(args.learning_rate),
        weight_decay=0.0,
    )
    rca_step = optimizer_step(model, rca_optimizer, rca_rows, config=cfg, device="cpu")
    rca_changed = changed_parameter_names(before_rca, model)
    _assert_only_adapter_changed(rca_changed, "lora_rca")
    _assert_snapshot_equal_for_non_adapter(initial_snapshot, model, allowed_adapter="lora_rca")

    rca_post_rca = _ratio_stats(model, rca_rows, "lora_rca")
    if rca_post_rca["all_ratio_one"]:
        raise AssertionError("RCA update left every old/new importance ratio at one")

    # The Action policy was part of the same frozen rollout batch.  Updating RCA
    # must not invalidate Action's old logprobs because the shared base is frozen
    # and the Action adapter is independent.
    action_pre_action = _ratio_stats(model, action_rows, "lora_action")
    if not action_pre_action["all_ratio_one"]:
        raise AssertionError(
            "RCA update changed Action rollout probabilities before Action update; adapter isolation failed"
        )

    # Action update.  Switching roles globally clears RCA gradients and freezes
    # the already-updated RCA adapter.
    activate_exclusive_adapter(model, "lora_action")
    before_action = snapshot_named_parameters(model)
    action_optimizer = torch.optim.AdamW(
        trainable_parameter_list(model),
        lr=float(args.learning_rate),
        weight_decay=0.0,
    )
    action_step = optimizer_step(model, action_optimizer, action_rows, config=cfg, device="cpu")
    action_changed = changed_parameter_names(before_action, model)
    _assert_only_adapter_changed(action_changed, "lora_action")

    action_post = _ratio_stats(model, action_rows, "lora_action")
    if action_post["all_ratio_one"]:
        raise AssertionError("Action update left every old/new importance ratio at one")

    # Action update must not alter the RCA policy that was just trained.
    rca_post_action = _ratio_stats(model, rca_rows, "lora_rca")
    if abs(
        float(rca_post_action["ratio_mean"]) - float(rca_post_rca["ratio_mean"])
    ) > 2e-6 or abs(
        float(rca_post_action["ratio_min"]) - float(rca_post_rca["ratio_min"])
    ) > 2e-6 or abs(
        float(rca_post_action["ratio_max"]) - float(rca_post_rca["ratio_max"])
    ) > 2e-6:
        raise AssertionError(
            "Action update changed RCA policy probabilities; separate-adapter update contract failed"
        )

    # Across both sequential role updates, the only allowed changes relative to
    # the initial model are the two named LoRA adapters.  The shared base remains
    # bitwise unchanged.
    final_changed = changed_parameter_names(initial_snapshot, model)
    bad_final = [
        name for name in final_changed
        if not (
            parameter_belongs_to_adapter(name, "lora_rca")
            or parameter_belongs_to_adapter(name, "lora_action")
        )
    ]
    if bad_final:
        raise AssertionError(f"shared base changed across synchronized update: {bad_final[:20]}")

    print(json.dumps({
        "original_rollout_signal": {
            "rca": original_rca_signal,
            "action": original_action_signal,
            "note": (
                "The tiny end-to-end debug environment may produce zero within-incident advantages because "
                "its heuristic/fixed downstream components can ignore sampled prompt variation. Controlled "
                "credit below tests optimizer mechanics only and is never persisted."
            ),
        },
        "controlled_credit_audit_only": True,
        "pre_update_replay": {
            "rca": rca_pre,
            "action": action_pre,
        },
        "rca_update": {
            "optimizer": rca_step,
            "num_changed_parameter_tensors": len(rca_changed),
            "changed_only_lora_rca": True,
            "post_update_ratio": rca_post_rca,
        },
        "action_before_its_update_after_rca_update": action_pre_action,
        "action_update": {
            "optimizer": action_step,
            "num_changed_parameter_tensors": len(action_changed),
            "changed_only_lora_action": True,
            "post_update_ratio": action_post,
        },
        "rca_unchanged_by_action_update": True,
        "shared_base_bitwise_unchanged": True,
        "synchronized_update_order": ["lora_rca", "lora_action", "publish_together"],
        "status": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
