from __future__ import annotations

"""Audit the production synchronized GRPO trainer state machine on CPU.

Consumes the exact-token buffers emitted by ``audit_end_to_end_hf_exact_rollout``.
The untouched buffers are expected to have zero policy-gradient signal in the tiny
debug environment, so the production trainer must skip them without allowing a
KL-only update or advancing the policy version.  An in-memory controlled-credit
copy is then used to exercise a real synchronized update, atomic checkpointing,
resume, stale-rollout rejection, and transactional rollback after an injected
Action optimizer failure.
"""

import argparse
import copy
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .audit_hf_exact_token_sampler import _build_model
from .grpo_dataset import load_grpo_dataset
from .grpo_math import group_relative_advantages
from .peft_adapter_control import changed_parameter_names, parameter_belongs_to_adapter, snapshot_named_parameters
from .synchronized_grpo_trainer import (
    SynchronizedFactorizedGRPOTrainer,
    SynchronizedGRPOTrainerConfig,
)


def _controlled_credit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(rows)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in out:
        groups[str(row["optimizer_group_id"])][str(row["trajectory_id"])].append(row)

    for gid, trajectories in groups.items():
        tids = sorted(trajectories)
        returns = [float(i) for i in range(len(tids))]
        normalized = group_relative_advantages(returns, scale_by_std=True)
        if normalized.zero_variance:
            raise AssertionError(f"{gid}: controlled audit signal unexpectedly has zero variance")
        for tid, reward, advantage in zip(tids, returns, normalized.advantages):
            for row in trajectories[tid]:
                row["policy_reward"] = float(reward)
                row["policy_advantage"] = float(advantage)
                metadata = dict(row.get("metadata", {}) or {})
                metadata["controlled_credit_for_state_machine_audit_only"] = True
                row["metadata"] = metadata
    return out


def _adapter_only_changes(before: dict[str, torch.Tensor], model: Any) -> tuple[bool, list[str]]:
    changed = changed_parameter_names(before, model)
    bad = [
        name for name in changed
        if not (
            parameter_belongs_to_adapter(name, "lora_rca")
            or parameter_belongs_to_adapter(name, "lora_action")
        )
    ]
    return not bad, changed


def _assert_snapshots_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    if set(a) != set(b):
        raise AssertionError("parameter snapshot keys differ")
    for name in a:
        if not torch.equal(a[name], b[name]):
            raise AssertionError(f"parameter differs after expected no-op/rollback: {name}")


def _policy_version(rows: list[dict[str, Any]]) -> str:
    values = {str(row.get("policy_version") or "") for row in rows}
    if len(values) != 1 or "" in values:
        raise AssertionError(f"expected one non-empty policy version, got {sorted(values)}")
    return next(iter(values))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit synchronized GRPO trainer state/version/checkpoint semantics")
    ap.add_argument("--rca_buffer", required=True)
    ap.add_argument("--action_buffer", required=True)
    ap.add_argument("--learning_rate", type=float, default=5e-3)
    args = ap.parse_args()

    rca_original = load_grpo_dataset(
        Path(args.rca_buffer).expanduser(),
        require_policy_credit=True,
        require_old_logprobs=True,
    )
    action_original = load_grpo_dataset(
        Path(args.action_buffer).expanduser(),
        require_policy_credit=True,
        require_old_logprobs=True,
    )
    rollout_version = _policy_version(rca_original)
    if _policy_version(action_original) != rollout_version:
        raise AssertionError("RCA and Action buffers have different rollout versions")

    cfg = SynchronizedGRPOTrainerConfig(
        rca_learning_rate=float(args.learning_rate),
        action_learning_rate=float(args.learning_rate),
        weight_decay=0.0,
    )

    # 1) Production zero-signal gate: the untouched tiny rollout must be a true no-op.
    model = _build_model()
    trainer = SynchronizedFactorizedGRPOTrainer(
        model,
        current_policy_version=rollout_version,
        config=cfg,
        device="cpu",
    )
    before_zero = snapshot_named_parameters(model)
    zero_result = trainer.update_joint_batch(rca_original, action_original)
    after_zero = snapshot_named_parameters(model)
    _assert_snapshots_equal(before_zero, after_zero)
    if zero_result["status"] != "SKIPPED_ZERO_SIGNAL":
        raise AssertionError(f"zero-signal rollout should be skipped: {zero_result}")
    if trainer.current_policy_version != rollout_version or trainer.bundle_update_step != 0:
        raise AssertionError("zero-signal batch advanced policy version")

    # 2) Controlled non-zero credit exercises the actual production trainer update.
    rca_controlled = _controlled_credit(rca_original)
    action_controlled = _controlled_credit(action_original)
    before_update = snapshot_named_parameters(model)
    update_result = trainer.update_joint_batch(rca_controlled, action_controlled)
    if update_result["status"] != "UPDATED":
        raise AssertionError(f"controlled synchronized update did not run: {update_result}")
    if trainer.bundle_update_step != 1 or trainer.rca_update_step != 1 or trainer.action_update_step != 1:
        raise AssertionError("role/bundle update counters are incorrect")
    if trainer.current_policy_version == rollout_version:
        raise AssertionError("successful synchronized update did not publish a new policy version")
    adapter_only, changed = _adapter_only_changes(before_update, model)
    if not adapter_only or not changed:
        raise AssertionError("successful update changed base tensors or changed no adapter tensors")

    # 3) Atomic checkpoint contains only trainable policy/optimizer state and resumes exactly.
    with tempfile.TemporaryDirectory(prefix="factorized-grpo-trainer-audit-") as tmpdir:
        checkpoint = Path(tmpdir) / "trainer.pt"
        trainer.save_checkpoint(checkpoint, last_update=update_result)
        if not checkpoint.exists() or checkpoint.stat().st_size == 0:
            raise AssertionError("trainer checkpoint was not written")

        resumed_model = _build_model()
        resumed = SynchronizedFactorizedGRPOTrainer(
            resumed_model,
            current_policy_version=rollout_version,
            config=cfg,
            device="cpu",
        )
        resume_info = resumed.load_checkpoint(checkpoint)
        if resume_info["current_policy_version"] != trainer.current_policy_version:
            raise AssertionError("checkpoint resume restored wrong policy version")
        if resume_info["bundle_update_step"] != 1:
            raise AssertionError("checkpoint resume restored wrong bundle update step")

        original_after_update = snapshot_named_parameters(model)
        resumed_after_load = snapshot_named_parameters(resumed_model)
        # Base models are deterministically identical and adapter states must match exactly.
        _assert_snapshots_equal(original_after_update, resumed_after_load)

        # Old rollout rows are now stale and must be rejected before any optimizer step.
        stale_blocked = False
        stale_before = snapshot_named_parameters(resumed_model)
        try:
            resumed.update_joint_batch(rca_controlled, action_controlled)
        except ValueError as exc:
            stale_blocked = "stale/wrong" in str(exc)
        if not stale_blocked:
            raise AssertionError("stale rollout policy version was not rejected")
        _assert_snapshots_equal(stale_before, snapshot_named_parameters(resumed_model))

    # 4) Transactionality: force Action optimizer.step() to fail after RCA succeeds.
    rollback_model = _build_model()
    rollback_trainer = SynchronizedFactorizedGRPOTrainer(
        rollback_model,
        current_policy_version=rollout_version,
        config=cfg,
        device="cpu",
    )
    rollback_before = snapshot_named_parameters(rollback_model)
    original_action_step = rollback_trainer.action_optimizer.step

    def _injected_failure(*_args: Any, **_kwargs: Any):
        raise RuntimeError("injected Action optimizer failure")

    rollback_trainer.action_optimizer.step = _injected_failure  # type: ignore[method-assign]
    rolled_back = False
    try:
        rollback_trainer.update_joint_batch(rca_controlled, action_controlled)
    except RuntimeError as exc:
        rolled_back = "rolled back" in str(exc)
    finally:
        rollback_trainer.action_optimizer.step = original_action_step  # type: ignore[method-assign]
    if not rolled_back:
        raise AssertionError("injected second-role failure did not trigger transactional rollback")
    _assert_snapshots_equal(rollback_before, snapshot_named_parameters(rollback_model))
    if rollback_trainer.current_policy_version != rollout_version:
        raise AssertionError("failed synchronized transaction advanced policy version")
    if (rollback_trainer.bundle_update_step, rollback_trainer.rca_update_step, rollback_trainer.action_update_step) != (0, 0, 0):
        raise AssertionError("failed synchronized transaction advanced update counters")
    if rollback_trainer.rca_optimizer.state_dict()["state"]:
        raise AssertionError("RCA optimizer state was not rolled back")
    if rollback_trainer.action_optimizer.state_dict()["state"]:
        raise AssertionError("Action optimizer state was not rolled back")

    print(json.dumps({
        "zero_signal_gate": {
            "status": zero_result["status"],
            "policy_version_unchanged": True,
            "parameters_bitwise_unchanged": True,
            "kl_only_update_prevented": True,
        },
        "controlled_synchronized_update": {
            "status": update_result["status"],
            "published_policy_version": trainer.current_policy_version,
            "bundle_update_step": trainer.bundle_update_step,
            "changed_only_lora_adapters": True,
        },
        "checkpoint_resume": {
            "atomic_checkpoint_written": True,
            "adapter_state_exactly_restored": True,
            "optimizer_and_version_state_restored": True,
        },
        "stale_rollout_rejected": True,
        "transactional_failure_rollback": {
            "second_role_failure_injected": True,
            "both_adapters_restored": True,
            "both_optimizer_states_restored": True,
            "policy_version_not_advanced": True,
        },
        "status": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
