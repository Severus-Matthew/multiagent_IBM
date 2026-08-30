from __future__ import annotations

"""CPU end-to-end audit for real HF/PEFT rollout records.

This is the final rollout-side gate before wiring the large Qwen backend.  It runs
real Hugging Face ``generate()`` calls through the canonical

    RCA policy -> RCA solver -> twin -> Action policy -> Action agent -> verifier

trajectory path on one or more real processed scenarios, while the trainable RCA
and Action prompt policies share one tiny local causal LM with separate LoRA
adapters.  The test then requires the emitted factorized buffers to pass the
strict exact-token GRPO validator and replays every row through the audited
learner at importance ratio one.

No optimizer step is performed here.  Optimizer/gradient correctness is already
covered by ``audit_factorized_grpo_learner`` and ``audit_hf_peft_two_adapter``;
this audit specifically proves that the production rollout path emits the exact
fields those learners require.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from digital_twin_runtime.twin_verifier import BehavioralTwinVerifier

from .audit_hf_exact_token_sampler import TinyTokenizer, _build_model
from .data_loader import iter_scenarios
from .end_to_end_loop import run_end_to_end_trajectory_group
from .factorized_grpo_learner import FactorizedGRPOConfig, model_decision_loss
from .fixed_action_agent import FixedActionAgent
from .ground_truth import labels_from_full_state
from .grpo_dataset import load_grpo_dataset, summarize_dataset
from .hf_exact_token_sampler import ExactTokenGenerationConfig, HFExactTokenPolicySampler
from .rca_loop import HeuristicRCASolver
from .split_utils import read_scenario_ids
from .trainable_hf_prompt_policies import (
    TrainableHFActionPromptPolicy,
    TrainableHFRCAInstructionPolicy,
)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _stamp(row: dict[str, Any], *, adapter_id: str, optimizer_role: str) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata", {}) or {})
    metadata["adapter_id"] = adapter_id
    metadata["optimizer_role"] = optimizer_role
    metadata["sync_batch_id"] = "cpu-exact-token-audit"
    out["metadata"] = metadata
    out["adapter_id"] = adapter_id
    out["sync_batch_id"] = "cpu-exact-token-audit"
    return out


def _replay_all_at_ratio_one(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    cfg = FactorizedGRPOConfig(kl_coeff=0.01)
    for row in rows:
        adapter = str(row.get("adapter_id") or "")
        if adapter not in {"lora_rca", "lora_action"}:
            raise AssertionError(f"unexpected adapter_id in exact-token buffer: {adapter!r}")
        model.set_adapter(adapter)
        model.eval()
        d = model_decision_loss(model, row, config=cfg, device="cpu")
        ratio = float(d.ratio_mean.detach().cpu())
        if abs(ratio - 1.0) > 2e-6:
            raise AssertionError(
                f"{row.get('sample_id')}: unchanged rollout replay ratio must be 1; got {ratio}"
            )
        ratios.append(ratio)
    if not ratios:
        raise AssertionError("exact-token rollout audit produced no optimizer rows")
    return {
        "num_rows": len(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "ratio_mean": sum(ratios) / len(ratios),
        "all_ratio_one": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the canonical joint pipeline with tiny real HF/PEFT prompt policies and strict exact-token validation."
    )
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--trajectory_group_size", type=int, default=4)
    ap.add_argument("--rca_max_iterations", type=int, default=2)
    ap.add_argument("--action_max_iterations", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    args = ap.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.trajectory_group_size < 2:
        raise ValueError("--trajectory_group_size must be >= 2")
    if args.rca_max_iterations < 1 or args.action_max_iterations < 1:
        raise ValueError("iteration counts must be >= 1")

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rca_path = out_dir / "rca_policy_samples.jsonl"
    action_path = out_dir / "action_policy_samples.jsonl"
    trajectory_path = out_dir / "joint_trajectories.jsonl"
    summary_path = out_dir / "summary.json"
    for path in (rca_path, action_path, trajectory_path):
        if path.exists():
            path.unlink()

    model = _build_model()
    tokenizer = TinyTokenizer()
    sampler = HFExactTokenPolicySampler(
        model,
        tokenizer,
        config=ExactTokenGenerationConfig(
            max_new_tokens=int(args.max_new_tokens),
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
            max_prompt_tokens=32,
            seed=1907,
        ),
        device="cpu",
    )
    rca_policy = TrainableHFRCAInstructionPolicy(
        sampler,
        adapter_name="lora_rca",
        max_iterations=args.rca_max_iterations,
    )
    action_policy = TrainableHFActionPromptPolicy(sampler, adapter_name="lora_action")
    rca_solver = HeuristicRCASolver()
    action_agent = FixedActionAgent(max_commands=15)
    twin = BehavioralTwinVerifier()

    allowed_ids = read_scenario_ids(args.scenario_ids)
    scenario_count = 0
    trajectory_count = 0
    rca_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    unsafe_agent_inputs = 0

    for rec in iter_scenarios(args.processed_states):
        if allowed_ids is not None and rec.scenario_id not in allowed_ids:
            continue
        if not labels_from_full_state(rec.full_state):
            continue
        if scenario_count >= args.limit:
            break

        result = run_end_to_end_trajectory_group(
            rec.full_state,
            rec.compressed_state,
            rca_instruction_policy=rca_policy,
            rca_solver=rca_solver,
            action_prompt_policy=action_policy,
            action_agent=action_agent,
            twin_verifier=twin,
            trajectory_group_size=args.trajectory_group_size,
            rca_max_iterations=args.rca_max_iterations,
            action_max_iterations=args.action_max_iterations,
            rca_policy_model_name="tiny-hf-shared-base+lora_rca",
            action_policy_model_name="tiny-hf-shared-base+lora_action",
            policy_version="cpu-exact-token-audit-v1",
            agent_input_mode="training_safe",
            reward_mode="factorized_joint_pipeline_v2_no_double_count",
            min_twin_reproduction_score=0.0,
            rca_downstream_credit_weight=0.15,
            action_system_credit_weight=0.25,
        )

        scenario_count += 1
        trajectories = list(result.get("trajectories", []) or [])
        trajectory_count += len(trajectories)
        unsafe_agent_inputs += int(
            not bool((result.get("agent_input_safety") or {}).get("safe_for_training_agent"))
        )
        _append_jsonl(
            trajectory_path,
            {
                "scenario_id": result.get("scenario_id"),
                "trajectory_group_id": result.get("trajectory_group_id"),
                "trajectories": trajectories,
                "rca_group_zero_variance": result.get("rca_group_zero_variance"),
                "action_group_zero_variance": result.get("action_group_zero_variance"),
                "agent_input_safety": result.get("agent_input_safety"),
            },
        )

        for raw in result.get("rca_grpo_samples", []) or []:
            row = _stamp(raw, adapter_id="lora_rca", optimizer_role="rca_policy")
            rca_rows.append(row)
            _append_jsonl(rca_path, row)
        for raw in result.get("action_grpo_samples", []) or []:
            row = _stamp(raw, adapter_id="lora_action", optimizer_role="action_policy")
            action_rows.append(row)
            _append_jsonl(action_path, row)

        print(
            f"[HF-E2E-AUDIT] scenario={scenario_count} id={rec.scenario_id} "
            f"trajectories={len(trajectories)} rca_rows={len(result.get('rca_grpo_samples', []) or [])} "
            f"action_rows={len(result.get('action_grpo_samples', []) or [])}"
        )

    if scenario_count == 0:
        raise RuntimeError("no labeled scenarios matched the requested audit inputs")
    if unsafe_agent_inputs:
        raise AssertionError(f"unsafe agent-facing inputs detected: {unsafe_agent_inputs}")
    if not rca_rows or not action_rows:
        raise AssertionError("canonical end-to-end audit must emit both RCA and Action optimizer rows")

    # This is the gate the real Qwen learner will require.  It validates exact
    # prompt/completion token IDs, old-logprob alignment/sum, reference logprobs,
    # policy version/model identity, factorized advantages, and 1/D trajectory
    # decision weighting.
    strict_rca = load_grpo_dataset(
        rca_path,
        require_policy_credit=True,
        require_old_logprobs=True,
    )
    strict_action = load_grpo_dataset(
        action_path,
        require_policy_credit=True,
        require_old_logprobs=True,
    )

    rca_replay = _replay_all_at_ratio_one(model, strict_rca)
    action_replay = _replay_all_at_ratio_one(model, strict_action)

    # Confirm the role buffers cannot be accidentally crossed.
    if any(row.get("adapter_id") != "lora_rca" for row in strict_rca):
        raise AssertionError("RCA exact-token buffer contains a non-RCA adapter row")
    if any(row.get("adapter_id") != "lora_action" for row in strict_action):
        raise AssertionError("Action exact-token buffer contains a non-Action adapter row")

    summary = {
        "scenario_count": scenario_count,
        "trajectory_count": trajectory_count,
        "unsafe_agent_inputs": unsafe_agent_inputs,
        "rca": summarize_dataset(strict_rca),
        "action": summarize_dataset(strict_action),
        "rca_exact_replay": rca_replay,
        "action_exact_replay": action_replay,
        "strict_policy_credit_validation": True,
        "strict_old_logprob_validation": True,
        "canonical_joint_rollout_path": True,
        "shared_base_two_role_adapters": True,
        "uses_real_training_update": False,
        "status": "PASS",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
