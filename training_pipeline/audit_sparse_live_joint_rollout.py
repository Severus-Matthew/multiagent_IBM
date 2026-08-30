from __future__ import annotations

"""Run the minimum real joint RCA -> Action sparse-Twin rollout group."""

import argparse
import json
from pathlib import Path
from typing import Any

from digital_twin_runtime.sparse_live_verifier import (
    SparseLiveTwinVerifier,
    SparseLiveVerifierConfig,
)

from .action_prompt_policy import StructuredActionPromptPolicy
from .data_loader import read_json
from .end_to_end_loop import run_end_to_end_trajectory_group
from .fixed_action_agent import FixedActionAgent
from .rca_loop import HeuristicRCAInstructionPolicy


class AuditedInjectibleRCASolver:
    """Deterministic solver used to audit plumbing, not model quality."""

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        del compressed_state, instruction
        return (
            "user-timeline-service::infra_failure::"
            "assign_to_non_existent_node"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_namespace", default="test-social-network")
    ap.add_argument(
        "--scenario_dir",
        default=(
            "AIOpsLab/processed_states/"
            "gen_assign_to_non_existent_node_social_net-analysis-"
            "user-timeline-service-default"
        ),
    )
    ap.add_argument(
        "--application_source_root",
        default="AIOpsLab/aiopslab-applications/socialNetwork",
    )
    ap.add_argument("--state_abstraction_root", default="state_abstraction_full")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    full_state = read_json(scenario_dir / "state_abstraction.json", {})
    compressed_state = read_json(
        scenario_dir / "state_abstraction_compressed.json", {}
    )
    if not full_state or not compressed_state:
        raise FileNotFoundError(f"missing processed state in {scenario_dir}")

    verifier = SparseLiveTwinVerifier(
        SparseLiveVerifierConfig(
            source_namespace=args.source_namespace,
            application_source_root=str(
                Path(args.application_source_root).expanduser().resolve()
            ),
            state_abstraction_root=str(
                Path(args.state_abstraction_root).expanduser().resolve()
            ),
            reproduction_threshold=0.1,
        )
    )
    result = run_end_to_end_trajectory_group(
        full_state,
        compressed_state,
        rca_instruction_policy=HeuristicRCAInstructionPolicy(),
        rca_solver=AuditedInjectibleRCASolver(),
        action_prompt_policy=StructuredActionPromptPolicy(),
        action_agent=FixedActionAgent(),
        twin_verifier=verifier,
        trajectory_group_size=2,
        rca_max_iterations=1,
        action_max_iterations=1,
        rca_policy_model_name="deterministic-live-interface-audit",
        action_policy_model_name="structured-live-interface-audit",
        policy_version="sparse-live-joint-v1",
        min_twin_reproduction_score=0.1,
    )
    trajectories = result.get("trajectories", []) or []
    summary = {
        "status": "PENDING_SPARSE_LIVE_JOINT_ROLLOUT",
        "trajectory_group_size": len(trajectories),
        "all_trajectories_succeeded": bool(trajectories)
        and all(row.get("trajectory_success") for row in trajectories),
        "system_rewards": [row.get("system_reward") for row in trajectories],
        "rca_policy_returns": [row.get("rca_policy_return") for row in trajectories],
        "action_policy_returns": [row.get("action_policy_return") for row in trajectories],
        "rca_advantages": [row.get("rca_policy_advantage") for row in trajectories],
        "action_advantages": [row.get("action_policy_advantage") for row in trajectories],
        "rca_verified": [
            bool((row.get("action_result", {}) or {}).get("rca_twin_gate", {}).get("rca_twin_verified"))
            for row in trajectories
        ],
        "action_succeeded": [
            bool((row.get("action_result", {}) or {}).get("success"))
            for row in trajectories
        ],
        "namespace_cleaned_after_group": verifier.action_namespace() is None,
        "uses_real_training_update": False,
    }
    action_successes = list(summary["action_succeeded"])
    rca_advantages = [float(x or 0.0) for x in summary["rca_advantages"]]
    action_advantages = [float(x or 0.0) for x in summary["action_advantages"]]
    summary["exploration_reward_contrast"] = bool(
        action_successes == [True, False]
        and rca_advantages[0] > 0.0 > rca_advantages[1]
        and action_advantages[0] > 0.0 > action_advantages[1]
    )
    passed = bool(
        len(trajectories) == 2
        and all(summary["rca_verified"])
        and summary["exploration_reward_contrast"]
        and summary["namespace_cleaned_after_group"]
    )
    summary["status"] = (
        "PASS_SPARSE_LIVE_JOINT_ROLLOUT"
        if passed
        else "FAIL_SPARSE_LIVE_JOINT_ROLLOUT"
    )
    if args.output:
        Path(args.output).expanduser().write_text(
            json.dumps({"summary": summary, "result": result}, indent=2, default=str)
        )
    if len(trajectories) != 2:
        raise AssertionError(summary)
    if not all(summary["rca_verified"]):
        raise AssertionError(summary)
    if not summary["exploration_reward_contrast"]:
        raise AssertionError(summary)
    if not summary["namespace_cleaned_after_group"]:
        raise AssertionError(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
