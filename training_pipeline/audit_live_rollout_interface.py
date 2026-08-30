from __future__ import annotations

"""Fast contract audit for the live Action handoff (no Kubernetes required)."""

from typing import Any

from .action_loop import run_action_prompt_optimizer_loop
from .action_prompt_policy import StructuredActionPromptPolicy
from .fixed_action_agent import FixedActionAgent
from .schemas import FaultLabel


class RecordingLiveVerifier:
    is_live = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def action_namespace(self) -> str:
        return "aiops-twin-opaque123456"

    def apply_commands_and_score(
        self,
        full_state: dict[str, Any],
        rca_faults: list[FaultLabel],
        mitigation_action: dict[str, Any],
        commands: list[str],
        compressed_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del full_state, compressed_state
        self.calls.append({
            "faults": [fault.injection_key() for fault in rca_faults],
            "action": mitigation_action,
            "commands": list(commands),
        })
        return {
            "resolved": True,
            "twin_resolved": True,
            "sla_restored": True,
            "target_sla_restored": True,
            "symptom_reduction": 1.0,
            "global_symptom_reduction": 1.0,
            "target_symptom_reduction": 1.0,
            "action_repairs_fault_type": True,
        }


def main() -> None:
    verifier = RecordingLiveVerifier()
    fault = FaultLabel(
        service="user-timeline-service",
        fault_family="infra_failure",
        fault_type="infra_failure",
        fault_mechanism="assign_to_non_existent_node",
        variant_name="default",
    )
    gate = {
        "mode": "sparse_live_kubernetes_v1",
        "reproduction_score": 0.6,
        "predicted_fault_injection_checked": True,
        "rca_twin_verified": True,
    }
    result = run_action_prompt_optimizer_loop(
        {"scenario_id": "must-not-be-a-namespace", "fault_context": {}},
        {"scenario_id": "must-not-be-a-namespace", "services": ["user-timeline-service"]},
        {"final_prediction": "user-timeline-service::infra_failure::assign_to_non_existent_node"},
        [fault],
        StructuredActionPromptPolicy(),
        FixedActionAgent(),
        verifier,
        max_iterations=1,
        require_rca_twin_verification=True,
        skip_action_if_rca_unverified=True,
        min_twin_reproduction_score=0.1,
        rca_twin_gate=gate,
        group_size=1,
        require_upstream_label_success_for_gate=False,
    )
    assert result["success"] is True
    assert len(verifier.calls) == 1
    commands = verifier.calls[0]["commands"]
    assert commands, "the live verifier must receive the executable commands"
    assert all("-n aiops-twin-opaque123456" in command for command in commands)
    assert all("must-not-be-a-namespace" not in command for command in commands)
    assert any("/spec/template/spec/nodeSelector" in command for command in commands)
    print("PASS_LIVE_ROLLOUT_INTERFACE")


if __name__ == "__main__":
    main()
