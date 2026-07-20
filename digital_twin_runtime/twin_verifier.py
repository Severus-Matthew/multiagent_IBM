from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from training_pipeline.schemas import FaultLabel
from .telemetry_comparator import compare_symptoms, score_resolution


def _import_behavioral_simulator():
    repo_root = Path(__file__).resolve().parents[1]
    state_abs = repo_root / "state_abstraction_full"
    if str(state_abs) not in sys.path:
        sys.path.insert(0, str(state_abs))
    from behavioral_simulator import AbstractBehavioralSimulator  # type: ignore
    return AbstractBehavioralSimulator


class BehavioralTwinVerifier:
    """Offline Stage-2 verifier using the existing abstract behavioral simulator."""
    def validate_rca_prediction(self, full_state: dict[str, Any], compressed_state: dict[str, Any],
                                predicted_faults: list[FaultLabel]) -> dict[str, Any]:
        Sim = _import_behavioral_simulator(); sim = Sim(full_state.get("graph", {}))
        twin_state = full_state
        for fault in predicted_faults:
            twin_state, _ = sim.simulate(twin_state, rca_fault={"service": fault.service, "fault_type": fault.fault_type})
        result = compare_symptoms(compressed_state, twin_state)
        result["mode"] = "behavioral_offline"
        return result

    def apply_action_and_score(self, full_state: dict[str, Any], rca_faults: list[FaultLabel],
                               mitigation_action: dict[str, Any]) -> dict[str, Any]:
        Sim = _import_behavioral_simulator(); sim = Sim(full_state.get("graph", {}))
        before = full_state
        for fault in rca_faults:
            before, _ = sim.simulate(before, rca_fault={"service": fault.service, "fault_type": fault.fault_type})
        after, after_sla = sim.simulate(before, mitigation_action=mitigation_action)
        score = score_resolution(before, after)
        score["after_sla"] = after_sla; score["mode"] = "behavioral_offline"
        return score
