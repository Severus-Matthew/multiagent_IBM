import json

from twin_utils import read_jsonl, write_json
from twin_runtime_config import (
    STATES_JSONL,
    SIMULATED_STATE_JSON,
    RCA_VALIDATION_JSON,
    MITIGATION_RESULT_JSON,
)
from digital_twin_runtime import MinimalDigitalTwin
from rca_twin_validator import aggregate_similarity, explain_match, verdict_from_similarity


def main():
    states = read_jsonl(STATES_JSONL)
    if not states:
        raise RuntimeError("No states found in states.jsonl")

    # choose one real state as reference
    real_state = states[0]

    # ---------------------------
    # RCA hypothesis to test
    # ---------------------------
    rca_hypothesis = {
        "service": "search",
        "fault_type": "dependency_failure",
    }

    # ---------------------------
    # Mitigation to test
    # ---------------------------
    mitigation = {
        "action": "restart_service",
        "service": "search",
    }

    twin = MinimalDigitalTwin()

    # 1) simulate RCA fault only
    simulated_state, simulated_sla = twin.simulate(
        base_state=real_state,
        rca_fault=rca_hypothesis,
        mitigation_action=None,
    )

    # 2) compare simulated abstraction against real abstraction
    sim = aggregate_similarity(real_state, simulated_state)
    reasons = explain_match(real_state, simulated_state)
    verdict = verdict_from_similarity(sim["overall_similarity"])

    validation_result = {
        "rca_hypothesis": rca_hypothesis,
        "similarity": sim,
        "verdict": verdict,
        "reasons": reasons,
        "simulated_sla_without_mitigation": simulated_sla,
    }

    # 3) apply mitigation in the twin
    mitigated_state, mitigated_sla = twin.simulate(
        base_state=real_state,
        rca_fault=rca_hypothesis,
        mitigation_action=mitigation,
    )

    mitigation_result = {
        "rca_hypothesis": rca_hypothesis,
        "mitigation": mitigation,
        "sla_after_mitigation": mitigated_sla,
    }

    write_json(simulated_state, SIMULATED_STATE_JSON)
    write_json(validation_result, RCA_VALIDATION_JSON)
    write_json(mitigation_result, MITIGATION_RESULT_JSON)

    print("Wrote simulated state to:", SIMULATED_STATE_JSON)
    print("Wrote RCA validation to:", RCA_VALIDATION_JSON)
    print("Wrote mitigation result to:", MITIGATION_RESULT_JSON)
    print("RCA verdict:", verdict)
    print("Similarity:", sim)


if __name__ == "__main__":
    main()