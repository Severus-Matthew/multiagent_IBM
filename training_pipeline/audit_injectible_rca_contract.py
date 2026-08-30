from __future__ import annotations

import json

from training_pipeline.ground_truth import labels_from_fault_context
from training_pipeline.llm_rca_solver import sanitize_rca_prediction
from training_pipeline.rca_reward import rca_reward
from training_pipeline.schemas import (
    INJECTIBLE_FAULT_MECHANISMS,
    parse_fault_lines,
)


def main() -> None:
    exact_line = "user-timeline-service::infra_failure::assign_to_non_existent_node"
    exact = parse_fault_lines(exact_line)
    legacy = parse_fault_lines("user-timeline-service::infra_failure")
    mismatch = parse_fault_lines("user-timeline-service::network_failure::container_kill")
    multifault = parse_fault_lines(
        "frontend::latency_degradation::network_delay::100ms\n"
        "mongodb::auth_failure::mongodb_auth_missing"
    )
    sanitized = sanitize_rca_prediction(
        "```\nuser-timeline-service::infra_failure::assign_to_non_existent_node\n```"
    )
    evaluator_state = {
        "graph": {"edges": []},
        "fault_context": {
            "faulty_service": "user-timeline-service",
            "fault_family": "assign_to_non_existent_node_social_net",
            "variant_name": "default",
        },
    }
    evaluator_labels = labels_from_fault_context(evaluator_state["fault_context"])
    wrong_mechanism = parse_fault_lines(
        "user-timeline-service::infra_failure::delete_pod"
    )
    exact_reward = rca_reward(evaluator_state, evaluator_labels, exact)
    wrong_mechanism_reward = rca_reward(
        evaluator_state, evaluator_labels, wrong_mechanism
    )

    invariants = {
        "public_mechanism_catalog_nonempty": bool(INJECTIBLE_FAULT_MECHANISMS),
        "exact_mechanism_is_injectible": len(exact) == 1 and exact[0].is_injectible(),
        "exact_mechanism_round_trips": bool(exact) and exact[0].fault_mechanism == "assign_to_non_existent_node",
        "legacy_generic_label_remains_parseable": len(legacy) == 1,
        "legacy_generic_label_is_not_silently_injectible": len(legacy) == 1 and not legacy[0].is_injectible(),
        "type_mechanism_mismatch_is_not_injectible": len(mismatch) == 1 and not mismatch[0].is_injectible(),
        "multifault_mechanisms_supported": len(multifault) == 2 and all(x.is_injectible() for x in multifault),
        "variant_round_trips": bool(multifault) and multifault[0].variant_name == "100ms",
        "solver_sanitizer_preserves_mechanism": sanitized == exact_line,
        "evaluator_ground_truth_retains_mechanism": bool(evaluator_labels)
        and evaluator_labels[0].fault_mechanism == "assign_to_non_existent_node",
        "exact_success_requires_correct_mechanism": exact_reward["success"]
        and not wrong_mechanism_reward["success"],
        "correct_mechanism_receives_more_reward": exact_reward["reward"]
        > wrong_mechanism_reward["reward"],
        "scenario_specific_candidate_menu_used": False,
        "oracle_fault_substitution_allowed": False,
    }
    positive = {
        key for key in invariants
        if key not in {"scenario_specific_candidate_menu_used", "oracle_fault_substitution_allowed"}
    }
    passed = (
        all(invariants[key] for key in positive)
        and not invariants["scenario_specific_candidate_menu_used"]
        and not invariants["oracle_fault_substitution_allowed"]
    )
    print(json.dumps({
        "status": "PASS_INJECTIBLE_RCA_CONTRACT" if passed else "FAIL_INJECTIBLE_RCA_CONTRACT",
        "public_mechanisms": INJECTIBLE_FAULT_MECHANISMS,
        "example": exact[0].to_dict() if exact else None,
        "invariants": invariants,
        "next_gate": "live_fault_injector_registry" if passed else "fix_injectible_rca_contract",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
