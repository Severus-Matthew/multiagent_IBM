from __future__ import annotations

"""Fail-closed audit of channel-aware Twin comparison and evidence-based admission.

Covers three contracts introduced to make trace-less and mislabeled captures
usable without weakening the verifier:

1. ``compare_symptoms_scoped`` scores the trace channel only when the original
   capture observed it; a Twin that reproduces a fault faithfully is not
   penalized for request edges the incident never recorded, and a traced
   original is scored exactly as before.
2. ``assess_record_for_live_reward`` admits captures whose labeled mechanism is
   structural and whose target Deployment state is recorded, rejects captures
   with no target evidence in any scored channel, and admits weak evidence only
   on explicit request.
3. Label corrections drop only components whose upstream injector provably did
   not mutate the labeled service, never touch the agent-visible state, and
   turn an unfaithful multi-fault record into a source-faithful record.

Also exercises the image-embedded configuration discovery against a synthetic
bundle plus the application source tree, without any cluster access.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from digital_twin_runtime.fault_mutation_discovery import (
    MutationDiscoveryError,
    config_file_overlay_objects,
    discover_config_corruption,
)
from digital_twin_runtime.telemetry_comparator import compare_symptoms_scoped, observed_channels
from training_pipeline.label_corrections import apply_label_correction, correction_for_record
from training_pipeline.live_dataset_admission import (
    assess_record_for_live_reward,
    comparable_symptom_evidence,
)
from training_pipeline.schemas import FaultLabel


def _assert(condition: bool, name: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def _state(*, traced: bool, unavailable: int, desired: int = 1, logs: bool = False) -> dict[str, Any]:
    state: dict[str, Any] = {
        "services": ["front", "target", "dep"],
        "system": {
            "target": {
                "deployment": {"replicas_desired": desired, "replicas_ready": desired - unavailable,
                               "replicas_unavailable": unavailable},
                "health": {"status": "no_ready_endpoints" if unavailable else "healthy",
                           "pods_unready": unavailable},
            },
            "front": {"deployment": {"replicas_desired": 1, "replicas_ready": 1}, "health": {"status": "healthy"}},
        },
        "traces": (
            {"per_edge": {"front->target": {"error_ratio": 1.0, "source": "front", "target": "target"}},
             "summary": {"num_edges": 1}}
            if traced else {"per_edge": {}, "summary": {"num_edges": 0}}
        ),
        "logs": {"target": {"signal": {"error_count": 3}}} if logs else {},
        "redaction": {"safe_for_rca_agent": True},
    }
    return state


def _record(full_state: dict[str, Any], compressed: dict[str, Any], sid: str = "synthetic") -> Any:
    return SimpleNamespace(scenario_id=sid, full_state=full_state, compressed_state=compressed)


def _full(labels: list[tuple[str, str]]) -> dict[str, Any]:
    instances = [
        {"index": i, "faulty_service": svc, "fault_family": fam, "variant_name": "default",
         "variant_params": {}, "task": "detection", "app": "hotel"}
        for i, (svc, fam) in enumerate(labels)
    ]
    return {"fault_context": {
        "fault_family": "multifault" if len(labels) > 1 else labels[0][1],
        "faulty_service": labels[0][0], "is_multifault": len(labels) > 1,
        "fault_instances": instances, "primary_fault": instances[0],
        "expected_faulty_services": [svc for svc, _ in labels],
    }}


def run_audit() -> dict[str, Any]:
    checks: list[str] = []
    scope = ["front", "target", "dep"]

    # 1. channel-aware comparison
    untraced = _state(traced=False, unavailable=1)
    traced = _state(traced=True, unavailable=1)
    twin = _state(traced=True, unavailable=1)
    _assert(not observed_channels(untraced)["traces"], "header_only_trace_export_is_unobserved", checks)
    _assert(observed_channels(traced)["traces"], "collected_edges_are_observed", checks)
    a = compare_symptoms_scoped(untraced, twin, scope, target_services=["target"])
    b = compare_symptoms_scoped(traced, twin, scope, target_services=["target"])
    _assert(not a["trace_channel_scored"] and "traces" in a["channels_unobserved_in_original"],
            "unobserved_trace_channel_is_not_scored", checks)
    _assert(a["active_channel_weights"]["trace_edges"] == 0.0, "unobserved_trace_weight_is_zero", checks)
    _assert(a["reproduction_score"] == 1.0, "faithful_twin_not_penalized_for_unobserved_channel", checks)
    _assert(b["trace_channel_scored"] and b["reproduction_score"] == 1.0, "traced_original_scored_with_traces", checks)
    twin_missing_edge = _state(traced=False, unavailable=1)
    c = compare_symptoms_scoped(traced, twin_missing_edge, scope, target_services=["target"])
    _assert(c["trace_channel_scored"] and c["reproduction_score"] < 1.0,
            "traced_original_still_penalizes_twin_without_edges", checks)
    wrong_twin = _state(traced=True, unavailable=0)
    d = compare_symptoms_scoped(untraced, wrong_twin, scope, target_services=["target"])
    _assert(d["reproduction_score"] < a["reproduction_score"],
            "wrong_reproduction_scores_lower_without_trace_channel", checks)
    healthy = _state(traced=False, unavailable=0)
    e = compare_symptoms_scoped(healthy, twin, scope, target_services=["target"])
    _assert(e["score_reason"] != "no_original_symptoms_in_sparse_scope" or e["reproduction_score"] == 0.0,
            "fail_closed_semantics_preserved", checks)

    # 2. evidence-based admission
    structural_label = [FaultLabel("target", "infra_failure", "scale_pod_zero_social_net", "default", "scale_replicas_zero")]
    request_label = [FaultLabel("target", "network_failure", "network_loss_hotel_res", "default", "network_loss")]
    ev = comparable_symptom_evidence(_state(traced=False, unavailable=1), structural_label)
    _assert(ev["strength"] == "strong" and ev["per_label"][0]["basis"] == "structural_target_state",
            "structural_mechanism_without_traces_is_strong", checks)
    ev = comparable_symptom_evidence(_state(traced=False, unavailable=1, desired=2), structural_label)
    _assert(ev["strength"] == "strong", "scale_variant_count_is_structural_evidence", checks)
    ev = comparable_symptom_evidence(_state(traced=False, unavailable=0), request_label)
    _assert(ev["strength"] == "none", "request_level_mechanism_without_traces_or_logs_is_none", checks)
    ev = comparable_symptom_evidence(_state(traced=False, unavailable=0, logs=True), request_label)
    _assert(ev["strength"] == "weak", "log_errors_only_is_weak", checks)
    ev = comparable_symptom_evidence(_state(traced=True, unavailable=0), request_label)
    _assert(ev["strength"] == "strong", "collected_traces_are_strong_for_any_mechanism", checks)

    rec = _record(_full([("target", "network_loss_hotel_res")]), _state(traced=False, unavailable=0, logs=True))
    _assert(not assess_record_for_live_reward(rec)["eligible"], "weak_evidence_rejected_by_default", checks)
    _assert(assess_record_for_live_reward(rec, admit_weak_evidence=True)["eligible"],
            "weak_evidence_admitted_only_on_request", checks)
    rec = _record(_full([("target", "network_loss_hotel_res")]), _state(traced=False, unavailable=0))
    _assert("historical_capture_has_no_comparable_symptom_evidence" in assess_record_for_live_reward(rec, admit_weak_evidence=True)["reasons"],
            "no_evidence_rejected_even_with_weak_admission", checks)

    # 3. label corrections
    full = _full([("mongodb-recommendation", "revoke_auth_mongodb"), ("profile", "network_loss_hotel_res")])
    comp = _state(traced=True, unavailable=0)
    rec = _record(full, comp, "multi")
    before = assess_record_for_live_reward(rec)
    _assert("source_injection_not_faithful_to_label" in before["reasons"], "noop_component_makes_record_unfaithful", checks)
    entry = correction_for_record(rec)
    _assert(entry is not None and [r["service"] for r in entry["dropped"]] == ["mongodb-recommendation"]
            and [r["service"] for r in entry["retained"]] == ["profile"], "correction_drops_only_noop_component", checks)
    corrected = apply_label_correction(full, entry)
    _assert(full["fault_context"]["is_multifault"] is True, "correction_does_not_mutate_input", checks)
    _assert(corrected["fault_context"]["is_multifault"] is False
            and corrected["fault_context"]["faulty_service"] == "profile"
            and corrected["label_correction"]["reason"], "corrected_record_is_single_fault_with_provenance", checks)
    after = assess_record_for_live_reward(_record(corrected, comp, "multi"))
    _assert(after["source_injection"]["valid"] and after["eligible"], "corrected_record_is_source_faithful", checks)
    _assert(correction_for_record(_record(_full([("mongodb-recommendation", "revoke_auth_mongodb"), ("mongodb-profile", "user_unregistered_mongodb")]), comp)) is None,
            "all_noop_record_is_not_correctable", checks)
    _assert(correction_for_record(_record(_full([("geo", "misconfig_app_hotel_res"), ("profile", "network_loss_hotel_res")]), comp)) is None,
            "faithful_record_needs_no_correction", checks)
    _assert(json.dumps(comp) == json.dumps(rec.compressed_state), "agent_visible_state_untouched", checks)

    # 4. image-embedded configuration discovery (source tree only, no cluster)
    root = Path(__file__).resolve().parents[1] / "AIOpsLab" / "aiopslab-applications" / "hotelReservation"
    if root.is_dir():
        bundle = SimpleNamespace(objects=[
            {"kind": "Deployment", "metadata": {"name": "geo"},
             "spec": {"template": {"spec": {"containers": [{"name": "hotel-reserv-geo", "image": "x", "command": ["geo"]}]}}}},
            {"kind": "Deployment", "metadata": {"name": "search"},
             "spec": {"template": {"spec": {"containers": [{"name": "hotel-reserv-search", "image": "x", "command": ["search"]}]}}}},
            {"kind": "Service", "metadata": {"name": "geo"}},
            {"kind": "Service", "metadata": {"name": "mongodb-geo"}},
        ], object_refs=[])
        corruption = discover_config_corruption(bundle, "geo", application_source_root=root)
        _assert(corruption.target_kind == "ConfigFile" and corruption.key == "GeoMongoAddress"
                and corruption.target_name.endswith("/config.json"), "hotel_geo_config_file_endpoint_discovered", checks)
        _assert(corruption.faulted_value.endswith(":27017") and "invalid" in corruption.faulted_value,
                "corrupted_endpoint_keeps_port_and_is_unroutable", checks)
        try:
            discover_config_corruption(bundle, "search", application_source_root=root)
            raise AssertionError("search_should_fail_closed")
        except MutationDiscoveryError as exc:
            _assert(exc.reason == "no_service_scoped_endpoint_in_image_config_file", "search_without_config_endpoint_fails_closed", checks)
        try:
            discover_config_corruption(bundle, "geo")
            raise AssertionError("no_source_root_should_fail_closed")
        except MutationDiscoveryError as exc:
            _assert(exc.reason == "no_discoverable_configuration_endpoint_for_service", "without_source_root_behaviour_unchanged", checks)
        live = json.loads((root / "config.json").read_text())
        cm, dep = config_file_overlay_objects(bundle.objects[0], corruption, "aiops-twin-x", live)
        mount = dep["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]
        _assert(mount["subPath"] == "config.json" and mount["mountPath"] == corruption.target_name
                and json.loads(cm["data"]["config.json"])["GeoMongoAddress"] == corruption.faulted_value,
                "overlay_objects_mount_only_the_corrupted_file", checks)
        try:
            config_file_overlay_objects(bundle.objects[0], corruption, "aiops-twin-x", {**live, "GeoMongoAddress": "other:1"})
            raise AssertionError("mismatch_should_fail_closed")
        except MutationDiscoveryError as exc:
            _assert(exc.reason == "image_config_file_disagrees_with_source_tree", "live_file_mismatch_fails_closed", checks)
    return {"status": "PASS_COMPARABLE_EVIDENCE_ADMISSION_AUDIT", "checks": checks, "check_count": len(checks)}


def main() -> None:
    try:
        result = run_audit()
    except AssertionError as exc:
        print(json.dumps({"status": "FAIL_COMPARABLE_EVIDENCE_ADMISSION_AUDIT", "failed_check": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
