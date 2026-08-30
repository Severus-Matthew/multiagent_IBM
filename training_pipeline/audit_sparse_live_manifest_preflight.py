from __future__ import annotations

import argparse
import json
from pathlib import Path

from digital_twin_runtime.sparse_live_manifest import discover_sparse_manifest_plan
from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
from training_pipeline.audit_sparse_live_twin_spec import (
    _augment_with_static_topology,
    _default_application_source_root,
    _find_scenario,
    _parse_fault,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Read-only preflight for cloning only the fault-conditioned sparse "
            "service set into a dedicated live-Twin namespace."
        )
    )
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", required=True)
    ap.add_argument("--source_namespace", required=True)
    ap.add_argument(
        "--fault",
        action="append",
        required=True,
        help="Predicted fault as SERVICE::FAULT_TYPE; repeat for multifault.",
    )
    ap.add_argument("--application_source_root", default=None)
    ap.add_argument("--upstream_hops", type=int, default=2)
    ap.add_argument("--downstream_support_hops", type=int, default=2)
    ap.add_argument("--symptom_hops", type=int, default=2)
    ap.add_argument("--max_entry_path_hops", type=int, default=8)
    args = ap.parse_args()

    rec = _find_scenario(args.processed_states, args.scenario_id)
    faults = [_parse_fault(x) for x in args.fault]

    app_root = (
        Path(args.application_source_root).expanduser().resolve()
        if args.application_source_root
        else _default_application_source_root(args.processed_states)
    )
    planner_state, static_topology = _augment_with_static_topology(
        rec.compressed_state, app_root
    )
    spec = build_sparse_live_twin_spec(
        planner_state,
        faults,
        upstream_hops=args.upstream_hops,
        downstream_support_hops=args.downstream_support_hops,
        symptom_hops=args.symptom_hops,
        max_entry_path_hops=args.max_entry_path_hops,
    )

    rs = spec.resource_summary or {}
    plan = discover_sparse_manifest_plan(
        args.source_namespace,
        spec.services_to_keep,
    )

    selected = set(spec.services_to_keep)
    service_object_names = {
        str(x.get("name")) for x in plan.service_objects if x.get("name")
    }
    missing_service_objects = sorted(selected - service_object_names)

    roots = {f.service for f in faults}
    root_service_objects_present = roots <= service_object_names
    entrypoints = set(spec.entrypoint_services)
    selected_entrypoints = entrypoints & selected
    entrypoint_service_objects_present = selected_entrypoints <= service_object_names

    sparse_spec_valid = bool(selected) and not rs.get("invalid_predicted_root", False) and not rs.get("invalid_topology", False)
    no_full_app_fallback = int(rs.get("kept_services", 0) or 0) < int(rs.get("total_application_services", 0) or 0)
    controllers_complete = not plan.missing_selected_controllers and len(plan.controllers) == len(selected)
    required_refs_resolved = not plan.missing_required_refs
    service_dns_objects_complete = not missing_service_objects
    storage_clone_policy_required = bool(plan.persistent_volume_claims)

    invariants = {
        "cluster_mutated": False,
        "read_only_manifest_discovery": bool(plan.read_only),
        "agent_state_is_redacted": True,
        "oracle_labels_used": False,
        "fault_context_used": False,
        "static_topology_is_fault_independent": True,
        "sparse_spec_valid": sparse_spec_valid,
        "full_application_fallback_used": not no_full_app_fallback,
        "all_selected_controllers_resolved": controllers_complete,
        "all_required_kubernetes_refs_exist": required_refs_resolved,
        "all_selected_service_dns_objects_resolved": service_dns_objects_complete,
        "predicted_root_service_objects_present": root_service_objects_present,
        "selected_entrypoint_service_objects_present": entrypoint_service_objects_present,
        "source_namespace_is_only_manifest_source": True,
    }

    valid = all(
        invariants[k]
        for k in (
            "read_only_manifest_discovery",
            "sparse_spec_valid",
            "all_selected_controllers_resolved",
            "all_required_kubernetes_refs_exist",
            "all_selected_service_dns_objects_resolved",
            "predicted_root_service_objects_present",
            "selected_entrypoint_service_objects_present",
        )
    ) and no_full_app_fallback

    report = {
        "status": (
            "PASS_SPARSE_LIVE_MANIFEST_PREFLIGHT"
            if valid
            else "FAIL_SPARSE_LIVE_MANIFEST_PREFLIGHT"
        ),
        "scenario_id": rec.scenario_id,
        "source_namespace": args.source_namespace,
        "predicted_faults": [f.to_dict() for f in faults],
        "sparse_twin_spec": spec.to_dict(),
        "manifest_plan": plan.to_dict(),
        "manifest_diagnostics": {
            "missing_service_objects": missing_service_objects,
            "storage_clone_policy_required": storage_clone_policy_required,
            "static_topology_edges_discovered": int(
                static_topology.get("static_topology_edges_discovered", 0) or 0
            ),
            "application_source_root": str(app_root),
            "note": (
                "Source readiness is diagnostic only. The next stage renders "
                "sanitized copies into a dedicated Twin namespace; it does not "
                "reuse source pod runtime state."
            ),
        },
        "invariants": invariants,
        "next_gate": (
            "sparse_live_namespace_manifest_render"
            if valid and not storage_clone_policy_required
            else "sparse_live_storage_policy_then_manifest_render"
            if valid
            else "fix_sparse_live_manifest_preflight"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
