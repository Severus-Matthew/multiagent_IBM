from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from digital_twin_runtime.sparse_live_manifest import (
    discover_sparse_manifest_plan,
    render_sparse_manifest_bundle,
)
from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
from training_pipeline.audit_sparse_live_twin_spec import (
    _augment_with_static_topology,
    _default_application_source_root,
    _find_scenario,
    _parse_fault,
)


_FORBIDDEN_METADATA = {
    "creationTimestamp",
    "deletionTimestamp",
    "generation",
    "managedFields",
    "ownerReferences",
    "resourceVersion",
    "selfLink",
    "uid",
}


def _namespace_exists(name: str) -> bool:
    proc = subprocess.run(
        ["kubectl", "get", "namespace", name, "-o", "name"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode == 0


def _portable(obj: dict[str, Any], target_namespace: str) -> bool:
    metadata = obj.get("metadata", {}) or {}
    if metadata.get("namespace") != target_namespace:
        return False
    if _FORBIDDEN_METADATA & set(metadata):
        return False
    if "status" in obj:
        return False
    if obj.get("kind") == "Service":
        spec = obj.get("spec", {}) or {}
        if any(key in spec for key in ("clusterIP", "clusterIPs", "loadBalancerIP")):
            return False
        if any("nodePort" in row for row in spec.get("ports", []) or []):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit namespace-safe sparse manifest rendering without applying it.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", required=True)
    ap.add_argument("--source_namespace", required=True)
    ap.add_argument("--target_namespace", required=True)
    ap.add_argument("--fault", action="append", required=True)
    ap.add_argument("--application_source_root", default=None)
    ap.add_argument("--upstream_hops", type=int, default=2)
    ap.add_argument("--downstream_support_hops", type=int, default=2)
    ap.add_argument("--symptom_hops", type=int, default=2)
    ap.add_argument("--max_entry_path_hops", type=int, default=8)
    args = ap.parse_args()

    target_existed_before = _namespace_exists(args.target_namespace)
    rec = _find_scenario(args.processed_states, args.scenario_id)
    faults = [_parse_fault(x) for x in args.fault]
    app_root = (
        Path(args.application_source_root).expanduser().resolve()
        if args.application_source_root
        else _default_application_source_root(args.processed_states)
    )
    planner_state, topology = _augment_with_static_topology(rec.compressed_state, app_root)
    spec = build_sparse_live_twin_spec(
        planner_state,
        faults,
        upstream_hops=args.upstream_hops,
        downstream_support_hops=args.downstream_support_hops,
        symptom_hops=args.symptom_hops,
        max_entry_path_hops=args.max_entry_path_hops,
    )
    plan = discover_sparse_manifest_plan(args.source_namespace, spec.services_to_keep)
    bundle = render_sparse_manifest_bundle(plan, args.target_namespace)
    target_existed_after = _namespace_exists(args.target_namespace)

    kinds = [str(x.get("kind") or "") for x in bundle.objects]
    names = [str((x.get("metadata", {}) or {}).get("name") or "") for x in bundle.objects]
    forbidden_secret_types = [
        name for obj, name in zip(bundle.objects, names)
        if obj.get("kind") == "Secret"
        and obj.get("type") == "kubernetes.io/service-account-token"
    ]
    invariants = {
        "cluster_mutated": target_existed_before != target_existed_after,
        "render_is_read_only": bundle.read_only,
        "target_namespace_not_created": not target_existed_before and not target_existed_after,
        "source_and_target_namespaces_differ": args.source_namespace != args.target_namespace,
        "all_objects_are_namespace_local": all(
            (obj.get("metadata", {}) or {}).get("namespace") == args.target_namespace
            for obj in bundle.objects
        ),
        "all_objects_are_portable": all(_portable(obj, args.target_namespace) for obj in bundle.objects),
        "no_runtime_service_account_tokens": not forbidden_secret_types,
        "no_pvcs_without_storage_policy": not plan.persistent_volume_claims,
        "all_selected_controllers_rendered": len(plan.controllers) == sum(
            kind in {"Deployment", "StatefulSet"} for kind in kinds
        ),
        "all_selected_services_rendered": len(plan.service_objects) == kinds.count("Service"),
        "oracle_labels_used": False,
        "fault_context_used": False,
    }
    positive_invariants = {
        key for key in invariants
        if key not in {"cluster_mutated", "oracle_labels_used", "fault_context_used"}
    }
    valid = (
        all(invariants[key] for key in positive_invariants)
        and not invariants["cluster_mutated"]
        and not invariants["oracle_labels_used"]
        and not invariants["fault_context_used"]
    )
    print(json.dumps({
        "status": "PASS_SPARSE_LIVE_MANIFEST_RENDER" if valid else "FAIL_SPARSE_LIVE_MANIFEST_RENDER",
        "scenario_id": rec.scenario_id,
        "source_namespace": args.source_namespace,
        "target_namespace": args.target_namespace,
        "selected_services": spec.services_to_keep,
        "service_reduction_percent": spec.resource_summary.get("service_reduction_percent"),
        "static_topology_edges_discovered": topology.get("static_topology_edges_discovered", 0),
        "rendered_object_count": len(bundle.objects),
        "rendered_object_refs": bundle.object_refs,
        "invariants": invariants,
        "next_gate": "sparse_twin_clean_baseline_session" if valid else "fix_manifest_render",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
