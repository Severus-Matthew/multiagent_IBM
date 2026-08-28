from __future__ import annotations

import argparse
import json
from pathlib import Path

from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec
from training_pipeline.data_loader import iter_scenarios
from training_pipeline.schemas import FaultLabel, normalize_fault_type


def _parse_fault(text: str) -> FaultLabel:
    if "::" not in text:
        raise ValueError(f"fault must be SERVICE::FAULT_TYPE, got {text!r}")
    service, fault_type = text.split("::", 1)
    service = service.strip()
    fault_type = normalize_fault_type(fault_type.strip())
    if not service:
        raise ValueError("fault service cannot be empty")
    return FaultLabel(service=service, fault_type=fault_type)


def _find_scenario(processed_states: str | Path, scenario_id: str):
    exact = None
    partial = []
    for rec in iter_scenarios(processed_states):
        if rec.scenario_id == scenario_id or rec.scenario_dir.name == scenario_id:
            exact = rec
            break
        if scenario_id in rec.scenario_id or scenario_id in rec.scenario_dir.name:
            partial.append(rec)
    if exact is not None:
        return exact
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise RuntimeError(
            "scenario_id is ambiguous; matches: "
            + ", ".join(r.scenario_id for r in partial[:20])
        )
    raise RuntimeError(f"scenario not found: {scenario_id}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read-only audit of the fault-conditioned sparse live-Twin service plan."
    )
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_id", required=True)
    ap.add_argument(
        "--fault",
        action="append",
        required=True,
        help="Predicted fault as SERVICE::FAULT_TYPE; repeat for multifault.",
    )
    ap.add_argument("--upstream_hops", type=int, default=2)
    ap.add_argument("--downstream_support_hops", type=int, default=1)
    ap.add_argument("--symptom_hops", type=int, default=2)
    ap.add_argument("--max_entry_path_hops", type=int, default=8)
    args = ap.parse_args()

    rec = _find_scenario(args.processed_states, args.scenario_id)
    faults = [_parse_fault(x) for x in args.fault]
    spec = build_sparse_live_twin_spec(
        rec.compressed_state,
        faults,
        upstream_hops=args.upstream_hops,
        downstream_support_hops=args.downstream_support_hops,
        symptom_hops=args.symptom_hops,
        max_entry_path_hops=args.max_entry_path_hops,
    )

    all_services = set(rec.compressed_state.get("services", []) or [])
    roots = {f.service for f in faults}
    keep = set(spec.services_to_keep)
    prune = set(spec.services_to_prune)
    rs = spec.resource_summary or {}
    topology_edges = int(rs.get("deduplicated_observable_edges", 0) or 0)
    topology_available = len(all_services) <= 1 or topology_edges > 0
    path_available = bool(spec.selected_paths)

    invariants = {
        "uses_redacted_compressed_state_only": True,
        "oracle_labels_used": False,
        "fault_context_used": False,
        "predicted_roots_present_in_observable_services": roots <= all_services,
        "predicted_roots_kept": roots <= keep,
        "keep_and_prune_disjoint": not bool(keep & prune),
        "keep_and_prune_cover_observable_services": (keep | prune) == all_services,
        "full_application_fallback_used": len(keep) == len(all_services),
        "observable_topology_available": topology_available,
        "entrypoint_evidence_available": bool(spec.entrypoint_services),
        "root_reachable_from_entrypoint": path_available,
        "cluster_mutated": False,
    }

    valid = all(
        invariants[k]
        for k in (
            "predicted_roots_present_in_observable_services",
            "predicted_roots_kept",
            "keep_and_prune_disjoint",
            "keep_and_prune_cover_observable_services",
            "observable_topology_available",
        )
    ) and not rs.get("invalid_predicted_root", False) and not rs.get("invalid_topology", False)

    report = {
        "status": "PASS_SPARSE_LIVE_TWIN_SPEC" if valid else "FAIL_SPARSE_LIVE_TWIN_SPEC",
        "scenario_id": rec.scenario_id,
        "scenario_dir": str(rec.scenario_dir),
        "predicted_faults": [f.to_dict() for f in faults],
        "twin_spec": spec.to_dict(),
        "invariants": invariants,
        "topology_diagnostics": {
            "graph_edge_records": int(rs.get("graph_edge_records", 0) or 0),
            "trace_edge_records": int(rs.get("trace_edge_records", 0) or 0),
            "deduplicated_observable_edges": topology_edges,
            "entrypoint_services": spec.entrypoint_services,
            "selected_paths": spec.selected_paths,
        },
        "next_gate": "sparse_live_manifest_preflight" if valid else "fix_sparse_twin_spec",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
