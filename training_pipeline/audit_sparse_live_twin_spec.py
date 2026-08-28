from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from digital_twin_runtime.application_topology import discover_application_topology
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


def _state_topology_counts(state: dict) -> tuple[int, int]:
    graph_count = 0
    for edge in (state.get("graph", {}) or {}).get("edges", []) or []:
        if isinstance(edge, dict) and (edge.get("src") or edge.get("source")) and (edge.get("dst") or edge.get("target")):
            graph_count += 1
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2 and edge[0] and edge[1]:
            graph_count += 1

    trace_count = 0
    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", {}) if isinstance(traces, dict) else {}
    if isinstance(per_edge, dict):
        for edge_id, feats in per_edge.items():
            feats = feats if isinstance(feats, dict) else {}
            src, dst = feats.get("source"), feats.get("target")
            if (not src or not dst) and "->" in str(edge_id):
                src, dst = str(edge_id).split("->", 1)
            if src and dst:
                trace_count += 1
    return graph_count, trace_count


def _default_application_source_root(processed_states: str | Path) -> Path:
    # processed_states normally lives at AIOpsLab/processed_states, while the
    # application source submodule lives at AIOpsLab/aiopslab-applications.
    root = Path(processed_states).expanduser().resolve()
    return root.parent / "aiopslab-applications" / "socialNetwork"


def _augment_with_static_topology(compressed_state: dict, application_source_root: Path) -> tuple[dict, dict]:
    state = copy.deepcopy(compressed_state)
    services = {str(s) for s in state.get("services", []) or [] if s}
    topology = discover_application_topology(application_source_root, services)

    graph = state.setdefault("graph", {})
    raw_edges = list(graph.get("edges", []) or [])
    seen: set[tuple[str, str]] = set()
    merged = []

    for edge in raw_edges:
        if isinstance(edge, dict):
            src = edge.get("src") or edge.get("source")
            dst = edge.get("dst") or edge.get("target")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
        else:
            continue
        if not src or not dst:
            continue
        key = (str(src), str(dst))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"src": key[0], "dst": key[1], "features": {"source": "scenario_state"}})

    static_added = 0
    for src, dst in topology.edges:
        key = (str(src), str(dst))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"src": key[0], "dst": key[1], "features": {"source": topology.source_mode}})
        static_added += 1

    graph["edges"] = merged

    return state, {
        "application_source_root": str(application_source_root),
        "application_source_exists": application_source_root.exists(),
        "source_mode": topology.source_mode,
        "source_files_scanned": topology.source_files_scanned,
        "cpp_services_discovered": topology.cpp_services_discovered,
        "lua_frontend_edges_discovered": topology.lua_frontend_edges_discovered,
        "static_topology_edges_discovered": len(topology.edges),
        "static_topology_edges_added": static_added,
        "static_entrypoints": topology.entrypoints,
        "evidence_examples": dict(list(topology.evidence.items())[:12]),
    }


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
    ap.add_argument(
        "--application_source_root",
        default=None,
        help="Optional application source root used for fault-independent static topology discovery. "
             "Defaults to <AIOpsLab>/aiopslab-applications/socialNetwork.",
    )
    ap.add_argument("--upstream_hops", type=int, default=2)
    ap.add_argument("--downstream_support_hops", type=int, default=1)
    ap.add_argument("--symptom_hops", type=int, default=2)
    ap.add_argument("--max_entry_path_hops", type=int, default=8)
    args = ap.parse_args()

    rec = _find_scenario(args.processed_states, args.scenario_id)
    faults = [_parse_fault(x) for x in args.fault]

    original_graph_count, original_trace_count = _state_topology_counts(rec.compressed_state)
    app_root = (
        Path(args.application_source_root).expanduser().resolve()
        if args.application_source_root
        else _default_application_source_root(args.processed_states)
    )
    planner_state, static_topology = _augment_with_static_topology(rec.compressed_state, app_root)

    spec = build_sparse_live_twin_spec(
        planner_state,
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

    static_source_used = int(static_topology.get("static_topology_edges_added", 0) or 0) > 0
    invariants = {
        "agent_state_is_redacted": True,
        "static_topology_is_fault_independent": True,
        "static_topology_uses_oracle_labels": False,
        "oracle_labels_used": False,
        "fault_context_used": False,
        "predicted_roots_present_in_observable_services": roots <= all_services,
        "predicted_roots_kept": roots <= keep,
        "keep_and_prune_disjoint": not bool(keep & prune),
        "keep_and_prune_cover_observable_services": (keep | prune) == all_services,
        "full_application_fallback_used": len(keep) == len(all_services),
        "topology_available": topology_available,
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
            "topology_available",
            "entrypoint_evidence_available",
            "root_reachable_from_entrypoint",
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
            "original_state_graph_edge_records": original_graph_count,
            "original_state_trace_edge_records": original_trace_count,
            "static_source_used": static_source_used,
            **static_topology,
            "planner_graph_edge_records": int(rs.get("graph_edge_records", 0) or 0),
            "planner_trace_edge_records": int(rs.get("trace_edge_records", 0) or 0),
            "deduplicated_planner_edges": topology_edges,
            "entrypoint_services": spec.entrypoint_services,
            "selected_paths": spec.selected_paths,
        },
        "next_gate": "sparse_live_manifest_preflight" if valid else "fix_sparse_twin_spec",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
