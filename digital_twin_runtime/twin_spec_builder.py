from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Any

from training_pipeline.schemas import FaultLabel


@dataclass
class TwinSpec:
    scenario_id: str
    namespace: str | None
    mode: str
    services_to_keep: list[str]
    services_to_prune: list[str]
    target_faults: list[dict[str, Any]]
    reason: dict[str, list[str]] = field(default_factory=dict)
    impact_services: list[str] = field(default_factory=list)
    support_services: list[str] = field(default_factory=list)
    entrypoint_services: list[str] = field(default_factory=list)
    selected_paths: list[list[str]] = field(default_factory=list)
    selection_policy: str = "legacy"
    resource_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _graph_edge_records(state: dict[str, Any]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Return observable service edges from both graph and compressed traces.

    Some historical processed states have an empty/partial ``graph.edges`` even
    though ``traces.per_edge`` contains the actual request topology. The sparse
    live Twin must not silently interpret missing graph serialization as a
    one-service application, so we merge both observable sources here.
    """
    edges: list[tuple[str, str]] = []
    graph_count = 0
    trace_count = 0

    for edge in (state.get("graph", {}) or {}).get("edges", []) or []:
        if isinstance(edge, dict):
            src = edge.get("src") or edge.get("source")
            dst = edge.get("dst") or edge.get("target")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
        else:
            continue
        if src and dst:
            edges.append((str(src), str(dst)))
            graph_count += 1

    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", {}) if isinstance(traces, dict) else {}
    if isinstance(per_edge, dict):
        for edge_id, feats in per_edge.items():
            feats = feats if isinstance(feats, dict) else {}
            src = feats.get("source")
            dst = feats.get("target")
            if (not src or not dst) and "->" in str(edge_id):
                src, dst = str(edge_id).split("->", 1)
            if src and dst:
                edges.append((str(src), str(dst)))
                trace_count += 1

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if edge not in seen:
            deduped.append(edge)
            seen.add(edge)

    return deduped, {
        "graph_edge_records": graph_count,
        "trace_edge_records": trace_count,
        "deduplicated_observable_edges": len(deduped),
    }


def _graph_edges(state: dict[str, Any]) -> list[tuple[str, str]]:
    return _graph_edge_records(state)[0]


def _adjacency(edges: list[tuple[str, str]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    forward: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for src, dst in edges:
        forward.setdefault(src, set()).add(dst)
        reverse.setdefault(dst, set()).add(src)
    return forward, reverse


def _neighbors(state: dict[str, Any], service: str) -> set[str]:
    keep = {service}
    for src, dst in _graph_edges(state):
        if src == service and dst:
            keep.add(dst)
        if dst == service and src:
            keep.add(src)
    return keep


def _bounded_reachable(
    starts: set[str],
    adjacency: dict[str, set[str]],
    max_hops: int,
    allowed: set[str],
) -> set[str]:
    seen = set(starts) & allowed
    frontier = set(seen)
    for _ in range(max(0, int(max_hops))):
        nxt: set[str] = set()
        for node in frontier:
            nxt.update(adjacency.get(node, set()))
        nxt &= allowed
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def _shortest_path(
    starts: set[str],
    target: str,
    forward: dict[str, set[str]],
    allowed: set[str],
    max_hops: int,
) -> list[str]:
    starts = {s for s in starts if s in allowed}
    if target not in allowed or not starts:
        return []
    if target in starts:
        return [target]

    q: deque[tuple[str, list[str]]] = deque((s, [s]) for s in sorted(starts))
    seen = set(starts)
    while q:
        node, path = q.popleft()
        if len(path) - 1 >= max_hops:
            continue
        for nxt in sorted(forward.get(node, set())):
            if nxt not in allowed or nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt == target:
                return new_path
            seen.add(nxt)
            q.append((nxt, new_path))
    return []


def _entrypoints(all_services: set[str], edges: list[tuple[str, str]]) -> set[str]:
    explicit = {
        dst for src, dst in edges
        if str(src).upper() == "ROOT" and dst in all_services
    }
    if explicit:
        return explicit

    indegree = {svc: 0 for svc in all_services}
    outdegree = {svc: 0 for svc in all_services}
    for src, dst in edges:
        if src in all_services and dst in all_services:
            indegree[dst] += 1
            outdegree[src] += 1
    return {
        svc for svc in all_services
        if indegree.get(svc, 0) == 0 and outdegree.get(svc, 0) > 0
    }


def _observable_degraded_services(state: dict[str, Any]) -> set[str]:
    degraded: set[str] = set()
    for svc, h in (state.get("service_health", {}) or {}).items():
        if isinstance(h, dict) and str(h.get("status", "healthy")).lower() not in {"healthy", "unknown", ""}:
            degraded.add(str(svc))
    for svc, info in (state.get("system", {}) or {}).items():
        health = info.get("health", {}) if isinstance(info, dict) else {}
        if (
            health.get("infra_issue_flag")
            or float(health.get("pods_unready", 0) or 0) > 0
            or float(health.get("crashloop_count", 0) or 0) > 0
            or float(health.get("oomkilled_count", 0) or 0) > 0
        ):
            degraded.add(str(svc))
    return degraded


def _invalid_spec(
    compressed_state: dict[str, Any],
    predicted_faults: list[FaultLabel],
    all_services: set[str],
    entrypoints: set[str],
    *,
    mode: str,
    summary: dict[str, Any],
) -> TwinSpec:
    return TwinSpec(
        scenario_id=str(compressed_state.get("scenario_id", "unknown")),
        namespace=compressed_state.get("namespace"),
        mode=mode,
        services_to_keep=[],
        services_to_prune=sorted(all_services),
        target_faults=[x.to_dict() for x in predicted_faults],
        reason={},
        impact_services=[],
        support_services=[],
        entrypoint_services=sorted(entrypoints),
        selected_paths=[],
        selection_policy="fault_conditioned_sparse_live_v3",
        resource_summary=summary,
    )


def build_sparse_live_twin_spec(
    compressed_state: dict[str, Any],
    predicted_faults: list[FaultLabel],
    *,
    upstream_hops: int = 2,
    downstream_support_hops: int = 2,
    symptom_hops: int = 2,
    max_entry_path_hops: int = 8,
) -> TwinSpec:
    """Build a fault-conditioned sparse live-Twin plan from safe evidence.

    Scope is causal, not symptom-union based. We keep predicted roots, bounded
    upstream impact, bounded downstream runtime support, and one minimal entry
    path. Observable degraded services are retained as diagnostics only unless
    they already lie on that causal scaffold. This prevents unrelated/stale
    unready pods in historical captures from inflating the Twin.

    Hidden labels, fault_context, scenario-name hints and injection manifests are
    never consulted here.
    """
    all_services = {str(s) for s in (compressed_state.get("services", []) or []) if s}
    edges, topology_counts = _graph_edge_records(compressed_state)
    forward, reverse = _adjacency(edges)
    entrypoints = _entrypoints(all_services, edges)

    roots = {
        str(f.service) for f in predicted_faults
        if f.service and str(f.service) in all_services
    }

    if not roots:
        summary = {
            "total_application_services": len(all_services),
            "kept_services": 0,
            "pruned_services": len(all_services),
            "service_reduction_fraction": 1.0 if all_services else 0.0,
            "invalid_predicted_root": True,
            "invalid_topology": False,
            **topology_counts,
        }
        return _invalid_spec(
            compressed_state, predicted_faults, all_services, entrypoints,
            mode="rca_predicted_sparse_live_invalid_root", summary=summary,
        )

    if len(all_services) > 1 and not edges:
        summary = {
            "total_application_services": len(all_services),
            "kept_services": 0,
            "pruned_services": len(all_services),
            "service_reduction_fraction": 1.0,
            "service_reduction_percent": 100.0,
            "invalid_predicted_root": False,
            "invalid_topology": True,
            **topology_counts,
        }
        return _invalid_spec(
            compressed_state, predicted_faults, all_services, entrypoints,
            mode="rca_predicted_sparse_live_missing_observable_topology", summary=summary,
        )

    reason: dict[str, list[str]] = {}
    for root in sorted(roots):
        reason.setdefault(root, []).append("rca_predicted_root_cause")

    impact = _bounded_reachable(roots, reverse, upstream_hops, all_services)
    for svc in sorted(impact - roots):
        reason.setdefault(svc, []).append("bounded_upstream_impact")

    support = _bounded_reachable(roots, forward, downstream_support_hops, all_services) - roots
    for svc in sorted(support):
        reason.setdefault(svc, []).append("bounded_downstream_support")

    selected_paths: list[list[str]] = []
    path_services: set[str] = set()
    for root in sorted(roots):
        path = _shortest_path(entrypoints, root, forward, all_services, max_entry_path_hops)
        if path:
            selected_paths.append(path)
            path_services.update(path)
            for svc in path:
                reason.setdefault(svc, []).append(f"minimal_entry_path_to_{root}")

    causal_scope = (roots | impact | support | path_services) & all_services

    # Symptoms validate whether the causal plan covers the observed incident, but
    # they do not expand deployment scope. This is essential when historical runs
    # contain unrelated unready pods or stale health signals.
    degraded = _observable_degraded_services(compressed_state) & all_services
    degraded_on_scope = degraded & causal_scope
    degraded_outside_scope = degraded - causal_scope
    for svc in sorted(degraded_on_scope):
        reason.setdefault(svc, []).append("observable_degraded_on_causal_scope")

    keep = causal_scope
    prune = all_services - keep

    total = len(all_services)
    kept = len(keep)
    reduction = (total - kept) / total if total else 0.0

    return TwinSpec(
        scenario_id=str(compressed_state.get("scenario_id", "unknown")),
        namespace=compressed_state.get("namespace"),
        mode="rca_predicted_sparse_live",
        services_to_keep=sorted(keep),
        services_to_prune=sorted(prune),
        target_faults=[x.to_dict() for x in predicted_faults],
        reason={k: sorted(set(v)) for k, v in sorted(reason.items()) if k in keep},
        impact_services=sorted(impact),
        support_services=sorted(support),
        entrypoint_services=sorted(entrypoints),
        selected_paths=selected_paths,
        selection_policy="fault_conditioned_sparse_live_v3",
        resource_summary={
            "total_application_services": total,
            "kept_services": kept,
            "pruned_services": len(prune),
            "service_reduction_fraction": reduction,
            "service_reduction_percent": round(100.0 * reduction, 3),
            "upstream_hops": int(upstream_hops),
            "downstream_support_hops": int(downstream_support_hops),
            "symptom_hops": int(symptom_hops),
            "symptom_hops_role": "diagnostic_only_v3",
            "max_entry_path_hops": int(max_entry_path_hops),
            "invalid_predicted_root": False,
            "invalid_topology": False,
            "symptoms_expand_deployment_scope": False,
            "observable_degraded_services": sorted(degraded),
            "observable_degraded_on_causal_scope": sorted(degraded_on_scope),
            "observable_degraded_outside_causal_scope": sorted(degraded_outside_scope),
            **topology_counts,
        },
    )


def build_predicted_twin_spec(compressed_state: dict[str, Any], predicted_faults: list[FaultLabel]) -> TwinSpec:
    """Legacy offline predicted Twin spec; retained for old diagnostics only."""
    all_services = set(compressed_state.get("services", []) or [])
    keep: set[str] = set()
    reason: dict[str, list[str]] = {}
    for fault in predicted_faults:
        keep.add(fault.service)
        reason.setdefault(fault.service, []).append("rca_predicted_root_cause")
        for n in _neighbors(compressed_state, fault.service):
            keep.add(n)
            reason.setdefault(n, []).append(f"neighbor_of_{fault.service}")
    for svc, h in (compressed_state.get("service_health", {}) or {}).items():
        if isinstance(h, dict) and h.get("status", "healthy") != "healthy":
            keep.add(svc)
            reason.setdefault(svc, []).append("redacted_service_health_degraded")
    for svc, info in (compressed_state.get("system", {}) or {}).items():
        health = info.get("health", {}) if isinstance(info, dict) else {}
        if health.get("infra_issue_flag") or health.get("pods_unready", 0) > 0:
            keep.add(svc)
            reason.setdefault(svc, []).append("redacted_system_infra_signal")
    if not keep:
        keep = set(all_services)
        for svc in keep:
            reason.setdefault(svc, []).append("fallback_keep_all_no_signal")
    return TwinSpec(
        compressed_state.get("scenario_id", "unknown"),
        compressed_state.get("namespace"),
        "rca_predicted_redacted",
        sorted(keep & all_services),
        sorted(all_services - keep),
        [x.to_dict() for x in predicted_faults],
        reason,
    )


def build_oracle_twin_spec(full_state: dict[str, Any], gt_faults: list[FaultLabel]) -> TwinSpec:
    """Oracle spec for offline evaluation only; agents must never see it."""
    all_services = set(full_state.get("services", []) or [])
    keep = {f.service for f in gt_faults if f.service}
    reason = {s: ["oracle_ground_truth_fault"] for s in keep}
    for svc in list(keep):
        for n in _neighbors(full_state, svc):
            keep.add(n)
            reason.setdefault(n, []).append(f"oracle_neighbor_of_{svc}")
    ns = (full_state.get("fault_context", {}) or {}).get("target_namespace")
    return TwinSpec(
        full_state.get("scenario_id", "unknown"),
        ns,
        "oracle_offline",
        sorted(keep & all_services),
        sorted(all_services - keep),
        [x.to_dict() for x in gt_faults],
        reason,
    )
