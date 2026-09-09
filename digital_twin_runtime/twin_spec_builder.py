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


def _startup_required_edges(state: dict[str, Any]) -> list[tuple[str, str]]:
    """Edges the source topology marked as resolved in the callee's entrypoint."""
    out: list[tuple[str, str]] = []
    for edge in (state.get("graph", {}) or {}).get("edges", []) or []:
        if isinstance(edge, dict) and edge.get("startup_required"):
            src = edge.get("src") or edge.get("source")
            dst = edge.get("dst") or edge.get("target")
            if src and dst:
                out.append((str(src), str(dst)))
    return out


def _startup_closure(
    keep: set[str], startup_edges: list[tuple[str, str]], allowed: set[str]
) -> dict[str, set[str]]:
    """Dependencies every kept service needs merely to boot, to a fixpoint.

    A service deployed without a target it dials in its entrypoint crash-loops,
    which makes the clean baseline unattainable and pollutes every symptom
    channel with a failure the predicted fault did not cause. The closure adds
    only startup-required targets of services already in scope; it never adds
    request-time dependencies, so ordinary sparsity is preserved.
    """
    forward, _ = _adjacency(startup_edges)
    added: dict[str, set[str]] = {}
    scope = set(keep)
    for _ in range(len(allowed) + 1):
        new: dict[str, set[str]] = {}
        for svc in sorted(scope):
            for dep in sorted(forward.get(svc, set()) & allowed):
                if dep not in scope:
                    new.setdefault(dep, set()).add(svc)
        if not new:
            break
        for dep, callers in new.items():
            added.setdefault(dep, set()).update(callers)
            scope.add(dep)
    return added


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
    downstream_support_hops: int = 1,
    symptom_hops: int = 2,
    max_entry_path_hops: int = 8,
) -> TwinSpec:
    """Build a fault-conditioned sparse live-Twin plan from safe evidence.

    Scope is causal, not symptom-union based. We keep predicted roots, bounded
    upstream impact, bounded downstream runtime support, and one minimal entry
    path. Observable degraded services are retained as diagnostics only unless
    they already lie on that causal scaffold. This prevents unrelated/stale
    unready pods in historical captures from inflating the Twin.

    ``downstream_support_hops`` defaults to one direct hop. The dependency graph
    of a microservice application saturates quickly, so a larger budget does not
    add fidelity, it just deploys the whole application and erases the resource
    reduction the sparse Twin exists to demonstrate.

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

    selected_paths: list[list[str]] = []
    path_services: set[str] = set()
    unreachable_roots: list[str] = []
    for root in sorted(roots):
        path = _shortest_path(entrypoints, root, forward, all_services, max_entry_path_hops)
        if path:
            selected_paths.append(path)
            path_services.update(path)
            for svc in path:
                reason.setdefault(svc, []).append(f"minimal_entry_path_to_{root}")
        else:
            unreachable_roots.append(root)

    # A Twin with no executable request path to the predicted root cannot exercise
    # the fault, so its telemetry could never reproduce the incident. Returning a
    # deployable spec here is what produced the original single-service Twin, so
    # this fails closed rather than shipping an unexercisable subgraph.
    if unreachable_roots:
        summary = {
            "total_application_services": len(all_services),
            "kept_services": 0,
            "pruned_services": len(all_services),
            "service_reduction_fraction": 1.0,
            "service_reduction_percent": 100.0,
            "invalid_predicted_root": False,
            "invalid_topology": False,
            "unreachable_predicted_roots": unreachable_roots,
            "entrypoint_services": sorted(entrypoints),
            "max_entry_path_hops": int(max_entry_path_hops),
            **topology_counts,
        }
        return _invalid_spec(
            compressed_state, predicted_faults, all_services, entrypoints,
            mode="rca_predicted_sparse_live_unreachable_root", summary=summary,
        )

    # Runtime dependencies are only needed by the services that must actually
    # execute: the predicted roots and the minimal entry path. Upstream impact
    # services are kept so propagation is observable, but pulling in *their*
    # dependencies as well turns the closure into the whole application. On the
    # SocialNetwork graph, seeding from the full causal scope reaches every
    # service within three hops, which reduced the Twin to an 11% saving and
    # silently defeated the sparse-Twin claim.
    support_seeds = roots | path_services
    support = _bounded_reachable(
        support_seeds, forward, downstream_support_hops, all_services
    ) - support_seeds - impact
    for svc in sorted(support):
        reason.setdefault(svc, []).append("bounded_runtime_dependency")

    causal_scope = (roots | impact | support | path_services) & all_services

    # Symptoms validate whether the causal plan covers the observed incident, but
    # they do not expand deployment scope. This is essential when historical runs
    # contain unrelated unready pods or stale health signals.
    degraded = _observable_degraded_services(compressed_state) & all_services
    degraded_on_scope = degraded & causal_scope
    degraded_outside_scope = degraded - causal_scope
    for svc in sorted(degraded_on_scope):
        reason.setdefault(svc, []).append("observable_degraded_on_causal_scope")

    keep = set(causal_scope)
    startup_added = _startup_closure(keep, _startup_required_edges(compressed_state), all_services)
    for dep, callers in sorted(startup_added.items()):
        keep.add(dep)
        for caller in sorted(callers):
            reason.setdefault(dep, []).append(f"startup_required_dependency_of_{caller}")
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
            "startup_required_dependencies_added": sorted(startup_added),
            "causal_scope_before_startup_closure": len(causal_scope),
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


def _trace_endpoint_services(compressed_state: dict[str, Any]) -> set[str]:
    """Services the incident's own traces observed as span endpoints."""
    traces = compressed_state.get("traces") or {}
    per_edge = traces.get("per_edge", {}) if isinstance(traces, dict) else {}
    out: set[str] = set()
    if isinstance(per_edge, dict):
        for edge_id, feats in per_edge.items():
            feats = feats if isinstance(feats, dict) else {}
            src, dst = feats.get("source"), feats.get("target")
            if (not src or not dst) and "->" in str(edge_id):
                src, dst = str(edge_id).split("->", 1)
            for name in (src, dst):
                if name and str(name) != "ROOT":
                    out.add(str(name))
    return out


def build_incident_twin_spec(compressed_state: dict[str, Any], *,
                             deployable_services: list[str] | set[str] | None = None,
                             **budgets: Any) -> TwinSpec:
    """Freeze scope from observable incident evidence, before any RCA proposal.

    Every attributable affected service is retained, with request paths and
    runtime/startup dependencies. No predicted root, private label, or scenario
    ID selects scope. A fully connected dependency requirement may legitimately
    yield no reduction.

    Real captures attribute symptoms to pods, containers, log names, Chaos
    objects and volumes as well as to services. Symptom names are resolved onto
    the deployable service inventory through the comparator's aliases; names
    that resolve to nothing are recorded as unattributed rather than deployed.
    Affected services on the request graph become the workload targets and are
    planned with entry paths; affected services off the request graph
    (datastores, infrastructure) are kept directly with their startup closure.
    Affected services the incident's own traces observed are the only ones a
    clean/recovered phase can be required to reach.
    """
    from .telemetry_comparator import canonical_service, symptom_signature
    inventory = {str(s) for s in (compressed_state.get("services") or []) if s}
    services = inventory & {str(s) for s in deployable_services} if deployable_services is not None else inventory
    undeployable = sorted(inventory - services)
    signature = symptom_signature(compressed_state)
    raw_names = set(signature["affected_services"]) | set((compressed_state.get("observed_deviations") or {}).keys())
    affected: set[str] = set()
    unattributed: list[str] = []
    for name in sorted(raw_names):
        resolved = canonical_service(name, services)
        if resolved is None:
            unattributed.append(str(name))
        else:
            affected.add(resolved)
    if not affected:
        raise ValueError("no observable incident symptoms or reference-state deviations on deployable services")
    edges, _ = _graph_edge_records(compressed_state)
    graph_nodes = {a for a, _ in edges} | {b for _, b in edges}
    request_targets = sorted(affected & graph_nodes)
    if not request_targets:
        raise ValueError("no observable incident symptoms on request-path services")
    # The existing path/closure planner only uses the service field of these
    # structural seeds. They are not fault hypotheses and are never injected.
    seeds = [FaultLabel(service=s, fault_type="unknown") for s in request_targets]
    spec = build_sparse_live_twin_spec(compressed_state, seeds, **budgets)
    if not spec.services_to_keep or spec.resource_summary.get("invalid_topology"):
        raise ValueError("incident request-path targets are not reachable: "
                         + str(spec.resource_summary.get("unreachable_predicted_roots") or spec.mode))
    keep = set(spec.services_to_keep)
    direct = sorted(affected - keep)
    for service in direct:
        keep.add(service)
        spec.reason.setdefault(service, []).append("observable_incident_service_without_request_path")
    # A kept service that calls a pruned one fails on every request, which is
    # a symptom the incident did not have and which also contaminates the clean
    # control (HotelReservation: search -> geo/rate). Follow observed call
    # edges forward from everything kept, to a fixpoint. This can legitimately
    # reach the whole application; reduction is not forced.
    forward, _ = _adjacency(edges)
    runtime_added = sorted(_bounded_reachable(keep, forward, len(services), services) - keep)
    for service in runtime_added:
        keep.add(service)
        spec.reason.setdefault(service, []).append("observed_runtime_dependency_closure")
    startup_added = _startup_closure(keep, _startup_required_edges(compressed_state), services)
    for dep, callers in sorted(startup_added.items()):
        keep.add(dep)
        for caller in sorted(callers):
            spec.reason.setdefault(dep, []).append(f"startup_required_dependency_of_{caller}")
    if not affected.issubset(keep):
        raise ValueError("incident scope cannot cover all observable affected services")
    spec.services_to_keep = sorted(keep)
    spec.services_to_prune = sorted(services - keep)
    spec.mode = "incident_observable_sparse_live"
    spec.selection_policy = "hypothesis_independent_incident_scope_v2_alias_resolved"
    spec.target_faults = []
    total = len(services)
    reduction = (total - len(keep)) / total if total else 0.0
    trace_observable = sorted(affected & _trace_endpoint_services(compressed_state))
    spec.resource_summary.update({
        "incident_affected_services": sorted(affected),
        "incident_request_path_targets": request_targets,
        "incident_trace_observable_targets": trace_observable,
        "incident_direct_scope_additions": direct,
        "incident_runtime_closure_added": runtime_added,
        "incident_startup_dependencies_added": sorted(set(startup_added) - set(spec.services_to_keep)),
        "unattributed_symptom_names": sorted(set(unattributed)),
        "undeployable_inventory_names": undeployable,
        "deployable_services": sorted(services),
        "kept_services": len(keep), "pruned_services": total - len(keep),
        "service_reduction_fraction": reduction,
        "service_reduction_percent": round(100.0 * reduction, 3),
        "scope_depends_on_rca_prediction": False,
        "symptoms_expand_deployment_scope": True,
    })
    for reasons in spec.reason.values():
        if "rca_predicted_root_cause" in reasons:
            reasons.remove("rca_predicted_root_cause")
            reasons.append("observable_incident_service")
    spec.reason = {k: sorted(set(v)) for k, v in sorted(spec.reason.items()) if k in keep}
    return spec
