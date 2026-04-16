from collections import Counter, defaultdict

from twin_config import (
    STATES_JSONL,
    GRAPH_JSON,
    TWIN_DIR,
    COMPLEXITY_REPORT_JSON,
    TOPOLOGY_JSON,
    FAULT_LIBRARY_JSON,
    ACTION_LIBRARY_JSON,
    OBSERVATION_MODEL_JSON,
    TWIN_SPEC_JSON,
    DEFAULT_ACTIONS,
    DEFAULT_FAULT_TYPES,
)
from twin_utils import ensure_dir, read_json, read_jsonl, write_json


def load_inputs():
    states = read_jsonl(STATES_JSONL)
    graph = read_json(GRAPH_JSON)
    return states, graph


def get_services_from_states(states):
    if not states:
        return []
    first = states[0]
    if "graph" in first and "services" in first["graph"]:
        return sorted(first["graph"]["services"])
    if "metrics" in first:
        return sorted(first["metrics"].keys())
    return []


def get_edges_from_graph(graph):
    edges = graph.get("edges", [])
    out = []
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        feats = e.get("features", {})
        out.append({
            "src": src,
            "dst": dst,
            "edge_id": f"{src}->{dst}",
            "features": feats,
        })
    return out


def compute_observability_summary(states):
    logs_available_count = 0
    metric_available_count = 0
    trace_available_count = 0
    system_available_count = 0

    for st in states:
        # logs
        logs_present = any(
            feats.get("line_count", 0) > 0
            for feats in st.get("logs", {}).values()
        )
        if logs_present:
            logs_available_count += 1

        # metrics
        metrics_present = False
        for feats in st.get("metrics", {}).values():
            if any(feats.get(k, 0.0) not in [0, 0.0] for k in ["cpu", "memory", "network_rx", "network_tx", "restarts"]):
                metrics_present = True
                break
        if metrics_present:
            metric_available_count += 1

        # traces
        if len(st.get("traces", {})) > 0:
            trace_available_count += 1

        # system
        if len(st.get("system", {})) > 0:
            system_available_count += 1

    total = max(1, len(states))

    blind_spots = []
    if logs_available_count == 0:
        blind_spots.append("log_pipeline_empty")
    if metric_available_count == 0:
        blind_spots.append("resource_metric_pipeline_empty")
    if trace_available_count == 0:
        blind_spots.append("trace_pipeline_empty")
    if system_available_count == 0:
        blind_spots.append("system_pipeline_empty")

    return {
        "logs_available_fraction": logs_available_count / total,
        "resource_metrics_available_fraction": metric_available_count / total,
        "trace_available_fraction": trace_available_count / total,
        "system_available_fraction": system_available_count / total,
        "blind_spots": blind_spots,
    }


def compute_state_complexity(states, services):
    if not states:
        return {}

    first = states[0]

    per_service_metric_dim = {}
    per_service_log_dim = {}
    per_service_system_dim = {}

    for svc in services:
        per_service_metric_dim[svc] = len(first.get("metrics", {}).get(svc, {}))
        per_service_log_dim[svc] = len(first.get("logs", {}).get(svc, {}))
        per_service_system_dim[svc] = len(first.get("system", {}).get(svc, {}))

    trace_edge_dim = 0
    traces = first.get("traces", {})
    if traces:
        some_edge = next(iter(traces.values()))
        trace_edge_dim = len(some_edge)

    workload_dim = len(first.get("workload", {}))
    history_dim = len(first.get("history", {}))
    rca_dim = len(first.get("rca_features", {})) if "rca_features" in first else 0

    return {
        "num_snapshots": len(states),
        "per_service_metric_dim": per_service_metric_dim,
        "per_service_log_dim": per_service_log_dim,
        "per_service_system_dim": per_service_system_dim,
        "per_edge_trace_dim": trace_edge_dim,
        "workload_dim": workload_dim,
        "history_dim": history_dim,
        "rca_feature_dim": rca_dim,
    }


def classify_edge_criticality(edge_features):
    er = edge_features.get("error_ratio", 0.0)
    rank = edge_features.get("edge_rank_score", 0.0)
    req = edge_features.get("request_count", 0)

    if er >= 0.9 or rank >= 0.9:
        return "critical"
    if er >= 0.3 or rank >= 0.5 or req >= 20:
        return "important"
    return "normal"


def build_topology(graph, services):
    edges = get_edges_from_graph(graph)

    topo_edges = []
    self_loops = []
    incoming = defaultdict(list)
    outgoing = defaultdict(list)

    for e in edges:
        edge_id = e["edge_id"]
        src = e["src"]
        dst = e["dst"]
        feats = e["features"]

        criticality = classify_edge_criticality(feats)

        topo_edges.append({
            "edge_id": edge_id,
            "src": src,
            "dst": dst,
            "criticality": criticality,
            "observed_request_count": feats.get("request_count", 0),
            "observed_error_ratio": feats.get("error_ratio", 0.0),
            "observed_latency_p95_us": feats.get("latency_p95_us", 0.0),
            "failure_type": feats.get("failure_type", "unknown"),
            "self_loop": src == dst,
        })

        outgoing[src].append(dst)
        incoming[dst].append(src)

        if src == dst:
            self_loops.append(edge_id)

    nodes = []
    for svc in services:
        nodes.append({
            "service": svc,
            "incoming_neighbors": sorted(incoming.get(svc, [])),
            "outgoing_neighbors": sorted(outgoing.get(svc, [])),
        })

    return {
        "services": services,
        "nodes": nodes,
        "edges": topo_edges,
        "self_loops": sorted(self_loops),
        "num_services": len(services),
        "num_edges": len(topo_edges),
    }


def infer_fault_templates(states, topology):
    service_fault_signals = defaultdict(lambda: {
        "trace_error_edges": 0,
        "self_loop_failures": 0,
        "latency_degradation_edges": 0,
        "infra_issue_count": 0,
        "log_anomaly_count": 0,
    })

    for st in states:
        traces = st.get("traces", {})
        for edge, feats in traces.items():
            if "->" not in edge:
                continue
            src, dst = edge.split("->", 1)
            er = feats.get("error_ratio", 0.0)
            ft = feats.get("failure_type", "unknown")

            if er > 0.2:
                service_fault_signals[dst]["trace_error_edges"] += 1
            if src == dst and er > 0.2:
                service_fault_signals[dst]["self_loop_failures"] += 1
            if ft in ["high_latency_path", "degraded_latency_path"]:
                service_fault_signals[dst]["latency_degradation_edges"] += 1

        for svc, feats in st.get("system", {}).items():
            if feats.get("infra_issue_flag", False):
                service_fault_signals[svc]["infra_issue_count"] += 1

        for svc, feats in st.get("logs", {}).items():
            if feats.get("log_anomaly_score", 0.0) > 0.2:
                service_fault_signals[svc]["log_anomaly_count"] += 1

    fault_library = []

    for svc, sig in service_fault_signals.items():
        # dependency / ghost failure
        if sig["trace_error_edges"] > 0 and sig["infra_issue_count"] == 0:
            fault_library.append({
                "fault_id": f"{svc}_ghost_failure",
                "target_service": svc,
                "fault_type": "ghost_failure",
                "trigger_signature": {
                    "trace_error_edges": sig["trace_error_edges"],
                    "self_loop_failures": sig["self_loop_failures"],
                    "infra_issue_count": sig["infra_issue_count"],
                    "log_anomaly_count": sig["log_anomaly_count"],
                },
                "expected_observables": {
                    "pod_health": "healthy",
                    "trace_failure": "high",
                    "logs": "possibly_empty",
                    "resource_metrics": "possibly_uninformative",
                },
            })

        if sig["latency_degradation_edges"] > 0:
            fault_library.append({
                "fault_id": f"{svc}_latency_degradation",
                "target_service": svc,
                "fault_type": "latency_degradation",
                "trigger_signature": {
                    "latency_degradation_edges": sig["latency_degradation_edges"],
                },
                "expected_observables": {
                    "latency": "high",
                    "error_ratio": "low_or_moderate",
                    "pod_health": "healthy_or_degraded",
                },
            })

        if sig["infra_issue_count"] > 0:
            fault_library.append({
                "fault_id": f"{svc}_infra_failure",
                "target_service": svc,
                "fault_type": "infra_failure",
                "trigger_signature": {
                    "infra_issue_count": sig["infra_issue_count"],
                },
                "expected_observables": {
                    "pod_health": "unready_or_crashloop",
                    "trace_failure": "possible",
                    "logs": "possible",
                },
            })

    # always include generic templates
    existing_types = {f["fault_type"] for f in fault_library}
    for ft in DEFAULT_FAULT_TYPES:
        if ft not in existing_types:
            fault_library.append({
                "fault_id": f"generic_{ft}",
                "target_service": None,
                "fault_type": ft,
                "trigger_signature": {},
                "expected_observables": {},
            })

    return fault_library


def build_action_library():
    return [
        {
            "action": "restart_service",
            "parameters": ["service"],
            "expected_effect": [
                "reset application state",
                "possible recovery from transient app failure",
                "may not fix persistent config faults",
            ],
            "risk": "medium",
        },
        {
            "action": "scale_service",
            "parameters": ["service", "replicas"],
            "expected_effect": [
                "improve overload scenarios",
                "unlikely to resolve logic or config fault alone",
            ],
            "risk": "low_to_medium",
        },
        {
            "action": "rollback_config",
            "parameters": ["service"],
            "expected_effect": [
                "good candidate for config_error or ghost_failure caused by bad config",
            ],
            "risk": "medium",
        },
        {
            "action": "disable_edge",
            "parameters": ["src", "dst"],
            "expected_effect": [
                "reduces cascading failures",
                "may degrade functionality while preserving global SLA",
            ],
            "risk": "high",
        },
        {
            "action": "wait",
            "parameters": [],
            "expected_effect": [
                "observe natural evolution",
                "useful for transient faults",
            ],
            "risk": "low",
        },
    ]


def build_observation_model():
    return {
        "latent_to_observed_rules": [
            {
                "latent_fault": "ghost_failure",
                "observed_pattern": {
                    "pod_health": "healthy",
                    "trace_error_ratio": "high",
                    "self_loop_failure": "possible",
                    "logs": "empty_or_sparse",
                    "resource_metrics": "possibly_normal",
                },
            },
            {
                "latent_fault": "infra_failure",
                "observed_pattern": {
                    "pod_health": "unready_or_crashloop",
                    "trace_error_ratio": "moderate_to_high",
                    "logs": "may_exist",
                    "resource_metrics": "may_exist",
                },
            },
            {
                "latent_fault": "latency_degradation",
                "observed_pattern": {
                    "latency": "high",
                    "error_ratio": "low_or_partial",
                    "pod_health": "healthy_or_degraded",
                },
            },
            {
                "latent_fault": "config_error",
                "observed_pattern": {
                    "pod_health": "healthy",
                    "trace_error_ratio": "high",
                    "logs": "may_show_config_keywords_or_may_be_sparse",
                },
            },
        ],
        "observability_gap_rules": [
            {
                "gap": "log_pipeline_empty",
                "meaning": "absence of log evidence should not be interpreted as service health",
            },
            {
                "gap": "resource_metric_pipeline_empty",
                "meaning": "zero resource metrics may indicate missing telemetry, not true idleness",
            },
        ],
    }


def build_complexity_report(states, topology, observability, state_complexity):
    edges = topology["edges"]

    critical_edges = [e for e in edges if e["criticality"] == "critical"]
    important_edges = [e for e in edges if e["criticality"] == "important"]

    services = topology["services"]

    edge_types = Counter()
    for e in edges:
        if e["self_loop"]:
            edge_types["self_loop"] += 1
        else:
            edge_types["service_to_service"] += 1

    temporal_complexity = {
        "num_snapshots": len(states),
        "state_is_time_series": len(states) > 1,
        "supports_fault_progression_modeling": len(states) > 1,
    }

    report = {
        "structural_complexity": {
            "num_services": len(services),
            "num_edges": len(edges),
            "num_self_loops": len(topology["self_loops"]),
            "num_critical_edges": len(critical_edges),
            "num_important_edges": len(important_edges),
            "edge_types": dict(edge_types),
            "services": services,
        },
        "state_complexity": state_complexity,
        "observability_complexity": observability,
        "control_complexity": {
            "num_actions": len(DEFAULT_ACTIONS),
            "actions": DEFAULT_ACTIONS,
        },
        "failure_complexity": {
            "supported_fault_types": DEFAULT_FAULT_TYPES,
            "num_fault_types": len(DEFAULT_FAULT_TYPES),
        },
        "temporal_complexity": temporal_complexity,
        "digital_twin_minimum_requirements": [
            "preserve service topology",
            "preserve dependency edge failures",
            "preserve per-service app vs infra health distinction",
            "preserve observability gaps",
            "support RCA hypothesis injection",
            "support action simulation",
            "emit abstracted state compatible with SLA checker",
        ],
    }

    return report


def build_twin_spec(states, topology, fault_library, action_library, observation_model, complexity_report):
    services = topology["services"]
    edges = topology["edges"]

    spec = {
        "twin_name": "minimal_incident_twin",
        "purpose": [
            "low_cost_mitigation_check",
            "RCA_hypothesis_validation",
            "SLA_based_action_filtering",
        ],
        "latent_state_schema": {
            "node_state": {
                "allowed_values": ["healthy", "degraded", "failed"],
                "fields": [
                    "app_health",
                    "infra_health",
                    "resource_health",
                    "log_health",
                    "latency_health",
                ],
            },
            "edge_state": {
                "allowed_values": ["healthy", "degraded", "failing"],
                "fields": [
                    "request_rate",
                    "error_ratio",
                    "latency_p95_us",
                    "failure_type",
                ],
            },
        },
        "observable_state_schema": {
            "matches_states_jsonl": True,
            "categories": [
                "metrics",
                "logs",
                "traces",
                "system",
                "workload",
                "graph",
                "history",
                "rca_features",
            ],
        },
        "topology": {
            "services": services,
            "num_services": len(services),
            "num_edges": len(edges),
        },
        "fault_injection_interface": {
            "inputs": ["fault_type", "target_service", "severity"],
            "supported_fault_types": sorted({f['fault_type'] for f in fault_library}),
        },
        "action_interface": {
            "inputs": ["action", "parameters"],
            "supported_actions": [a["action"] for a in action_library],
        },
        "validation_interface": {
            "input": "abstracted_state",
            "output": "SLA_result",
            "comparison_mode": "compare_real_vs_simulated_abstraction",
        },
        "assumptions": [
            "topology is approximately fixed during short incident windows",
            "trace-derived graph is a sufficient first approximation",
            "observability gaps must be represented explicitly",
            "first twin is rule-based or hybrid, not fully learned",
        ],
        "complexity_summary": {
            "num_services": complexity_report["structural_complexity"]["num_services"],
            "num_edges": complexity_report["structural_complexity"]["num_edges"],
            "num_self_loops": complexity_report["structural_complexity"]["num_self_loops"],
        },
    }
    return spec


def main():
    ensure_dir(TWIN_DIR)

    states, graph = load_inputs()
    services = get_services_from_states(states)

    topology = build_topology(graph, services)
    observability = compute_observability_summary(states)
    state_complexity = compute_state_complexity(states, services)
    fault_library = infer_fault_templates(states, topology)
    action_library = build_action_library()
    observation_model = build_observation_model()
    complexity_report = build_complexity_report(
        states=states,
        topology=topology,
        observability=observability,
        state_complexity=state_complexity,
    )
    twin_spec = build_twin_spec(
        states=states,
        topology=topology,
        fault_library=fault_library,
        action_library=action_library,
        observation_model=observation_model,
        complexity_report=complexity_report,
    )

    write_json(complexity_report, COMPLEXITY_REPORT_JSON)
    write_json(topology, TOPOLOGY_JSON)
    write_json(fault_library, FAULT_LIBRARY_JSON)
    write_json(action_library, ACTION_LIBRARY_JSON)
    write_json(observation_model, OBSERVATION_MODEL_JSON)
    write_json(twin_spec, TWIN_SPEC_JSON)

    print(f"Wrote: {COMPLEXITY_REPORT_JSON}")
    print(f"Wrote: {TOPOLOGY_JSON}")
    print(f"Wrote: {FAULT_LIBRARY_JSON}")
    print(f"Wrote: {ACTION_LIBRARY_JSON}")
    print(f"Wrote: {OBSERVATION_MODEL_JSON}")
    print(f"Wrote: {TWIN_SPEC_JSON}")


if __name__ == "__main__":
    main()