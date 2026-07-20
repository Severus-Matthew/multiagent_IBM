import argparse
from collections import defaultdict
from pathlib import Path
from config import DEFAULT_ACTIONS, DEFAULT_FAULT_TYPES
from utils import read_json, read_jsonl, write_json, ensure_dir


def classify_edge_criticality(feats):
    er = feats.get("error_ratio", 0.0)
    rank = feats.get("edge_rank_score", 0.0)
    req = feats.get("request_count", 0)
    if er >= 0.9 or rank >= 0.9:
        return "critical"
    if er >= 0.3 or rank >= 0.5 or req >= 20:
        return "important"
    return "normal"


def build_topology(graph):
    services = sorted(graph.get("nodes", {}).keys())
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    edges = []
    for e in graph.get("edges", []):
        src, dst = e.get("src"), e.get("dst")
        if not src or not dst:
            continue
        feats = e.get("features", {}) or {}
        outgoing[src].append(dst)
        incoming[dst].append(src)
        edges.append({
            "edge_id": f"{src}->{dst}",
            "src": src,
            "dst": dst,
            "criticality": classify_edge_criticality(feats),
            "observed_request_count": feats.get("request_count", 0),
            "observed_error_ratio": feats.get("error_ratio", 0.0),
            "observed_latency_p95_us": feats.get("latency_p95_us", 0.0),
            "failure_type": feats.get("failure_type", "unknown"),
            "self_loop": src == dst,
        })
    nodes = [{"service": s, "incoming_neighbors": sorted(incoming[s]), "outgoing_neighbors": sorted(outgoing[s])} for s in services]
    return {"services": services, "nodes": nodes, "edges": edges, "num_services": len(services), "num_edges": len(edges)}


def build_fault_library(states, topology):
    sig = defaultdict(lambda: defaultdict(int))
    for st in states:
        fc = st.get("fault_context", {})
        instances = fc.get("fault_instances") or []
        if not instances and fc.get("faulty_service"):
            instances = [{
                "faulty_service": fc.get("faulty_service"),
                "fault_family": fc.get("fault_family"),
                "variant_name": fc.get("variant_name", "default"),
                "variant_params": fc.get("variant_params", {}),
            }]
        for inst in instances:
            svc = inst.get("faulty_service")
            if not svc:
                continue
            family = str(inst.get("fault_family") or fc.get("fault_family") or "known_fault_context")
            sig[svc][family] += 1
            variant_name = inst.get("variant_name")
            if variant_name and variant_name != "default":
                sig[svc][f"variant:{variant_name}"] += 1
        for edge, feats in st.get("traces", {}).items():
            if "->" in edge:
                _, dst = edge.split("->", 1)
                if feats.get("error_ratio", 0.0) > 0.2:
                    sig[dst]["trace_error_edges"] += 1
                if feats.get("latency_p95_us", 0.0) > 100000:
                    sig[dst]["latency_degradation_edges"] += 1
        for svc, sy in st.get("system", {}).items():
            status = sy.get("service_health_status")
            has_direct_infra = status not in (None, "healthy", "unknown")
            if has_direct_infra:
                sig[svc]["infra_issue_count"] += 1
                sig[svc][f"system_status:{status}"] += 1
        for svc, lg in st.get("logs", {}).items():
            if lg.get("log_anomaly_score", 0.0) > 0.2:
                sig[svc]["log_anomaly_count"] += 1
    lib = []
    for svc, s in sig.items():
        if s.get("trace_error_edges", 0) > 0 or any("auth" in k for k in s):
            ft = "auth_failure" if any("auth" in k for k in s) else "dependency_failure"
            lib.append({"fault_id": f"{svc}_{ft}", "target_service": svc, "fault_type": ft, "trigger_signature": dict(s), "expected_observables": {"pod_health": "healthy_possible", "trace_failure": "high_or_missing", "logs": "sparse_or_error"}})
        if s.get("latency_degradation_edges", 0) > 0:
            lib.append({"fault_id": f"{svc}_latency_degradation", "target_service": svc, "fault_type": "latency_degradation", "trigger_signature": dict(s), "expected_observables": {"latency": "high"}})
        if s.get("infra_issue_count", 0) > 0:
            lib.append({"fault_id": f"{svc}_infra_failure", "target_service": svc, "fault_type": "infra_failure", "trigger_signature": dict(s), "expected_observables": {"pod_health": "unready_or_crashloop"}})
        if any(k.startswith("variant:") for k in s):
            lib.append({"fault_id": f"{svc}_variant_specific", "target_service": svc, "fault_type": "variant_specific", "trigger_signature": dict(s), "expected_observables": {"variant": "see trigger_signature"}})
    existing = {f["fault_type"] for f in lib}
    for ft in DEFAULT_FAULT_TYPES:
        if ft not in existing:
            lib.append({"fault_id": f"generic_{ft}", "target_service": None, "fault_type": ft, "trigger_signature": {}, "expected_observables": {}})
    return lib


def build_action_library():
    return [
        {"action": "restart_service", "parameters": ["service"], "expected_effect": ["reset application state", "helps transient app faults"], "risk": "medium"},
        {"action": "scale_service", "parameters": ["service", "replicas"], "expected_effect": ["helps overload/latency"], "risk": "low_to_medium"},
        {"action": "rollback_config", "parameters": ["service"], "expected_effect": ["helps config/auth/misconfiguration"], "risk": "medium"},
        {"action": "disable_edge", "parameters": ["src", "dst"], "expected_effect": ["isolate cascading dependency"], "risk": "high"},
        {"action": "wait", "parameters": [], "expected_effect": ["observe transient recovery"], "risk": "low"},
    ]


def build_observation_model():
    return {"latent_to_observed_rules": [
        {"latent_fault": "auth_failure", "observed_pattern": {"pod_health": "healthy", "trace_error_ratio": "high_or_missing", "logs": "may_be_sparse", "resource_metrics": "normal_possible"}},
        {"latent_fault": "dependency_failure", "observed_pattern": {"trace_error_ratio": "high", "pod_health": "healthy_possible"}},
        {"latent_fault": "infra_failure", "observed_pattern": {"pod_health": "unready_or_crashloop"}},
        {"latent_fault": "latency_degradation", "observed_pattern": {"latency": "high"}},
    ], "observability_gap_rules": [
        {"gap": "trace_pipeline_empty", "meaning": "lack of traces weakens dependency RCA"},
        {"gap": "resource_metric_pipeline_empty", "meaning": "zero metrics can mean missing telemetry"},
        {"gap": "log_pipeline_empty", "meaning": "absence of logs is not service health"},
    ]}


def build_complexity(states, topology):
    edges = topology["edges"]
    return {"structural_complexity": {"num_services": topology["num_services"], "num_edges": topology["num_edges"], "num_critical_edges": sum(e["criticality"] == "critical" for e in edges), "num_important_edges": sum(e["criticality"] == "important" for e in edges)}, "temporal_complexity": {"num_snapshots": len(states), "state_is_time_series": len(states) > 1}, "simulator_minimum_requirements": ["preserve topology", "preserve edge failures", "preserve service app vs infra health", "preserve observability gaps", "support action simulation", "emit SLA-compatible state"]}


def build_specs(processed_dir):
    processed_dir = Path(processed_dir)
    states = read_jsonl(processed_dir / "states.jsonl")
    graph = read_json(processed_dir / "graph.json", {})
    out = processed_dir / "simulator_inputs"
    ensure_dir(out)
    topology = build_topology(graph)
    faults = build_fault_library(states, topology)
    actions = build_action_library()
    obs = build_observation_model()
    complexity = build_complexity(states, topology)
    spec = {"simulator_name": "abstract_behavioral_simulator", "purpose": ["RCA validation", "SLA action filtering", "low-cost mitigation simulation"], "topology": {"num_services": topology["num_services"], "num_edges": topology["num_edges"]}, "fault_injection_interface": {"inputs": ["fault_type", "target_service", "severity"], "supported_fault_types": sorted({f["fault_type"] for f in faults})}, "action_interface": {"inputs": ["action", "parameters"], "supported_actions": [a["action"] for a in actions]}, "observable_state_schema": {"matches_states_jsonl": True}}
    write_json(topology, out / "topology.json")
    write_json(faults, out / "fault_library.json")
    write_json(actions, out / "action_library.json")
    write_json(obs, out / "observation_model.json")
    write_json(complexity, out / "complexity_report.json")
    write_json(spec, out / "simulator_spec.json")
    print(f"[OK] wrote simulator specs to {out}")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    args = ap.parse_args()
    build_specs(args.processed_dir)

if __name__ == "__main__":
    main()
