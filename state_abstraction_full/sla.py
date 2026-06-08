from pathlib import Path
from utils import read_jsonl, write_json

class SLAConfig:
    MAX_ERROR_RATIO = 0.2
    MAX_P95_LATENCY_MS = 800
    MAX_LATENCY_MS = 500
    MAX_LOG_ANOMALY = 0.5
    MAX_POD_UNREADY = 0
    MAX_CRASHLOOP = 0


def evaluate_service_sla(state):
    out = {}
    services = state.get("graph", {}).get("services") or sorted(set(state.get("metrics", {})) | set(state.get("logs", {})) | set(state.get("system", {})))
    for svc in services:
        m = state.get("metrics", {}).get(svc, {})
        l = state.get("logs", {}).get(svc, {})
        sy = state.get("system", {}).get(svc, {})
        reasons = []
        if m.get("latency", 0.0) > SLAConfig.MAX_LATENCY_MS:
            reasons.append(f"latency_high={m.get('latency')}")
        if l.get("log_anomaly_score", 0.0) > SLAConfig.MAX_LOG_ANOMALY:
            reasons.append(f"log_anomaly={l.get('log_anomaly_score'):.3f}")
        if sy.get("pods_unready", 0) > SLAConfig.MAX_POD_UNREADY:
            reasons.append("pods_unready")
        if sy.get("crashloop_count", 0) > SLAConfig.MAX_CRASHLOOP:
            reasons.append("crashloop")
        out[svc] = {"healthy": not reasons, "reasons": reasons}
    return out


def evaluate_dependency_sla(state):
    result = {}
    violations = []
    for edge, feats in state.get("traces", {}).items():
        reasons = []
        er = feats.get("error_ratio", 0.0)
        p95_ms = feats.get("latency_p95_us", 0.0) / 1000.0
        if er > SLAConfig.MAX_ERROR_RATIO:
            reasons.append(f"error_ratio={er:.3f}")
        if p95_ms > SLAConfig.MAX_P95_LATENCY_MS:
            reasons.append(f"latency_p95_ms={p95_ms:.3f}")
        if reasons:
            violations.append(edge)
        result[edge] = {"healthy": not reasons, "error_ratio": er, "latency_p95_ms": p95_ms, "reasons": reasons}
    return result, violations


def evaluate_workload_sla(state):
    w = state.get("workload", {})
    reasons = []
    if w.get("estimated_request_rate", 0) == 0:
        reasons.append("no_trace_traffic_observed")
    return {"healthy": not reasons, "reasons": reasons, "total_requests": w.get("estimated_request_rate", 0)}


def evaluate_sla(state):
    service = evaluate_service_sla(state)
    dep, dep_violations = evaluate_dependency_sla(state)
    workload = evaluate_workload_sla(state)
    unhealthy = [s for s, v in service.items() if not v["healthy"]]
    global_reasons = []
    if dep_violations:
        global_reasons.append("dependency_failures_present")
    if len(unhealthy) > 1:
        global_reasons.append("multiple_unhealthy_services")
    # Do not fail global solely because trace traffic is absent; mark as observability gap.
    return {
        "service_sla": service,
        "dependency_sla": dep,
        "workload_sla": workload,
        "global_sla": {
            "healthy": not global_reasons,
            "reasons": global_reasons,
            "num_unhealthy_services": len(unhealthy),
            "num_dependency_violations": len(dep_violations),
        },
        "violated": bool(global_reasons),
    }


def evaluate_states_file(states_path, output_path):
    rows = read_jsonl(states_path)
    results = [{"timestamp": st.get("timestamp"), "sla": evaluate_sla(st)} for st in rows]
    write_json(results, output_path)
    return results
