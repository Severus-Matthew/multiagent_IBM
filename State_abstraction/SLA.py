import json
from pathlib import Path

from config import SERVICES
from utils import write_json


class SLAConfig:
    # --- thresholds (tune later) ---
    MAX_ERROR_RATIO = 0.2
    MAX_CRITICAL_ERROR_RATIO = 0.5

    MAX_LATENCY_MS = 500
    MAX_P95_LATENCY_MS = 800

    MAX_POD_UNREADY = 0
    MAX_CRASHLOOP = 0

    MAX_GLOBAL_ERROR_SERVICES = 1

    # scoring thresholds
    HEALTHY_SCORE = 0.8
    DEGRADED_SCORE = 0.5


def evaluate_service_sla(metrics, logs):
    result = {}

    for svc in SERVICES:
        m = metrics.get(svc, {})
        l = logs.get(svc, {})

        latency = m.get("latency", 0.0)
        log_anomaly = l.get("log_anomaly_score", 0.0)

        healthy = True
        reasons = []

        if latency > SLAConfig.MAX_LATENCY_MS:
            healthy = False
            reasons.append(f"latency_high={latency}")

        if log_anomaly > 0.5:
            healthy = False
            reasons.append(f"log_anomaly={log_anomaly:.3f}")

        result[svc] = {
            "healthy": healthy,
            "reasons": reasons,
            "latency": latency,
            "log_anomaly": log_anomaly,
        }

    return result


def evaluate_dependency_sla(traces):
    result = {}
    violations = []

    for edge, feats in traces.items():
        er = feats.get("error_ratio", 0.0)
        latency_p95 = feats.get("latency_p95_us", 0.0) / 1000.0  # ms

        healthy = True
        reasons = []

        if er > SLAConfig.MAX_ERROR_RATIO:
            healthy = False
            reasons.append(f"error_ratio={er:.3f}")

        if latency_p95 > SLAConfig.MAX_P95_LATENCY_MS:
            healthy = False
            reasons.append(f"latency_p95={latency_p95:.3f}")

        if "->" in edge:
            src, dst = edge.split("->", 1)
            if src == dst and er > 0.3:
                healthy = False
                reasons.append("self_loop_failure")

        result[edge] = {
            "healthy": healthy,
            "error_ratio": er,
            "latency_p95_ms": latency_p95,
            "reasons": reasons,
        }

        if not healthy:
            violations.append(edge)

    return result, violations


def evaluate_system_sla(system):
    result = {}

    for svc, feats in system.items():
        healthy = True
        reasons = []

        if feats.get("pods_unready", 0) > SLAConfig.MAX_POD_UNREADY:
            healthy = False
            reasons.append("pods_unready")

        if feats.get("crashloop_count", 0) > SLAConfig.MAX_CRASHLOOP:
            healthy = False
            reasons.append("crashloop")

        result[svc] = {
            "healthy": healthy,
            "reasons": reasons,
            "pods_ready": feats.get("pods_ready", 0),
        }

    return result


def evaluate_workload_sla(workload, traces):
    total_requests = workload.get("estimated_request_rate", 0)

    error_edges = [
        edge for edge, feats in traces.items()
        if feats.get("error_ratio", 0) > SLAConfig.MAX_ERROR_RATIO
    ]

    workload_ok = True
    reasons = []

    if total_requests == 0:
        workload_ok = False
        reasons.append("no_traffic")

    if len(error_edges) > 0:
        workload_ok = False
        reasons.append("failing_dependencies")

    return {
        "healthy": workload_ok,
        "reasons": reasons,
        "total_requests": total_requests,
    }


def evaluate_global_sla(service_sla, dependency_violations):
    unhealthy_services = [
        svc for svc, s in service_sla.items()
        if not s["healthy"]
    ]

    global_ok = True
    reasons = []

    if len(unhealthy_services) > SLAConfig.MAX_GLOBAL_ERROR_SERVICES:
        global_ok = False
        reasons.append("too_many_unhealthy_services")

    if len(dependency_violations) > 0:
        global_ok = False
        reasons.append("dependency_failures_present")

    return {
        "healthy": global_ok,
        "reasons": reasons,
        "num_unhealthy_services": len(unhealthy_services),
        "num_dependency_violations": len(dependency_violations),
    }


def evaluate_sla(state):
    metrics = state["metrics"]
    logs = state["logs"]
    traces = state["traces"]
    system = state["system"]
    workload = state["workload"]

    service_sla = evaluate_service_sla(metrics, logs)
    dependency_sla, dep_violations = evaluate_dependency_sla(traces)
    system_sla = evaluate_system_sla(system)
    workload_sla = evaluate_workload_sla(workload, traces)
    global_sla = evaluate_global_sla(service_sla, dep_violations)

    sla_result = {
        "service_sla": service_sla,
        "dependency_sla": dependency_sla,
        "system_sla": system_sla,
        "workload_sla": workload_sla,
        "global_sla": global_sla,
        "violated": not global_sla["healthy"],
    }

    return sla_result


def evaluate_states_file(states_path: Path, output_path: Path):
    results = []

    with open(states_path, "r") as f:
        for line in f:
            state = json.loads(line)
            sla = evaluate_sla(state)

            results.append({
                "timestamp": state["timestamp"],
                "sla": sla,
            })

    write_json(results, output_path)
    print(f"SLA evaluation written to {output_path}")