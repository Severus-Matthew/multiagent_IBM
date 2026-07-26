from collections import defaultdict


def infer_fault_type(fault_context):
    """Map a fault-family-like string to a coarse type.

    This helper is kept for evaluator/offline utilities. `build_rca_features` below
    intentionally does not use injected fault context as evidence, because its
    output can be copied into agent-visible compressed state.
    """
    fam = str(fault_context.get("fault_family") or "").lower()
    if fam == "multifault":
        return "multifault"
    if "auth" in fam and "mongo" in fam:
        return "auth_failure"
    if "mongo" in fam:
        return "dependency_failure"
    if "latency" in fam or "delay" in fam:
        return "latency_degradation"
    if "network_loss" in fam or "loss" in fam:
        return "network_failure"
    if "assign_non_existent_node" in fam or "non_existent_node" in fam:
        return "infra_failure"
    if "scale_pod" in fam or "pod_failure" in fam or "pod_kill" in fam or "container_kill" in fam:
        return "infra_failure"
    if "cpu" in fam or "memory" in fam or "oom" in fam:
        return "resource_exhaustion"
    if "config" in fam or "misconfig" in fam or "wrong_bin" in fam:
        return "config_error"
    return "unknown"


def infer_instance_fault_type(instance):
    return infer_fault_type({"fault_family": instance.get("fault_family")})


def _observable_error_family(best_score, logs, metrics, traces, system):
    """Infer a coarse dominant error family from observable signals only."""
    for svc, sy in system.items():
        if sy.get("infra_issue_flag"):
            return "infra_failure"
    for svc, l in logs.items():
        dom = str(l.get("dominant_error_type") or "").lower()
        if any(x in dom for x in ["auth", "permission", "forbidden", "unauthorized"]):
            return "auth_failure"
        if any(x in dom for x in ["mongo", "database", "db"]):
            return "dependency_failure"
        if any(x in dom for x in ["timeout", "latency", "slow"]):
            return "latency_degradation"
    for edge, t in traces.items():
        failure_type = str(t.get("failure_type") or "").lower()
        if any(x in failure_type for x in ["timeout", "latency", "slow"]):
            return "latency_degradation"
        if any(x in failure_type for x in ["network", "loss", "unreachable"]):
            return "network_failure"
    for svc, m in metrics.items():
        if (m.get("latency_ms") or m.get("latency") or 0.0) > 500:
            return "latency_degradation"
        raw = m.get("raw_kpis", {}) if isinstance(m, dict) else {}
        text = " ".join(str(k).lower() for k in raw.keys())
        if "cpu" in text or "memory" in text:
            return "resource_exhaustion"
    return "dependency_failure" if best_score > 0.4 else "unknown"


def build_rca_features(metrics, logs, traces, system, fault_context):
    """Build RCA helper features from observable telemetry only.

    IMPORTANT REDACTION INVARIANT:
    This function must not use `fault_context`, `fault_instances`,
    `faulty_service`, or `expected_faulty_services` to score/rank services.
    Its output may flow into `service_health` and compressed RCA-agent input.
    Oracle labels belong only in evaluator-side reward code.
    """
    scores = defaultdict(float)
    reasons = defaultdict(list)

    for svc, l in logs.items():
        s = l.get("log_anomaly_score", 0.0)
        if s:
            scores[svc] += s
            reasons[svc].append(f"log_anomaly={s:.3f}")
        det = l.get("dominant_error_type")
        if det and det != "none":
            reasons[svc].append(f"log_error_type={det}")

    for svc, m in metrics.items():
        latency = m.get("latency_ms") or m.get("latency", 0.0)
        if latency > 500:
            scores[svc] += 0.25
            reasons[svc].append(f"high_latency_ms={latency:.3f}")
        if m.get("restarts", 0.0) > 0:
            scores[svc] += 0.2
            reasons[svc].append("container_restarts_metric")

    for svc, sy in system.items():
        if sy.get("infra_issue_flag"):
            scores[svc] += 0.5
            reasons[svc].append(f"infra_issue={sy.get('service_health_status')}")
        if sy.get("pods_unready", 0) > 0:
            scores[svc] += 0.25
            reasons[svc].append(f"pods_unready={sy.get('pods_unready')}")
        if sy.get("restart_count", 0) > 0:
            scores[svc] += 0.15
            reasons[svc].append(f"restart_count={sy.get('restart_count')}")

    best_edge = None
    best_score = -1.0
    for edge, t in traces.items():
        if "->" not in edge:
            continue
        src, dst = edge.split("->", 1)
        es = t.get("edge_rank_score", 0.0)
        er = t.get("error_ratio", 0.0)
        if es > 0:
            scores[dst] += es
        if es > 0.2:
            reasons[dst].append(f"trace_edge_issue={edge}")
            reasons[dst].append(f"trace_failure_type={t.get('failure_type')}")
        if src == dst and er > 0.3:
            scores[dst] += 0.4
            reasons[dst].append("self_loop_failure")
        if es > best_score:
            best_score = es
            best_edge = edge

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [
        {"service": svc, "score": round(sc, 4), "reasons": reasons[svc][:10]}
        for svc, sc in ranked[:8]
    ]

    blind = []
    if not any(l.get("line_count", 0) for l in logs.values()):
        blind.append("log_pipeline_empty")
    if not any(m.get("metric_signal_present") for m in metrics.values()):
        blind.append("resource_metric_pipeline_empty")
    if not traces:
        blind.append("trace_pipeline_empty")
    if not system:
        blind.append("system_pipeline_empty")

    conf = max(0.2, min(0.95, 0.85 - 0.12 * len(blind)))
    observable_fault_type = _observable_error_family(best_score, logs, metrics, traces, system)
    primary_hypothesis = {
        "service": candidates[0]["service"] if candidates else None,
        "fault_type": observable_fault_type,
        "confidence": round(conf, 3),
        "source": "observable_telemetry_only",
    }

    return {
        "candidate_root_causes": candidates,
        "most_suspicious_service": candidates[0]["service"] if candidates else None,
        "most_suspicious_edge": best_edge,
        "dominant_error_family": observable_fault_type,
        # Keep legacy keys for downstream compatibility, but never populate them
        # from oracle/injected fault context.
        "known_fault_hypotheses": [],
        "is_multifault": False,
        "expected_faulty_services": [],
        "observability_gaps": blind,
        "confidence": round(conf, 3),
        "hypothesis": primary_hypothesis,
        "hypotheses": [primary_hypothesis] if primary_hypothesis["service"] else [],
        "observability": {
            "logs_available": "log_pipeline_empty" not in blind,
            "resource_metrics_available": "resource_metric_pipeline_empty" not in blind,
            "trace_available": "trace_pipeline_empty" not in blind,
            "system_available": "system_pipeline_empty" not in blind,
            "blind_spots": blind,
        },
    }


def infer_service_health(services, system, logs, traces, rca):
    """Infer service health from observable telemetry only.

    Do not use rca.expected_faulty_services or other oracle-derived RCA fields.
    """
    out = {}
    suspicious = {c["service"] for c in rca.get("candidate_root_causes", [])[:3] if c.get("service")}

    for svc in services:
        status = "healthy"
        reason = []
        sy_status = system.get(svc, {}).get("service_health_status")
        if system.get(svc, {}).get("infra_issue_flag") and sy_status not in ["healthy", "unknown", None]:
            status = "infra_degraded"
            reason.append(sy_status)
        if system.get(svc, {}).get("pods_unready", 0) > 0:
            status = "infra_degraded"
            reason.append(f"pods_unready={system.get(svc, {}).get('pods_unready')}")
        if logs.get(svc, {}).get("log_anomaly_score", 0.0) > 0.4:
            status = "app_degraded"
            reason.append("log_anomaly")
        for edge, feats in traces.items():
            if edge.endswith("->" + svc) and feats.get("is_suspicious"):
                status = "dependency_degraded"
                reason.append(f"incoming_trace_issue={edge}")
        if svc in suspicious and status == "healthy":
            status = "observable_suspect"
            reason.append("ranked_by_observable_telemetry")
        out[svc] = {"status": status, "reasons": reason[:8]}
    return out
