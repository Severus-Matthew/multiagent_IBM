from collections import defaultdict


def infer_fault_type(fault_context):
    fam = str(fault_context.get("fault_family") or "").lower()
    if fam == "multifault":
        return "multifault"
    if "auth" in fam and "mongo" in fam:
        return "auth_failure"
    if "mongo" in fam:
        return "dependency_failure"
    if "latency" in fam or "delay" in fam:
        return "latency_degradation"
    if "assign_non_existent_node" in fam or "non_existent_node" in fam:
        return "infra_failure"
    if "scale_pod" in fam or "pod_failure" in fam or "pod_kill" in fam or "container_kill" in fam:
        return "infra_failure"
    if "cpu" in fam or "memory" in fam or "pod" in fam:
        return "infra_failure"
    if "config" in fam or "misconfig" in fam:
        return "config_error"
    return "unknown"


def infer_instance_fault_type(instance):
    return infer_fault_type({"fault_family": instance.get("fault_family")})


def build_rca_features(metrics, logs, traces, system, fault_context):
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
    # Use generated fault context as weak evidence. For multifault scenarios,
    # keep every generated fault instance instead of collapsing to one service.
    fault_instances = fault_context.get("fault_instances") or []
    if not fault_instances and fault_context.get("faulty_service"):
        fault_instances = [{
            "faulty_service": fault_context.get("faulty_service"),
            "fault_family": fault_context.get("fault_family"),
        }]

    known_hypotheses = []
    for inst in fault_instances:
        svc = inst.get("faulty_service")
        ft = infer_instance_fault_type(inst)
        if not svc:
            continue
        direct_system = system.get(svc, {})
        direct_status = direct_system.get("service_health_status")
        direct_infra = direct_status not in (None, "healthy", "unknown")
        # Generated fault context is weak by itself, but strong when the same
        # service has direct K8s evidence such as no ready endpoints/pending pods.
        scores[svc] += 1.25 if direct_infra else 0.35
        reasons[svc].append(f"scenario_fault_context={ft}")
        if direct_infra:
            reasons[svc].append(f"direct_system_status={direct_status}")
        if inst.get("variant_name") and inst.get("variant_name") != "default":
            reasons[svc].append(f"scenario_variant={inst.get('variant_name')}")
        known_hypotheses.append({
            "service": svc,
            "fault_type": ft,
            "fault_family": inst.get("fault_family"),
            "variant_name": inst.get("variant_name", "default"),
            "variant_params": inst.get("variant_params", {}),
            "source": "generated_fault_context",
        })

    known_fault_service = (
        known_hypotheses[0]["service"] if known_hypotheses
        else fault_context.get("faulty_service")
    )
    known_fault_type = infer_fault_type(fault_context)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [{"service": svc, "score": round(sc, 4), "reasons": reasons[svc][:10]} for svc, sc in ranked[:8]]
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
    primary_hypothesis = {
        "service": candidates[0]["service"] if candidates else known_fault_service,
        "fault_type": (
            known_hypotheses[0]["fault_type"]
            if known_hypotheses
            else known_fault_type
        ),
        "confidence": round(conf, 3),
    }

    return {
        "candidate_root_causes": candidates,
        "most_suspicious_service": candidates[0]["service"] if candidates else known_fault_service,
        "most_suspicious_edge": best_edge,
        "dominant_error_family": known_fault_type if known_fault_type != "unknown" else "dependency_failure" if best_score > 0.4 else "unknown",
        "known_fault_hypotheses": known_hypotheses,
        "is_multifault": bool(fault_context.get("is_multifault")),
        "expected_faulty_services": fault_context.get("expected_faulty_services", []),
        "observability_gaps": blind,
        "confidence": round(conf, 3),
        "hypothesis": primary_hypothesis,
        "hypotheses": known_hypotheses or [primary_hypothesis],
        "observability": {
            "logs_available": "log_pipeline_empty" not in blind,
            "resource_metrics_available": "resource_metric_pipeline_empty" not in blind,
            "trace_available": "trace_pipeline_empty" not in blind,
            "system_available": "system_pipeline_empty" not in blind,
            "blind_spots": blind,
        },
    }


def infer_service_health(services, system, logs, traces, rca):
    out = {}
    suspicious = {c["service"] for c in rca.get("candidate_root_causes", [])[:3]}
    suspicious |= set(rca.get("expected_faulty_services", []))
    for svc in services:
        status = "healthy"
        reason = []
        sy_status = system.get(svc, {}).get("service_health_status")
        if system.get(svc, {}).get("infra_issue_flag") and sy_status not in ["healthy", "unknown", None]:
            status = "infra_degraded"
            reason.append(sy_status)
        if logs.get(svc, {}).get("log_anomaly_score", 0.0) > 0.4:
            status = "app_degraded"
            reason.append("log_anomaly")
        for edge, feats in traces.items():
            if edge.endswith("->" + svc) and feats.get("is_suspicious"):
                status = "dependency_degraded"
                reason.append(f"incoming_trace_issue={edge}")
        if svc in suspicious and status == "healthy":
            status = "suspect_silent_failure"
            reason.append("ranked_by_rca_context_or_weak_signals")
        out[svc] = {"status": status, "reasons": reason[:8]}
    return out
