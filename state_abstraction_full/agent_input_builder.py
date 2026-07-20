"""
Builds structured inputs for the RCA, prompt optimizer, and action agent.

Important privacy rule for training/evaluation:
  RCA must consume only the redacted compressed state. It must not receive
  ground_truth, fault_context.raw_spec, faulty_service, fault_instances, or
  algorithmic RCA hints derived from generated labels.
"""

from __future__ import annotations

import json


def _as_redacted_state(state: dict, compressed: dict | None = None) -> dict:
    """Return the redacted compressed state regardless of old/new call style."""
    if isinstance(compressed, dict) and compressed:
        return compressed
    if isinstance(state, dict) and state.get("state_type", "").startswith("redacted_compressed"):
        return state
    # Backward-compatible emergency fallback. This is not ideal for final RCA,
    # but it avoids crashes when called before compression in debugging.
    return {
        "scenario_id": state.get("scenario_id"),
        "namespace": (state.get("fault_context", {}) or {}).get("target_namespace"),
        "task": (state.get("fault_context", {}) or {}).get("task"),
        "services": state.get("services", []),
        "metrics": state.get("metrics", {}),
        "logs": state.get("logs", {}),
        "traces": state.get("traces", {}),
        "system": state.get("system", {}),
        "workload": state.get("workload", {}),
        "graph": state.get("graph", {}),
        "sla": state.get("sla", {}),
        "service_health": state.get("service_health", {}),
        "observability_metadata": state.get("observability_metadata", {}),
        "clusters": {},
        "llm_view": {},
        "redaction": {"fallback_from_full_state": True, "ground_truth_removed": True, "fault_context_removed": True},
    }


def _top_suspicious_edges(redacted: dict, limit: int = 10) -> list[dict]:
    traces = redacted.get("traces", {})
    per_edge = traces.get("per_edge", traces if isinstance(traces, dict) else {})
    out = []
    for edge, feats in per_edge.items():
        if not isinstance(feats, dict):
            continue
        if feats.get("is_suspicious") or feats.get("error_ratio", 0) > 0 or feats.get("edge_rank_score", 0) > 0.1:
            out.append({
                "edge": edge,
                "error_ratio": feats.get("error_ratio", 0),
                "latency_p95_ms": feats.get("latency_p95_ms", feats.get("latency_p95_us", 0) / 1000.0),
                "failure_type": feats.get("failure_type", "unknown"),
                "edge_rank_score": feats.get("edge_rank_score", 0),
            })
    out.sort(key=lambda x: (x.get("edge_rank_score", 0), x.get("error_ratio", 0), x.get("latency_p95_ms", 0)), reverse=True)
    return out[:limit]


def _degraded_services(redacted: dict) -> dict:
    service_health = redacted.get("service_health", {}) or {}
    out = {}
    for svc, h in service_health.items():
        if isinstance(h, dict) and h.get("status", "healthy") != "healthy":
            out[svc] = {"status": h.get("status"), "reasons": h.get("reasons", [])}
    system = redacted.get("system", {}) or {}
    for svc, s in system.items():
        health = (s.get("health", {}) if isinstance(s, dict) else {}) or {}
        if health.get("infra_issue_flag") or health.get("pods_unready", 0) > 0 or health.get("crashloop_count", 0) > 0:
            out.setdefault(svc, {"status": health.get("status", "infra_degraded"), "reasons": []})
            out[svc]["reasons"] = list(set(out[svc].get("reasons", []) + [health.get("status", "infra_issue")]))
    return out


def build_rca_agent_input(state: dict, compressed: dict | None = None) -> dict:
    redacted = _as_redacted_state(state, compressed)
    llm_view = redacted.get("llm_view", {}) or {}
    traces = redacted.get("traces", {}) or {}
    trace_summary = traces.get("summary", {}) if isinstance(traces, dict) else {}
    structured = {
        "scenario_id": redacted.get("scenario_id"),
        "namespace": redacted.get("namespace"),
        "task": redacted.get("task"),
        "all_services": redacted.get("services", []),
        "service_health": _degraded_services(redacted),
        "top_error_services": llm_view.get("top_log_error_services", []),
        "suspicious_trace_edges": _top_suspicious_edges(redacted, limit=12),
        "trace_summary": {
            "failed_edges": trace_summary.get("failed_edges", [])[:20],
            "slow_edges": trace_summary.get("slow_edges", [])[:20],
            "top_edges_by_score": trace_summary.get("top_edges_by_score", [])[:10],
        },
        "service_clusters": redacted.get("clusters", {}),
        "sla": redacted.get("sla", {}),
        "observability_metadata": redacted.get("observability_metadata", {}),
        "redaction": redacted.get("redaction", {}),
    }
    return {"structured_input": structured, "prompt": _format_rca_prompt(structured)}


def _format_rca_prompt(s: dict) -> str:
    lines = [
        "# Root Cause Analysis Task",
        f"Scenario : {s.get('scenario_id')}",
        f"Namespace: {s.get('namespace')}",
        f"Task type: {s.get('task')}",
        "",
        "You are given REDACTED telemetry only. Ground truth and generated fault labels have been removed.",
        "Infer the root cause from service health, logs, traces, metrics, and SLA evidence.",
        "",
    ]
    degraded = s.get("service_health", {})
    lines.append("## Degraded / Suspicious Services")
    if degraded:
        for svc, h in degraded.items():
            lines.append(f"  - {svc}: {h.get('status')} ({', '.join(map(str, h.get('reasons', [])))})")
    else:
        lines.append("  No directly degraded Kubernetes service-health signal was found.")
    lines.append("")

    top_err = s.get("top_error_services", [])[:8]
    if top_err:
        lines.append("## Top Log Error Signals")
        for item in top_err:
            lines.append(f"  - {item.get('service')}: errors={item.get('error_count', 0)} dominant={item.get('dominant_error_type')}")
            for ev in item.get("evidence", [])[:2]:
                lines.append(f"      evidence: {str(ev)[:220]}")
        lines.append("")

    susp = s.get("suspicious_trace_edges", [])[:8]
    if susp:
        lines.append("## Suspicious Trace Edges")
        for e in susp:
            lines.append(f"  - {e.get('edge')} error_ratio={e.get('error_ratio', 0):.3f} p95_ms={e.get('latency_p95_ms', 0):.2f} type={e.get('failure_type')}")
        lines.append("")

    clusters = s.get("service_clusters", {})
    if clusters:
        lines.append("## Service Behaviour Clusters")
        for bucket, svcs in clusters.items():
            if svcs:
                lines.append(f"  {bucket}: {', '.join(svcs)}")
        lines.append("")

    lines += [
        "## Your Task",
        "Return ONLY valid JSON in this exact schema:",
        "```json",
        "{",
        '  "root_cause_service": "<service_name>",',
        '  "fault_type": "<auth_failure|dependency_failure|infra_failure|latency_degradation|config_error|pod_failure|network_failure|resource_exhaustion>",',
        '  "explanation": "<concise explanation citing specific evidence>",',
        '  "affected_services": ["<svc1>", "<svc2>"],',
        '  "digital_twin_injection_spec": {',
        '    "target_service": "<service_name>",',
        '    "fault_type": "<normalized_fault_type>",',
        '    "expected_symptoms": ["<symptom1>", "<symptom2>"]',
        "  },",
        '  "confidence": <0.0-1.0>',
        "}",
        "```",
    ]
    return "\n".join(lines)


def build_prompt_optimizer_input(state: dict, compressed: dict | None, rca_result: dict | None = None, reward_history: list | None = None, iteration: int = 0) -> dict:
    # Backward-compatible signature: (state, compressed, rca_result, ...)
    redacted = _as_redacted_state(state, compressed)
    rca_result = rca_result or {}
    faulty_svc = rca_result.get("root_cause_service") or rca_result.get("digital_twin_injection_spec", {}).get("target_service") or ""
    fault_type = rca_result.get("fault_type") or rca_result.get("digital_twin_injection_spec", {}).get("fault_type") or ""
    namespace = redacted.get("namespace", "")
    affected = rca_result.get("affected_services", []) or []
    system = redacted.get("system", {}) or {}
    relevant_svcs = list({faulty_svc} | set(affected))
    pod_details = {}
    for svc in relevant_svcs:
        if not svc:
            continue
        sys_info = system.get(svc, {}) or {}
        health = sys_info.get("health", sys_info)
        deploy = sys_info.get("deployment", {}) if isinstance(sys_info.get("deployment", {}), dict) else {}
        pod_details[svc] = {
            "pods": sys_info.get("pods", sys_info.get("pod_names", [])),
            "health_status": health.get("status", sys_info.get("service_health_status", "unknown")),
            "restart_count": health.get("restart_count", sys_info.get("restart_count", 0)),
            "crashloop_count": health.get("crashloop_count", sys_info.get("crashloop_count", 0)),
            "deployment": deploy.get("deployment_name"),
        }
    return {
        "scenario_id": redacted.get("scenario_id"),
        "namespace": namespace,
        "iteration": iteration,
        "max_iterations": 7,
        "rca_result": rca_result,
        "fault_summary": {"faulty_service": faulty_svc, "fault_type": fault_type, "explanation": rca_result.get("explanation", ""), "affected_services": affected},
        "pod_details": pod_details,
        "action_library": _build_action_library(faulty_svc, fault_type, namespace),
        "previous_attempts": [{"iteration": a.get("iteration"), "commands": a.get("commands", []), "result": a.get("result"), "reward": a.get("reward"), "failure_reason": a.get("failure_reason")} for a in (reward_history or [])],
        "task_instruction": "Generate a detailed instruction prompt for the Action Agent. The Action Agent must output only kubectl/helm commands in order. Include service names, namespace, pod/deployment details, verification commands, and a different strategy if previous attempts failed.",
    }


def build_action_agent_input(instruction_prompt: str, context: dict) -> str:
    namespace = context.get("namespace", "")
    scenario = context.get("scenario_id", "")
    iteration = context.get("iteration", 0)
    return (
        f"# Kubernetes Fault Remediation\n"
        f"Scenario : {scenario} (iteration {iteration + 1} / {context.get('max_iterations', 7)})\n"
        f"Namespace: {namespace}\n"
        f"kubectl context: kind-kind\n\n---\n\n"
        f"{instruction_prompt}\n\n---\n\n"
        "Output ONLY executable commands, one per line, in exact execution order. "
        "Use real kubectl/helm syntax. No explanations, no markdown, no comments."
    )


def _build_action_library(faulty_svc: str, fault_type: str, namespace: str) -> list[dict]:
    ns = namespace or "<namespace>"
    svc = faulty_svc or "<service>"
    actions = [
        {"template": f"kubectl rollout restart deployment/{svc} -n {ns}", "effect": "restart deployment pods", "risk": "medium", "use_for": ["transient_errors", "config_error", "infra_failure"]},
        {"template": f"kubectl rollout status deployment/{svc} -n {ns} --timeout=120s", "effect": "verify rollout", "risk": "none", "use_for": ["verification"]},
        {"template": f"kubectl delete pod <pod-name> -n {ns} --grace-period=0 --force", "effect": "recreate stuck pod", "risk": "low", "use_for": ["crashloop", "stuck_pod"]},
        {"template": f"kubectl scale deployment/{svc} --replicas=<n> -n {ns}", "effect": "change replica count", "risk": "medium", "use_for": ["overload", "pod_failure"]},
        {"template": f"kubectl patch configmap <name> -n {ns} --type=merge -p '{{\"data\":{{\"<key>\":\"<value>\"}}}}'", "effect": "fix config", "risk": "medium", "use_for": ["config_error", "misconfiguration"]},
        {"template": f"helm rollback <release> <revision> -n {ns}", "effect": "rollback release", "risk": "medium", "use_for": ["bad_deployment", "regression"]},
        {"template": f"kubectl get pod -n {ns} -o wide", "effect": "verify pod status", "risk": "none", "use_for": ["verification", "diagnostic"]},
    ]
    if "auth" in (fault_type or "").lower():
        actions.append({"template": f"kubectl exec -it <mongodb-pod> -n {ns} -- mongosh admin --eval \"db.updateUser('<user>', {{pwd: '<password>'}})\"", "effect": "repair MongoDB credentials", "risk": "high", "use_for": ["auth_failure"]})
    return actions
