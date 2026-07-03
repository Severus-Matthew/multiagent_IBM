"""
agent_input_builder.py

Builds structured, LLM-ready inputs for the three agents in the pipeline:

  1. RCA Agent (fixed LLM — Claude/GPT)
     Input: state abstraction → structured context + formatted prompt
     Output: {root_cause_service, fault_type, explanation, affected_services, confidence}

  2. Prompt Optimization Agent (trainable — Qwen2.5-Coder-7B)
     Input: RCA result + state + reward history from prior iterations
     Output: a detailed instruction prompt for the Action Agent

  3. Action Agent (fixed LLM — Claude/GPT)
     Input: instruction prompt from Prompt Optimization Agent
     Output: kubectl/helm commands (one per line)

Usage:
    from agent_input_builder import build_rca_agent_input, build_prompt_optimizer_input, build_action_agent_input

    # After run_pipeline.py:
    state      = read_json("processed_state/state_abstraction.json")
    compressed = read_json("processed_state/state_abstraction_compressed.json")

    rca_input = build_rca_agent_input(state, compressed)
    # → rca_input["prompt"] is the LLM prompt string
    # → rca_input["structured_input"] is the raw data dict

    # After RCA agent responds:
    rca_result = {"root_cause_service": "geo", "fault_type": "auth_failure", ...}
    opt_input = build_prompt_optimizer_input(state, compressed, rca_result, reward_history=[], iteration=0)

    # After Prompt Optimization Agent produces instruction_prompt:
    action_prompt = build_action_agent_input(instruction_prompt, opt_input)
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# 1. RCA Agent input
# ─────────────────────────────────────────────────────────────────────────────

def build_rca_agent_input(state: dict, compressed: dict | None = None) -> dict:
    """
    Build the full input package for the RCA Agent.

    Returns:
        {
            "structured_input": dict,   # machine-readable context
            "prompt": str,              # formatted LLM prompt
        }
    """
    fault_ctx      = state.get("fault_context", {})
    service_health = state.get("service_health", {})
    rca_features   = state.get("rca_features", {})
    sla            = state.get("sla", {})
    traces         = state.get("traces", {})

    llm_view      = (compressed or {}).get("llm_view", {})
    trace_summary = (compressed or {}).get("traces", {}).get("summary", {})
    clusters      = (compressed or {}).get("clusters", {})

    # Top suspicious trace edges (for context)
    suspicious_edges = [
        {"edge": edge, "error_ratio": feats.get("error_ratio", 0),
         "failure_type": feats.get("failure_type")}
        for edge, feats in traces.items()
        if feats.get("is_suspicious")
    ]
    suspicious_edges.sort(key=lambda x: x["error_ratio"], reverse=True)

    structured = {
        "scenario_id": state.get("scenario_id"),
        "namespace": fault_ctx.get("target_namespace"),
        "task": fault_ctx.get("task"),
        "all_services": state.get("services", []),

        "scenario_description": fault_ctx.get("problem_description", "")[:2000],

        "service_health": {
            svc: {"status": h.get("status"), "reasons": h.get("reasons", [])}
            for svc, h in service_health.items()
        },

        "top_error_services": llm_view.get("top_log_error_services", []),

        "suspicious_trace_edges": suspicious_edges[:10],

        "trace_summary": {
            "failed_edges": trace_summary.get("failed_edges", []),
            "slow_edges": trace_summary.get("slow_edges", []),
            "top_edges_by_score": trace_summary.get("top_edges_by_score", [])[:5],
        },

        "service_clusters": clusters,

        "sla": {
            "violated": sla.get("violated", False),
            "unhealthy_services": [
                svc for svc, v in sla.get("per_service", {}).items()
                if not v.get("healthy")
            ][:10],
            "violated_edges": sla.get("dependency_sla", {}).get("violations", [])[:5],
        },

        # Algorithmic hints — not the answer, but signal the agent can confirm/reject
        "algorithmic_hints": {
            "most_suspicious_service": rca_features.get("most_suspicious_service"),
            "dominant_error_family":   rca_features.get("dominant_error_family"),
            "top_candidates": [
                {"service": c["service"], "score": c["score"], "reasons": c["reasons"][:3]}
                for c in rca_features.get("candidate_root_causes", [])[:5]
            ],
            "observability_gaps": rca_features.get("observability_gaps", []),
            "confidence": rca_features.get("confidence"),
        },

        # Ground truth — NOT passed to the agent prompt; used for offline evaluation only
        "_ground_truth": state.get("ground_truth", {}),
    }

    return {
        "structured_input": structured,
        "prompt": _format_rca_prompt(structured),
    }


def _format_rca_prompt(s: dict) -> str:
    lines = [
        "# Root Cause Analysis Task",
        f"Scenario : {s['scenario_id']}",
        f"Namespace: {s['namespace']}",
        f"Task type: {s['task']}",
        "",
    ]

    desc = s["scenario_description"].strip()
    if desc:
        lines += ["## Problem Description", desc, ""]

    lines.append("## Service Health")
    degraded = {svc: h for svc, h in s["service_health"].items()
                if h.get("status", "healthy") != "healthy"}
    if degraded:
        for svc, h in degraded.items():
            r = ", ".join(h.get("reasons", []))
            lines.append(f"  - {svc}: {h['status']}" + (f"  ({r})" if r else ""))
    else:
        lines.append("  All services appear healthy by pod-level signals.")
    lines.append("")

    top_err = s["top_error_services"][:6]
    if top_err:
        lines.append("## Top Error Signals (from logs)")
        for item in top_err:
            svc = item.get("service")
            cnt = item.get("error_count", 0)
            dom = item.get("dominant_error_type")
            lines.append(f"  - {svc}: {cnt} errors  (dominant: {dom})")
            for ev in item.get("evidence", [])[:2]:
                lines.append(f"      └ {str(ev)[:180]}")
        lines.append("")

    susp = s["suspicious_trace_edges"][:6]
    if susp:
        lines.append("## Suspicious Trace Edges")
        for e in susp:
            lines.append(f"  - {e['edge']}  error_ratio={e['error_ratio']:.2f}  ({e['failure_type']})")
        lines.append("")

    clusters = s["service_clusters"]
    if clusters:
        lines.append("## Service Behaviour Clusters")
        for bucket, svcs in clusters.items():
            if svcs:
                lines.append(f"  {bucket}: {', '.join(svcs)}")
        lines.append("")

    sla = s["sla"]
    if sla.get("violated"):
        lines.append("## SLA Violations")
        for svc in sla.get("unhealthy_services", []):
            lines.append(f"  - {svc}")
        lines.append("")

    hints = s["algorithmic_hints"]
    if hints.get("most_suspicious_service"):
        lines += [
            "## Algorithmic Pre-screening (verify or refute these)",
            f"  Most suspicious service: {hints['most_suspicious_service']}",
            f"  Dominant error family  : {hints['dominant_error_family']}",
            f"  Pre-screening confidence: {hints['confidence']}",
        ]
        if hints["observability_gaps"]:
            lines.append(f"  Observability gaps: {', '.join(hints['observability_gaps'])}")
        lines.append("")

    lines += [
        "## Your Task",
        "Based on the telemetry above, identify the root cause and respond in this exact JSON format:",
        "",
        "```json",
        "{",
        '  "root_cause_service": "<service_name>",',
        '  "fault_type": "<auth_failure|dependency_failure|infra_failure|latency_degradation|config_error>",',
        '  "explanation": "<concise explanation citing specific evidence>",',
        '  "affected_services": ["<svc1>", "<svc2>"],',
        '  "confidence": <0.0-1.0>',
        "}",
        "```",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prompt Optimization Agent input
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_optimizer_input(
    state: dict,
    compressed: dict | None,
    rca_result: dict,
    reward_history: list | None = None,
    iteration: int = 0,
) -> dict:
    """
    Build the input for the Prompt Optimization Agent (Qwen2.5-Coder-7B).

    The agent must output a detailed instruction prompt for the Action Agent
    that will cause it to generate correct kubectl/helm fix commands.

    Args:
        state:          full state_abstraction dict
        compressed:     compressed state (for clusters / llm_view)
        rca_result:     output from the RCA Agent
        reward_history: list of {"iteration": int, "commands": list, "result": str,
                                  "reward": float, "failure_reason": str}
        iteration:      current iteration index (0-based)

    Returns a structured dict that becomes the Prompt Optimization Agent's context.
    """
    fault_ctx = state.get("fault_context", {})
    system    = state.get("system", {})

    faulty_svc = rca_result.get("root_cause_service") or fault_ctx.get("faulty_service", "")
    fault_type = rca_result.get("fault_type") or fault_ctx.get("fault_family", "")
    namespace  = fault_ctx.get("target_namespace", "")
    affected   = rca_result.get("affected_services", [])

    # Pod details for affected + faulty service
    relevant_svcs = list({faulty_svc} | set(affected))
    pod_details: dict[str, dict] = {}
    for svc in relevant_svcs:
        if not svc:
            continue
        sys_info = system.get(svc, {})
        pod_details[svc] = {
            "pods": sys_info.get("pod_names", []),
            "health_status": sys_info.get("service_health_status", "unknown"),
            "restart_count": sys_info.get("restart_count", 0),
            "crashloop_count": sys_info.get("crashloop_count", 0),
            "deployment": (sys_info.get("deployment") or {}).get("deployment_name"),
        }

    return {
        "scenario_id": state.get("scenario_id"),
        "namespace": namespace,
        "iteration": iteration,
        "max_iterations": 7,

        "rca_result": rca_result,

        "fault_summary": {
            "faulty_service": faulty_svc,
            "fault_type": fault_type,
            "explanation": rca_result.get("explanation", ""),
            "affected_services": affected,
        },

        "pod_details": pod_details,
        "action_library": _build_action_library(faulty_svc, fault_type, namespace),

        "previous_attempts": [
            {
                "iteration":      a.get("iteration"),
                "commands":       a.get("commands", []),
                "result":         a.get("result"),
                "reward":         a.get("reward"),
                "failure_reason": a.get("failure_reason"),
            }
            for a in (reward_history or [])
        ],

        "task_instruction": (
            "Generate a detailed instruction prompt for the Action Agent. "
            "The prompt must tell the Action Agent exactly how to fix the fault, "
            "which commands to run, in what order, and what success looks like. "
            "Be specific: include service names, namespace, pod names where relevant. "
            "The Action Agent will only output kubectl/helm commands — no prose."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action Agent input
# ─────────────────────────────────────────────────────────────────────────────

def build_action_agent_input(instruction_prompt: str, context: dict) -> str:
    """
    Build the final prompt string for the Action Agent (fixed LLM).

    instruction_prompt is the output of the Prompt Optimization Agent.
    context is the dict returned by build_prompt_optimizer_input.

    Returns a complete prompt string ready to send to the Action Agent.
    The Action Agent should respond with kubectl/helm commands only.
    """
    namespace = context.get("namespace", "")
    scenario  = context.get("scenario_id", "")
    iteration = context.get("iteration", 0)

    header = (
        f"# Kubernetes Fault Remediation\n"
        f"Scenario : {scenario}  (iteration {iteration + 1} / {context.get('max_iterations', 7)})\n"
        f"Namespace: {namespace}\n"
        f"kubectl context: kind-kind\n"
        "\n"
        "---\n"
        "\n"
        f"{instruction_prompt}\n"
        "\n"
        "---\n"
        "\n"
        "Output ONLY the commands to execute, one per line, in exact execution order.\n"
        "Use real kubectl / helm syntax. No explanations, no markdown, no comments.\n"
        "Example output format:\n"
        "  kubectl rollout restart deployment/geo -n test-hotel-reservation\n"
        "  kubectl rollout status deployment/geo -n test-hotel-reservation\n"
    )

    return header


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: action library
# ─────────────────────────────────────────────────────────────────────────────

def _build_action_library(faulty_svc: str, fault_type: str, namespace: str) -> list[dict]:
    """
    Return a list of available actions, pre-filled with known context where possible.
    Used by the Prompt Optimization Agent to understand what the Action Agent can do.
    """
    ns = namespace or "<namespace>"
    svc = faulty_svc or "<service>"

    actions = [
        {
            "template": f"kubectl rollout restart deployment/{svc} -n {ns}",
            "effect": "Restart all pods of a deployment (triggers re-init, re-auth, config reload)",
            "risk": "medium",
            "use_for": ["transient_errors", "auth_failure", "config_error", "infra_failure"],
        },
        {
            "template": f"kubectl rollout status deployment/{svc} -n {ns} --timeout=120s",
            "effect": "Wait for rollout to complete and verify pods are ready",
            "risk": "none",
            "use_for": ["verification"],
        },
        {
            "template": f"kubectl delete pod <pod-name> -n {ns} --grace-period=0 --force",
            "effect": "Force-delete a stuck / crashlooping pod (it will be recreated)",
            "risk": "low",
            "use_for": ["crashloop", "stuck_pod"],
        },
        {
            "template": f"kubectl scale deployment/{svc} --replicas=0 -n {ns}",
            "effect": "Scale down to zero (useful before config fix to stop restart loop)",
            "risk": "high",
            "use_for": ["stop_crashloop_before_fix"],
        },
        {
            "template": f"kubectl scale deployment/{svc} --replicas=1 -n {ns}",
            "effect": "Scale back up after fix",
            "risk": "medium",
            "use_for": ["restore_after_fix"],
        },
        {
            "template": f"kubectl patch configmap <name> -n {ns} "
                        "--type=merge -p '{\"data\":{\"<key>\":\"<value>\"}}'",
            "effect": "Update a ConfigMap value in-place",
            "risk": "medium",
            "use_for": ["config_error", "misconfiguration"],
        },
        {
            "template": f"helm rollback <release> <revision> -n {ns}",
            "effect": "Roll back a Helm release to a previous working revision",
            "risk": "medium",
            "use_for": ["bad_deployment", "regression"],
        },
        {
            "template": f"kubectl exec -it <pod> -n {ns} -- <command>",
            "effect": "Execute a command inside a running pod (e.g. mongosh for auth fix)",
            "risk": "low",
            "use_for": ["data_fix", "auth_repair", "diagnostic"],
        },
        {
            "template": f"kubectl get pod -n {ns} -o wide",
            "effect": "Check current pod status (use for verification)",
            "risk": "none",
            "use_for": ["verification", "diagnostic"],
        },
    ]

    # Add auth-specific action when fault is auth-related
    if "auth" in (fault_type or "").lower():
        actions.append({
            "template": (
                f"kubectl exec -it <mongodb-pod> -n {ns} -- "
                "mongosh admin --eval "
                "\"db.updateUser('<user>', {pwd: '<password>'})\" "
            ),
            "effect": "Reset MongoDB user password via mongosh admin",
            "risk": "high",
            "use_for": ["auth_failure", "mongodb_auth_broken"],
        })
        actions.append({
            "template": (
                f"kubectl delete pvc --all -n {ns} --ignore-not-found && "
                f"kubectl rollout restart deployment/{svc} -n {ns}"
            ),
            "effect": "Wipe corrupted PVC (MongoDB auth data) and restart — use as last resort",
            "risk": "very_high",
            "use_for": ["persistent_auth_failure", "corrupted_pvc"],
        })

    return actions
