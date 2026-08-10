from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


CANONICAL_FAULT_TYPES = [
    "infra_failure",
    "auth_failure",
    "dependency_failure",
    "resource_exhaustion",
    "latency_degradation",
    "network_failure",
    "config_error",
    "unknown",
]


@dataclass
class RCAPromptPlan:
    """Structured prompt plan emitted by the non-LM controller.

    In legacy/audit mode this can include focus services and fault-type bias.  In
    training-safe mode those fields are suppressed when rendered so the trainable
    model is not handed a root-cause menu.
    """

    evidence_priority: list[str]
    focus_services: list[str]
    root_cause_count_hint: int
    fault_type_bias: list[str]
    operators: list[str]
    retry_strategy: str
    safe_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperatorRCAInstructionPolicy:
    """Structured verifier-guided prompt-operator controller for RCA.

    The controller uses only redacted/compressed state and previous non-leaking
    feedback.  With safe_mode=True, the rendered instruction hides focus-service
    lists, root-cause-count hints, and canonical fault-type menus, making it more
    appropriate for transportable training/evaluation.
    """

    def __init__(self, profile: str = "auto", max_focus_services: int = 6, safe_mode: bool = False):
        self.profile = profile
        self.max_focus_services = max(1, int(max_focus_services))
        self.safe_mode = bool(safe_mode)

    def generate_instruction(
        self,
        compressed_state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int = 0,
        group_id: str | None = None,
    ) -> str:
        plan = self._build_plan(compressed_state, history, iteration, sample_index)
        self.last_policy_info = {"prompt_plan": plan.to_dict(), "safe_mode": self.safe_mode}
        return render_rca_prompt_plan(plan)

    def _build_plan(
        self,
        state: dict[str, Any],
        history: list[dict[str, Any]],
        iteration: int,
        sample_index: int,
    ) -> RCAPromptPlan:
        sig = _observable_signature(state)
        focus_services = _rank_focus_services(sig, self.max_focus_services)
        failed_edge_count = len(sig["failed_edges"])
        log_service_count = len(sig["top_error_services"])
        degraded_count = len(sig["degraded_services"])

        count_hint = 1
        if (
            (degraded_count >= 2 and log_service_count >= 2)
            or (failed_edge_count >= 3 and len(set(sig["trace_targets"])) >= 2)
            or len(sig["independent_signal_services"]) >= 3
        ):
            count_hint = 2

        profile = self.profile
        if profile == "auto":
            variants = ["system_first", "trace_first", "log_first", "multifault_first"]
            profile = variants[sample_index % len(variants)]

        if profile == "system_first":
            evidence_priority = ["system_health", "service_health", "logs", "traces", "metrics"]
            operators = ["ENFORCE_OUTPUT_SCHEMA", "PRIORITIZE_SYSTEM_HEALTH", "USE_SMALLEST_EXPLANATORY_SET", "AVOID_DOWNSTREAM_VICTIMS"]
            fault_bias = _fault_bias_from_signature(sig, preferred=["infra_failure", "resource_exhaustion"])
        elif profile == "trace_first":
            evidence_priority = ["traces", "service_graph", "logs", "system_health", "metrics"]
            operators = ["ENFORCE_OUTPUT_SCHEMA", "PRIORITIZE_TRACE_EDGES", "AVOID_DOWNSTREAM_VICTIMS", "USE_SMALLEST_EXPLANATORY_SET"]
            fault_bias = _fault_bias_from_signature(sig, preferred=["network_failure", "latency_degradation", "dependency_failure"])
        elif profile == "log_first":
            evidence_priority = ["logs", "database_dependency_logs", "traces", "system_health", "metrics"]
            operators = ["ENFORCE_OUTPUT_SCHEMA", "PRIORITIZE_LOG_ERRORS", "AVOID_DOWNSTREAM_VICTIMS", "USE_SMALLEST_EXPLANATORY_SET"]
            fault_bias = _fault_bias_from_signature(sig, preferred=["auth_failure", "dependency_failure", "config_error"])
        elif profile == "multifault_first":
            evidence_priority = ["system_health", "logs", "traces", "metrics", "service_graph"]
            operators = ["ENFORCE_OUTPUT_SCHEMA", "CHECK_FOR_INDEPENDENT_SYMPTOM_CLUSTERS", "PRIORITIZE_LOG_ERRORS", "PRIORITIZE_TRACE_EDGES", "AVOID_REPEATED_GUESSES", "AVOID_DOWNSTREAM_VICTIMS"]
            fault_bias = _fault_bias_from_signature(sig, preferred=["config_error", "auth_failure", "network_failure", "latency_degradation"])
            count_hint = max(count_hint, 2)
        else:
            raise ValueError(f"unknown operator profile {self.profile!r}")

        retry_strategy = "none"
        if history:
            retry_strategy = "avoid repeating previous wrong services/fault mechanisms; use only public feedback to shift evidence priority"
            if "AVOID_REPEATED_GUESSES" not in operators:
                operators.append("AVOID_REPEATED_GUESSES")

        return RCAPromptPlan(
            evidence_priority=evidence_priority,
            focus_services=focus_services,
            root_cause_count_hint=count_hint,
            fault_type_bias=fault_bias,
            operators=operators,
            retry_strategy=retry_strategy,
            safe_mode=self.safe_mode,
        )


def render_rca_prompt_plan(plan: RCAPromptPlan) -> str:
    if plan.safe_mode:
        lines = [
            "Read only the redacted telemetry/state abstraction.",
            "Do not assume a fixed candidate menu, injected fault family, or oracle root-cause count.",
            "Infer abnormal components from telemetry evidence such as service health, logs, traces, metrics, and graph symptoms.",
            "Separate upstream root causes from downstream cascade victims.",
            "Use the smallest explanation supported by independent evidence; use multiple lines only when clearly necessary.",
            "Evidence priority: " + ", ".join(plan.evidence_priority),
            "Operators: " + ", ".join(plan.operators),
            "Retry strategy: " + plan.retry_strategy,
            "Output one root cause per line as component::fault_mechanism. No prose, JSON, markdown, or bullets.",
        ]
        return "\n".join(lines)

    lines = [
        "Read only the redacted telemetry. You are given a structured RCA prompt plan produced by a verifier-guided controller.",
        "Your job is to identify the root-cause service and canonical fault type, not downstream victims.",
        "",
        "Structured prompt plan:",
        f"- Evidence priority: {', '.join(plan.evidence_priority)}",
        f"- Focus services: {', '.join(plan.focus_services) if plan.focus_services else 'none specified; infer from telemetry'}",
        f"- Root-cause count hint: {plan.root_cause_count_hint}",
        f"- Fault-type bias: {', '.join(plan.fault_type_bias)}",
        f"- Operators: {', '.join(plan.operators)}",
        f"- Retry strategy: {plan.retry_strategy}",
        "",
        "Reasoning constraints:",
        "- Prefer the smallest root-cause set that explains service health, logs, traces, metrics, and graph symptoms.",
        "- Separate upstream root causes from downstream cascade victims.",
        "- If the prompt plan asks for multifault, output multiple root causes only when independent evidence supports them.",
        "- Do not include explanations, JSON, markdown, bullets, or prose in the final answer.",
        "",
        "Output constraints:",
        "- Output one root cause per line.",
        "- Each line must be exactly: service::fault_type",
        "- Allowed fault types: " + ", ".join(CANONICAL_FAULT_TYPES),
    ]
    return "\n".join(lines)


def _observable_signature(state: dict[str, Any]) -> dict[str, Any]:
    degraded_services: set[str] = set()
    top_error_services: set[str] = set()
    trace_sources: set[str] = set()
    trace_targets: set[str] = set()
    failed_edges: set[str] = set()
    metric_services: set[str] = set()
    independent_signal_services: set[str] = set()

    for svc, info in (state.get("system", {}) or {}).items():
        health = info.get("health", info) if isinstance(info, dict) else {}
        if (
            health.get("infra_issue_flag")
            or float(health.get("pods_unready", 0) or 0) > 0
            or float(health.get("crashloop_count", 0) or 0) > 0
            or str(health.get("status", "")).lower() not in {"", "healthy", "unknown"}
        ):
            degraded_services.add(str(svc)); independent_signal_services.add(str(svc))

    for svc, health in (state.get("service_health", {}) or {}).items():
        if isinstance(health, dict) and str(health.get("status", "healthy")).lower() not in {"healthy", "unknown", ""}:
            degraded_services.add(str(svc)); independent_signal_services.add(str(svc))

    for item in (state.get("llm_view", {}) or {}).get("top_log_error_services", []) or []:
        if isinstance(item, dict) and item.get("service"):
            svc = str(item["service"]); top_error_services.add(svc); independent_signal_services.add(svc)

    traces = state.get("traces", {}) or {}
    per_edge = traces.get("per_edge", traces) if isinstance(traces, dict) else {}
    for edge, feats in per_edge.items():
        if not isinstance(feats, dict):
            continue
        error_ratio = _safe_float(feats.get("error_ratio"))
        if error_ratio > 0.2 or feats.get("is_suspicious"):
            edge_s = str(edge); failed_edges.add(edge_s)
            src = feats.get("source"); dst = feats.get("target")
            if (not src or not dst) and "->" in edge_s:
                src, dst = edge_s.split("->", 1)
            if src:
                trace_sources.add(str(src))
            if dst:
                trace_targets.add(str(dst)); independent_signal_services.add(str(dst))

    for svc, item in (state.get("metrics", {}) or {}).items():
        if isinstance(item, dict):
            flat = item.get("flat_summary", item)
            if isinstance(flat, dict) and _safe_float(flat.get("latency_ms")) > 500:
                metric_services.add(str(svc)); independent_signal_services.add(str(svc))

    clusters = state.get("clusters", {}) or {}
    for key in ("infra_unhealthy", "log_error_or_dependency_failure"):
        for svc in clusters.get(key, []) or []:
            independent_signal_services.add(str(svc))
            if key == "infra_unhealthy":
                degraded_services.add(str(svc))
            else:
                top_error_services.add(str(svc))

    return {
        "degraded_services": sorted(degraded_services),
        "top_error_services": sorted(top_error_services),
        "trace_sources": sorted(trace_sources),
        "trace_targets": sorted(trace_targets),
        "failed_edges": sorted(failed_edges),
        "metric_services": sorted(metric_services),
        "independent_signal_services": sorted(independent_signal_services),
    }


def _rank_focus_services(sig: dict[str, Any], limit: int) -> list[str]:
    scores: dict[str, float] = {}
    def add(items: list[str], weight: float) -> None:
        for svc in items:
            scores[svc] = scores.get(svc, 0.0) + weight
    add(sig["degraded_services"], 3.0)
    add(sig["top_error_services"], 2.2)
    add(sig["trace_targets"], 1.6)
    add(sig["trace_sources"], 0.9)
    add(sig["metric_services"], 0.7)
    add(sig["independent_signal_services"], 0.3)
    return [svc for svc, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def _fault_bias_from_signature(sig: dict[str, Any], preferred: list[str]) -> list[str]:
    bias = []
    for item in preferred:
        if item not in bias:
            bias.append(item)
    if sig["degraded_services"] and "infra_failure" not in bias:
        bias.append("infra_failure")
    if sig["top_error_services"] and "dependency_failure" not in bias:
        bias.append("dependency_failure")
    if sig["failed_edges"] and "network_failure" not in bias:
        bias.append("network_failure")
    bias.append("unknown")
    return [x for x in bias if x in CANONICAL_FAULT_TYPES][:5]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default
