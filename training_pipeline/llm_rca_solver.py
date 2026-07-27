from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import CANONICAL_FAULT_TYPES, normalize_fault_type, parse_fault_lines


_SYSTEM_PROMPT = """\
You are a fixed RCA solver for Kubernetes/AIOps incidents.

You receive a redacted, agent-visible RCA evidence object. Generated scenario IDs,
ground truth, raw specs, and faulty-service labels have been removed.

Your task is to identify the true upstream root cause service, not downstream victims,
helper services, Kubernetes nodes, app names, namespaces, or fault-family names.

Output contract:
- Output ONLY root-cause lines.
- One root cause per line.
- Each line must be exactly: service::fault_type
- service must be one of valid_services.
- fault_type must be one of:
  infra_failure, auth_failure, dependency_failure, resource_exhaustion,
  latency_degradation, network_failure, config_error, unknown
- No prose, no markdown, no JSON, no bullets, no explanations.

Diagnosis rules:
- Default to ONE root cause unless the evidence explicitly shows independent faults in unrelated services.
- Prefer candidate_root_causes when supported by service health, logs, traces, or SLA.
- Do not pick the service with the largest fan-out unless there is direct evidence it is the upstream cause.
- Degraded/log-error clusters often contain downstream victims; treat cluster-only dependency_failure as weak evidence.
- Prefer explicit fault-type evidence over generic degraded-service evidence.
- If pods are Pending/CrashLooping/unready, nodeName is invalid, scheduling fails, containers are killed, or replicas are unavailable, use infra_failure.
- If auth/unauthorized/credential/permission/MongoDB auth evidence dominates, use auth_failure.
- If target ports, service ports, endpoint/service mismatch, wrong binary, or app config dominates, use config_error.
- If latency/delay dominates, use latency_degradation.
- If packet loss/network unreachable dominates, use network_failure.
- Use dependency_failure only when a dependency is unhealthy and no more specific auth/config/network/infra cause is supported.
"""


_HELPER_SERVICE_PATTERNS = (
    "jaeger",
    "nginx-thrift",
    "zipkin",
    "prometheus",
    "grafana",
    "loadgenerator",
    "nginx-web-server",
)
_NON_ROOT_SERVICE_PATTERNS = (
    "kind-worker",
    "kind-control-plane",
    "container-kill",
    "social-network-microservices",
    "hotel-reservation-microservices",
)

_INFRA_TOKENS = (
    "pending", "crashloop", "crash", "killed", "kill", "unready", "not ready",
    "unavailable", "nodename", "node name", "schedule", "scheduling", "replica",
    "oom", "evicted", "containerstatus", "waiting", "terminated",
)
_AUTH_TOKENS = (
    "auth", "unauthorized", "not authorized", "credential", "permission", "forbidden",
    "access denied", "authentication", "login", "password", "token", "sasl",
)
# Keep generic "config" out of the main config detector. It appears in many
# harmless schema/prompt locations and caused v6 to label infra faults as config.
_CONFIG_TOKENS = (
    "target_port", "target port", "targetport", "port_misconfig", "port misconfig",
    "wrong bin", "wrong_bin", "endpoint mismatch", "service port", "container port",
    "connection refused", "bad gateway", "no endpoints", "endpoint not found",
)
_LATENCY_TOKENS = ("delay", "latency", "slow", "timeout", "timed out", "p99", "p95")
_NETWORK_TOKENS = ("packet", "loss", "network", "unreachable", "reset", "dropped", "drop")
_ERROR_TOKENS = ("error", "exception", "failed", "failure", "unhealthy", "degraded")

_CACHE_DB_MARKERS = ("redis", "memcached")
_DB_MARKERS = ("mongo", "mongodb", "-db", "db")


class LLMRCASolver:
    """Fixed API-backed RCA solver behind the prompt-policy/controller.

    The trainable object remains the RCA instruction policy. This solver consumes
    the policy-generated instruction plus only redacted compressed state.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        max_tokens: int = 300,
        temperature: float = 0.0,
        state_char_budget: int = 24000,
        cache_path: str | None = None,
        max_root_causes: int = 1,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.state_char_budget = int(state_char_budget)
        self.max_root_causes = max(1, int(max_root_causes))
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.debug_path = _default_debug_path(self.cache_path)
        self._cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

        from agents.llm_client import LLMClient

        self.client = LLMClient(provider=provider, model=model)
        self.model_name = self.client.model

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        compact = compact_state_for_llm(compressed_state, char_budget=self.state_char_budget)
        cache_key = _stable_hash({
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_root_causes": self.max_root_causes,
            "instruction": instruction,
            "state_hash": _stable_hash(compact),
        })
        if cache_key in self._cache:
            return self._cache[cache_key]

        user_prompt = self._build_user_prompt(compact, instruction)
        raw = self.client.call(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        sanitized_unlimited = sanitize_rca_prediction(raw, compressed_state, max_root_causes=None)
        cleaned = sanitize_rca_prediction(raw, compressed_state, max_root_causes=self.max_root_causes)
        evidence = compact.get("high_signal_evidence") if isinstance(compact, dict) else {}
        candidates = evidence.get("candidate_root_causes", []) if isinstance(evidence, dict) else []
        valid_services = set(str(x) for x in compact.get("valid_services", []) if x)
        final, repair_reason = _repair_prediction_with_candidates(cleaned, valid_services, candidates)

        debug = {
            "key": cache_key,
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_root_causes": self.max_root_causes,
            "state_hash": _stable_hash(compact),
            "input_top_level_keys": sorted(str(k) for k in (compressed_state or {}).keys()) if isinstance(compressed_state, dict) else [],
            "compact_keys": sorted(str(k) for k in compact.keys()) if isinstance(compact, dict) else [],
            "valid_services": sorted(valid_services)[:200],
            "instruction": instruction,
            "raw_response": raw,
            "sanitized_unlimited": sanitized_unlimited,
            "sanitized_prediction": cleaned,
            "final_prediction_after_validity_guard": final,
            "postprocess_mode": "sanitize_plus_candidate_guard_v2",
            "candidate_repair_reason": repair_reason,
            "high_signal_evidence": evidence,
            "candidate_root_causes": candidates,
        }
        self._cache[cache_key] = final
        self._append_cache(cache_key, final, raw)
        self._append_debug(debug)
        return final

    def _build_user_prompt(self, compact_state: dict[str, Any], instruction: str) -> str:
        payload = {
            "policy_instruction": instruction,
            "root_cause_count_instruction": f"Return at most {self.max_root_causes} root cause line(s). Use fewer if only one upstream cause is supported.",
            "valid_services": compact_state.get("valid_services", []),
            "candidate_root_causes": (compact_state.get("high_signal_evidence") or {}).get("candidate_root_causes", [])[:20],
            "root_cause_selection_guidance": [
                "A high reproduction/fan-out score is not enough; choose the most upstream specific service.",
                "Cluster-only dependency_failure candidates are weak unless supported by explicit auth/config/network/infra text.",
                "For MongoDB auth evidence, prefer the MongoDB service with the most local auth evidence, not a random MongoDB victim.",
                "For target-port/service-port evidence, prefer the affected *-service, not nginx or a downstream caller.",
                "For killed/crash/unready/scheduling evidence, prefer the application service over its database dependency.",
            ],
            "redacted_rca_evidence": compact_state,
            "output_contract": "Return only service::fault_type lines. No prose.",
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        with self.cache_path.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                key = row.get("key")
                pred = row.get("prediction")
                if key and isinstance(pred, str):
                    self._cache[str(key)] = pred

    def _append_cache(self, key: str, prediction: str, raw_response: str) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"key": key, "provider": self.provider, "model": self.model_name, "max_root_causes": self.max_root_causes, "prediction": prediction, "raw_response": raw_response}
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _append_debug(self, row: dict[str, Any]) -> None:
        if not self.debug_path:
            return
        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def compact_state_for_llm(compressed_state: dict[str, Any], char_budget: int = 24000) -> dict[str, Any]:
    agent_view = _build_schema_native_rca_view(compressed_state)
    structured = _strip_leaky_fields(agent_view.get("structured_input", {}))
    evidence = high_signal_evidence_summary(compressed_state, structured=structured)
    valid_services = evidence.get("observed_services_sample", []) or structured.get("all_services", []) or _fallback_services_from_text(compressed_state)
    valid_services = sorted({str(s) for s in valid_services if _looks_like_service(str(s)) and not _is_non_root_service(str(s))})

    compact = {
        "high_signal_evidence": evidence,
        "valid_services": valid_services,
        "rca_agent_structured_evidence": structured,
        "redaction_note": "Generated scenario_id, ground_truth, fault_context, raw_spec, and known faulty-service fields are excluded from this prompt.",
    }
    text = json.dumps(compact, sort_keys=True, default=str)
    if len(text) <= char_budget:
        return compact

    slim_structured = {
        "namespace": structured.get("namespace"),
        "task": structured.get("task"),
        "all_services": valid_services[:160],
        "service_health": structured.get("service_health", {}),
        "top_error_services": structured.get("top_error_services", [])[:12],
        "suspicious_trace_edges": structured.get("suspicious_trace_edges", [])[:12],
        "trace_summary": structured.get("trace_summary", {}),
        "service_clusters": structured.get("service_clusters", {}),
        "sla": _slim_sla(structured.get("sla", {})),
    }
    compact2 = {"high_signal_evidence": evidence, "valid_services": valid_services[:200], "rca_agent_structured_evidence": slim_structured}
    text2 = json.dumps(compact2, sort_keys=True, default=str)
    if len(text2) <= char_budget:
        return compact2

    compact2["_truncated_note"] = f"state truncated to approximately {char_budget} chars for LLM RCA solver"
    compact2["rca_agent_structured_evidence"] = {
        "all_services": valid_services[:200],
        "service_health": slim_structured.get("service_health", {}),
        "top_error_services": slim_structured.get("top_error_services", [])[:10],
        "suspicious_trace_edges": slim_structured.get("suspicious_trace_edges", [])[:10],
    }
    return compact2


def _build_schema_native_rca_view(compressed_state: dict[str, Any]) -> dict[str, Any]:
    try:
        from state_abstraction_full.agent_input_builder import build_rca_agent_input

        return build_rca_agent_input(compressed_state, compressed_state)
    except Exception as e:
        return {"structured_input": _fallback_structured_view(compressed_state), "builder_error": repr(e)}


def _fallback_structured_view(compressed_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "namespace": compressed_state.get("namespace"),
        "task": compressed_state.get("task"),
        "all_services": compressed_state.get("services", []),
        "service_health": compressed_state.get("service_health", {}),
        "top_error_services": (compressed_state.get("llm_view", {}) or {}).get("top_log_error_services", []),
        "suspicious_trace_edges": [],
        "trace_summary": (compressed_state.get("traces", {}) or {}).get("summary", {}) if isinstance(compressed_state.get("traces"), dict) else {},
        "service_clusters": compressed_state.get("clusters", {}),
        "sla": compressed_state.get("sla", {}),
        "observability_metadata": compressed_state.get("observability_metadata", {}),
    }


def high_signal_evidence_summary(compressed_state: dict[str, Any], structured: dict[str, Any] | None = None) -> dict[str, Any]:
    structured = structured or _strip_leaky_fields(_build_schema_native_rca_view(compressed_state).get("structured_input", {}))
    services = sorted({str(s) for s in (structured.get("all_services", []) or []) if _looks_like_service(str(s)) and not _is_non_root_service(str(s))})
    if not services:
        services = _fallback_services_from_text(compressed_state)
    service_set = set(services)

    signature = _symptom_signature(compressed_state)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    service_mentions: Counter[str] = Counter()
    signal_rows: dict[str, dict[str, Any]] = {}
    direct_health_rows: list[dict[str, Any]] = []

    def signal(svc: str, group: str, amount: float, evidence: str) -> None:
        svc = str(svc or "").strip()
        if not svc or _is_non_root_service(svc):
            return
        service_mentions[svc] += max(1, int(amount))
        row = signal_rows.setdefault(svc, {"service": svc, "signal_counts": {"infra": 0, "auth": 0, "config": 0, "latency": 0, "network": 0, "error": 0}, "evidence_excerpt": []})
        row["signal_counts"][group] = row["signal_counts"].get(group, 0) + amount
        if evidence and len(row["evidence_excerpt"]) < 8:
            row["evidence_excerpt"].append(evidence[:400])

    def add_candidate(svc: str, fault_type: str, score: float, reason: str) -> None:
        svc = str(svc or "").strip()
        if not svc or _is_non_root_service(svc):
            return
        ft = normalize_fault_type(str(fault_type or "unknown"))
        key = (svc, ft)
        row = candidates.setdefault(key, {"service": svc, "fault_type": ft, "score": 0.0, "reasons": []})
        row["score"] = round(float(row["score"]) + float(score), 6)
        if reason not in row["reasons"]:
            row["reasons"].append(reason)

    # Direct service health. Generic degraded status is weak; explicit type words are strong.
    service_health = structured.get("service_health", {}) or {}
    if isinstance(service_health, dict):
        for svc, h in service_health.items():
            if not isinstance(h, dict) or _is_non_root_service(str(svc)):
                continue
            status = str(h.get("status") or "")
            reasons = [str(x) for x in (h.get("reasons", []) or [])]
            text = " ".join([status] + reasons)
            explicit_ft = _suggest_fault_from_text(text)
            generic = explicit_ft == "dependency_failure" and not _has_any(text, _INFRA_TOKENS + _AUTH_TOKENS + _CONFIG_TOKENS + _LATENCY_TOKENS + _NETWORK_TOKENS)
            score = 2.0 if generic else 7.0 + _text_signal_score(text)
            if generic:
                # A bare degraded flag alone means the service is affected, not necessarily root cause.
                explicit_ft = "dependency_failure"
            direct_health_rows.append({
                "service": str(svc),
                "status": status,
                "reasons": reasons[:8],
                "suggested_fault_type": explicit_ft,
                "root_cause_signal_score": round(score, 3),
                "source": "agent_input_builder.service_health",
            })
            add_candidate(str(svc), explicit_ft, score, "direct_service_health" if not generic else "generic_degraded_service_health")
            for group, toks in [("infra", _INFRA_TOKENS), ("auth", _AUTH_TOKENS), ("config", _CONFIG_TOKENS), ("latency", _LATENCY_TOKENS), ("network", _NETWORK_TOKENS), ("error", _ERROR_TOKENS)]:
                c = _token_count(text, toks)
                if c:
                    signal(str(svc), group, c, text)
            if generic:
                signal(str(svc), "error", 1, text or "generic degraded service health")

    # Signature from behavioral comparator: useful to recover degraded services not exposed by builder.
    for svc in signature.get("degraded_services", []) or []:
        if str(svc) in service_set and not _is_non_root_service(str(svc)):
            add_candidate(str(svc), "dependency_failure", 1.25, "symptom_signature_degraded")
            signal(str(svc), "error", 1, "symptom_signature_degraded")
    for svc in signature.get("metric_anomaly_services", []) or []:
        if str(svc) in service_set:
            add_candidate(str(svc), "resource_exhaustion", 2.0, "metric_anomaly_service")
            signal(str(svc), "infra", 1, "metric_anomaly_service")

    # Logs: local explicit fault words are useful. Generic log-error fan-out is lower weight.
    top_error_services = [x for x in (structured.get("top_error_services", []) or []) if isinstance(x, dict)]
    top_error_names = [str(x.get("service") or "") for x in top_error_services]
    for item in top_error_services:
        svc = str(item.get("service") or "")
        if not svc or _is_non_root_service(svc):
            continue
        text = json.dumps(item, sort_keys=True, default=str)
        ft = _suggest_fault_from_text(text)
        cnt = _safe_float(item.get("error_count"), 1.0)
        explicit = ft != "dependency_failure"
        score = (2.0 if explicit else 0.75) + min(2.0, 0.02 * cnt) + min(3.0, _text_signal_score(text) * 0.15)
        add_candidate(svc, ft, score, "explicit_top_log_error_service" if explicit else "generic_top_log_error_service")
        signal(svc, _fault_group(ft), max(1.0, min(4.0, cnt)), text)

    # Traces: target is stronger than source, but still not enough to beat explicit health.
    trace_summary = structured.get("trace_summary", {}) or {}
    edge_rows = list(structured.get("suspicious_trace_edges", []) or [])
    for edge in trace_summary.get("failed_edges", []) or []:
        edge_rows.append({"edge": edge, "failure_type": "failed_edge"})
    for edge in trace_summary.get("slow_edges", []) or []:
        edge_rows.append({"edge": edge, "failure_type": "slow_edge"})
    for row in edge_rows:
        if not isinstance(row, dict):
            continue
        edge = str(row.get("edge") or "")
        parts = [p.strip() for p in re.split(r"->|=>|,", edge) if p.strip()]
        text = json.dumps(row, sort_keys=True, default=str)
        ft = _suggest_fault_from_text(text)
        if len(parts) >= 2:
            add_candidate(parts[-1], ft, 1.75, "suspicious_trace_target")
            add_candidate(parts[0], "dependency_failure", 0.4, "suspicious_trace_source")
            signal(parts[-1], _fault_group(ft), 2, text)
            signal(parts[0], "error", 1, text)

    for svc in _services_from_sla(structured.get("sla", {})):
        add_candidate(svc, "dependency_failure", 0.75, "sla_unhealthy_service")
        signal(svc, "error", 1, "sla unhealthy service")

    clusters = structured.get("service_clusters", {}) or {}
    if isinstance(clusters, dict):
        for bucket, vals in clusters.items():
            if not isinstance(vals, list):
                continue
            bucket_s = str(bucket)
            ft = _suggest_fault_from_text(bucket_s)
            for svc in vals:
                # Cluster-only evidence is weak; it should not dominate root-cause ranking.
                add_candidate(str(svc), ft, 0.35, f"weak_cluster:{bucket_s}")
                signal(str(svc), _fault_group(ft), 0.5, f"cluster:{bucket_s}")

    full_text = json.dumps(structured, sort_keys=True, default=str).lower()
    explicit_auth = _has_any(full_text, _AUTH_TOKENS)
    explicit_config = _has_any(full_text, _CONFIG_TOKENS)
    explicit_infra = _has_any(full_text, _INFRA_TOKENS)

    if explicit_auth:
        for svc in services:
            if _is_db_service(svc):
                local = _service_local_text(structured, svc)
                bonus = 5.0 if _has_any(local, _AUTH_TOKENS) else 2.0
                if svc in top_error_names:
                    bonus += 1.5
                add_candidate(svc, "auth_failure", bonus, "auth_hint_db_service")
                signal(svc, "auth", 2, local[:400] or "global auth hint")

    if explicit_config:
        for svc in services:
            if "service" in svc.lower() and not _is_helper_service(svc):
                local = _service_local_text(structured, svc)
                bonus = 5.0 if _has_any(local, _CONFIG_TOKENS) else 1.0
                if svc in top_error_names:
                    bonus += 1.0
                add_candidate(svc, "config_error", bonus, "specific_config_hint_service")
                signal(svc, "config", 2, local[:400] or "specific config hint")

    if explicit_infra:
        for svc in services:
            local = _service_local_text(structured, svc)
            if _has_any(local, _INFRA_TOKENS):
                add_candidate(svc, "infra_failure", 5.0 + _text_signal_score(local) * 0.2, "infra_hint_local_service")
                signal(svc, "infra", 2, local[:400])

    # Special non-oracle structural cue: in hotel-reservation, a business service
    # and its MongoDB dependency often both appear degraded. For killed/unready
    # infra text, prefer the business service as the upstream container/pod fault.
    if explicit_infra:
        for svc in services:
            if _is_db_or_cache_service(svc):
                continue
            paired = [d for d in services if d != svc and svc in d and _is_db_or_cache_service(d)]
            if paired:
                local = _service_local_text(structured, svc)
                if _has_any(local + " " + full_text[:20000], _INFRA_TOKENS):
                    add_candidate(svc, "infra_failure", 4.0, "business_service_over_paired_db_for_infra")

    rows = []
    for row in candidates.values():
        svc = str(row["service"])
        ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
        score = float(row["score"])
        reasons = list(row.get("reasons", []))

        if service_set and svc not in service_set:
            score -= 2.0
            reasons.append("not_in_valid_services_penalty")
        if _is_helper_service(svc):
            score -= 8.0
            reasons.append("helper_service_penalty")
        if svc.endswith("-frontend") or svc == "frontend":
            score -= 2.0
            reasons.append("frontend_penalty")
        if _is_cache_service(svc) and ft == "dependency_failure":
            score -= 2.0
            reasons.append("cache_dependency_fanout_penalty")
        if _is_db_service(svc) and ft == "dependency_failure" and not explicit_auth and not explicit_network_local(structured, svc):
            score -= 1.0
            reasons.append("generic_db_dependency_penalty")
        if ft == "dependency_failure" and reasons and all(r.startswith("weak_cluster") or r in {"symptom_signature_degraded", "sla_unhealthy_service"} for r in reasons):
            score -= 2.0
            reasons.append("cluster_only_dependency_penalty")
        rows.append({**row, "fault_type": ft, "score": round(score, 6), "reasons": reasons})

    rows.sort(key=lambda r: (float(r.get("score", 0.0)), _specificity_tiebreak(str(r.get("service", "")), str(r.get("fault_type", "")))), reverse=True)

    signal_list = []
    for row in signal_rows.values():
        row["score"] = round(sum(float(v) for v in row.get("signal_counts", {}).values()), 6)
        signal_list.append(row)
    signal_list.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    return {
        "extractor_version": "agent_input_builder_v2_typed_ranking",
        "candidate_root_causes": rows[:80],
        "direct_health_services": sorted(direct_health_rows, key=lambda x: float(x.get("root_cause_signal_score", 0.0)), reverse=True)[:80],
        "signal_by_service": signal_list[:80],
        "service_mention_counts": service_mentions.most_common(100),
        "observed_services_sample": services[:200],
        "top_log_error_services": top_error_services[:12],
        "suspicious_trace_edges": structured.get("suspicious_trace_edges", [])[:12],
        "trace_summary": structured.get("trace_summary", {}),
        "sla_excerpt": _slim_sla(structured.get("sla", {})),
        "global_signal_flags": {"explicit_auth": explicit_auth, "explicit_config": explicit_config, "explicit_infra": explicit_infra},
        "builder_source": "state_abstraction_full.agent_input_builder.build_rca_agent_input",
        "invalid_service_examples": list(_NON_ROOT_SERVICE_PATTERNS),
    }


def sanitize_rca_prediction(raw: str, compressed_state: dict[str, Any] | None = None, max_root_causes: int | None = None) -> str:
    text = _strip_fences(str(raw or "").strip())
    lines: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().strip("`*->•- ").strip()
        if "::" not in cleaned:
            continue
        left, right = [x.strip() for x in cleaned.split("::", 1)]
        left = _clean_service(left)
        right = normalize_fault_type(_clean_fault_type(right))
        if left:
            lines.append(f"{left}::{right}")
    if not lines:
        lines = _lines_from_json(_extract_json(text))
    if lines:
        return _dedupe_lines(lines, compressed_state=compressed_state, max_root_causes=max_root_causes)
    return "unknown::unknown"


def _repair_prediction_with_candidates(prediction: str, valid_services: set[str], candidates: list[dict[str, Any]]) -> tuple[str, str]:
    parsed = parse_fault_lines(prediction)
    best = _best_valid_candidate(candidates, valid_services)
    if not parsed:
        return (_candidate_key(best), "fallback_no_parse") if best else (prediction or "unknown::unknown", "no_parse_no_candidate")

    label = parsed[0]
    pred_key = label.canonical_key()
    pred_service = label.service
    pred_ft = normalize_fault_type(label.fault_type)

    if (valid_services and pred_service not in valid_services) or _is_non_root_service(pred_service):
        return (_candidate_key(best), "fallback_invalid_service") if best else (pred_key, "invalid_service_no_candidate")

    pred_score = _candidate_score_for(pred_service, pred_ft, candidates)
    best_score = _safe_float(best.get("score")) if best else 0.0
    best_service = str(best.get("service") or "") if best else ""
    best_ft = normalize_fault_type(str(best.get("fault_type") or "unknown")) if best else "unknown"

    # Same service, better fault type: safe to repair because service identity is unchanged.
    if best and best_service == pred_service and best_ft != pred_ft and best_score >= pred_score + 1.0:
        return _candidate_key(best), "same_service_stronger_fault_type"

    # Strong typed evidence candidate can override weak downstream dependency guesses.
    if best and best_score >= 8.0 and best_score >= pred_score + 3.0:
        if pred_ft == "dependency_failure" or _is_db_or_cache_service(pred_service) or pred_service.endswith("-frontend"):
            return _candidate_key(best), "strong_candidate_over_weak_downstream_prediction"

    # Keep a valid LLM prediction when no strong candidate contradicts it.
    return pred_key, "kept_valid_llm_prediction"


def _best_valid_candidate(candidates: list[dict[str, Any]], valid_services: set[str]) -> dict[str, Any] | None:
    for row in candidates or []:
        svc = str(row.get("service") or "")
        if svc and (not valid_services or svc in valid_services) and not _is_non_root_service(svc):
            return row
    return None


def _candidate_score_for(service: str, fault_type: str, candidates: list[dict[str, Any]]) -> float:
    best = 0.0
    for row in candidates or []:
        if str(row.get("service") or "") == service and normalize_fault_type(str(row.get("fault_type") or "unknown")) == normalize_fault_type(fault_type):
            best = max(best, _safe_float(row.get("score")))
    return best


def _candidate_key(row: dict[str, Any] | None) -> str:
    if not row:
        return "unknown::unknown"
    return f"{row.get('service')}::{normalize_fault_type(str(row.get('fault_type') or 'unknown'))}"


def _lines_from_json(parsed: Any) -> list[str]:
    out: list[str] = []
    rows = parsed if isinstance(parsed, list) else [parsed]
    for row in rows:
        if not isinstance(row, dict):
            continue
        service = row.get("service") or row.get("root_cause_service") or row.get("root_cause")
        fault_type = row.get("fault_type") or row.get("fault") or row.get("fault_family")
        if service:
            out.append(f"{_clean_service(str(service))}::{normalize_fault_type(str(fault_type or 'unknown'))}")
    return out


def _extract_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def _strip_leaky_fields(obj: Any) -> Any:
    forbidden = {"scenario_id", "ground_truth", "fault_context", "faulty_service", "fault_instances", "raw_spec", "problem_description", "known_fault_hypotheses", "primary_fault"}
    if isinstance(obj, dict):
        return {str(k): _strip_leaky_fields(v) for k, v in obj.items() if str(k) not in forbidden and not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_leaky_fields(v) for v in obj]
    return obj


def _symptom_signature(compressed_state: dict[str, Any]) -> dict[str, Any]:
    try:
        from digital_twin_runtime.telemetry_comparator import symptom_signature

        sig = symptom_signature(compressed_state)
        return sig if isinstance(sig, dict) else {}
    except Exception:
        return {}


def _slim_sla(sla: Any) -> Any:
    if not isinstance(sla, dict):
        return sla
    out = {}
    for key in ["violated", "global_sla", "dependency_sla", "counts", "hard_violations", "weighted_violations"]:
        if key in sla:
            out[key] = sla[key]
    per = sla.get("per_service")
    if isinstance(per, dict):
        out["unhealthy_services"] = [svc for svc, v in per.items() if isinstance(v, dict) and not v.get("healthy", True)][:40]
    return out


def _services_from_sla(sla: Any) -> list[str]:
    out = []
    if not isinstance(sla, dict):
        return out
    for svc in sla.get("unhealthy_services", []) or []:
        if _looks_like_service(str(svc)):
            out.append(str(svc))
    per = sla.get("per_service", {}) or {}
    if isinstance(per, dict):
        for svc, row in per.items():
            if isinstance(row, dict) and not row.get("healthy", True) and _looks_like_service(str(svc)):
                out.append(str(svc))
    return sorted(set(out))


def _fallback_services_from_text(obj: Any) -> list[str]:
    text = json.dumps(_strip_leaky_fields(obj), sort_keys=True, default=str).lower()[:200000]
    vals = set()
    for m in re.finditer(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b", text):
        tok = m.group(0)
        if _looks_like_service(tok) and not _is_non_root_service(tok):
            vals.add(tok)
    return sorted(vals)


def _service_local_text(structured: dict[str, Any], service: str) -> str:
    svc = str(service or "").lower()
    chunks: list[str] = []
    health = structured.get("service_health", {}) or {}
    if isinstance(health, dict) and service in health:
        chunks.append(json.dumps(health.get(service), sort_keys=True, default=str))
    for item in structured.get("top_error_services", []) or []:
        if isinstance(item, dict) and str(item.get("service") or "").lower() == svc:
            chunks.append(json.dumps(item, sort_keys=True, default=str))
    for item in structured.get("suspicious_trace_edges", []) or []:
        if isinstance(item, dict) and svc in json.dumps(item, sort_keys=True, default=str).lower():
            chunks.append(json.dumps(item, sort_keys=True, default=str))
    clusters = structured.get("service_clusters", {}) or {}
    if isinstance(clusters, dict):
        for bucket, vals in clusters.items():
            if isinstance(vals, list) and service in vals:
                chunks.append(f"cluster:{bucket}")
    return "\n".join(chunks).lower()


def _suggest_fault_from_text(text: str) -> str:
    t = str(text or "").lower()
    scores = {
        "auth_failure": _token_count(t, _AUTH_TOKENS) * 3.0,
        "config_error": _token_count(t, _CONFIG_TOKENS) * 3.0,
        "network_failure": _token_count(t, _NETWORK_TOKENS) * 2.0,
        "latency_degradation": _token_count(t, _LATENCY_TOKENS) * 2.0,
        "infra_failure": _token_count(t, _INFRA_TOKENS) * 2.0,
    }
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else "dependency_failure"


def _fault_group(fault_type: str) -> str:
    ft = normalize_fault_type(str(fault_type or "unknown"))
    return {
        "infra_failure": "infra",
        "auth_failure": "auth",
        "config_error": "config",
        "latency_degradation": "latency",
        "network_failure": "network",
        "resource_exhaustion": "infra",
        "dependency_failure": "error",
    }.get(ft, "error")


def _text_signal_score(text: str) -> float:
    t = str(text or "").lower()
    return float(
        _token_count(t, _AUTH_TOKENS) * 3.0
        + _token_count(t, _CONFIG_TOKENS) * 3.0
        + _token_count(t, _INFRA_TOKENS) * 2.0
        + _token_count(t, _LATENCY_TOKENS) * 1.5
        + _token_count(t, _NETWORK_TOKENS) * 1.5
        + _token_count(t, _ERROR_TOKENS) * 0.75
    )


def _token_count(text: str, tokens: tuple[str, ...]) -> int:
    t = str(text or "").lower()
    return sum(t.count(tok) for tok in tokens)


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    t = str(text or "").lower()
    return any(tok in t for tok in tokens)


def explicit_network_local(structured: dict[str, Any], service: str) -> bool:
    return _has_any(_service_local_text(structured, service), _NETWORK_TOKENS)


def _specificity_tiebreak(service: str, fault_type: str) -> float:
    s = service.lower()
    ft = normalize_fault_type(fault_type)
    score = 0.0
    if ft != "dependency_failure":
        score += 2.0
    if "-service" in s:
        score += 1.0
    if _is_db_service(s) and ft == "auth_failure":
        score += 1.0
    if _is_db_or_cache_service(s) and ft == "dependency_failure":
        score -= 1.0
    if s.endswith("-frontend") or s == "frontend":
        score -= 2.0
    return score


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _clean_service(text: str) -> str:
    value = str(text or "").strip().strip('"\'`,.;:()[]{}')
    value = re.sub(r"^service\s*=\s*", "", value, flags=re.IGNORECASE)
    return value.strip()


def _clean_fault_type(text: str) -> str:
    value = str(text or "").strip().strip('"\'`,.;:()[]{}')
    value = re.sub(r"\s+#.*$", "", value)
    value = re.sub(r"\s+", "_", value)
    return value.strip()


def _dedupe_lines(lines: list[str], compressed_state: dict[str, Any] | None = None, max_root_causes: int | None = None) -> str:
    seen = set()
    out = []
    for line in lines:
        parsed = parse_fault_lines(line)
        if not parsed:
            continue
        normalized = parsed[0].canonical_key()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    if not out:
        return "unknown::unknown"
    if max_root_causes is not None:
        out = out[: max(1, int(max_root_causes))]
    return "\n".join(out)


def _looks_like_service(token: str) -> bool:
    t = str(token or "").strip().lower()
    if not t or len(t) < 3:
        return False
    deny = {"default", "true", "false", "none", "null", "namespace", "service", "services", "pod", "pods", "metrics", "metric", "logs", "log", "trace", "traces", "output", "status", "health", "error", "warning", "failed", "success", "normal", "unknown", "node", "name", "type", "value", "analysis", "mitigation", "localization", "root"}
    if t in deny:
        return False
    if _is_non_root_service(t):
        return False
    if "-" in t:
        return True
    if t in {"geo", "rate", "profile", "recommendation", "reservation", "search", "frontend", "consul"}:
        return True
    if "mongo" in t or t.endswith("db"):
        return True
    return False


def _is_helper_service(service: str) -> bool:
    s = str(service or "").lower()
    return any(pat in s for pat in _HELPER_SERVICE_PATTERNS)


def _is_non_root_service(service: str) -> bool:
    s = str(service or "").lower()
    return any(pat == s or pat in s for pat in _NON_ROOT_SERVICE_PATTERNS) or _is_helper_service(s)


def _is_cache_service(service: str) -> bool:
    s = str(service or "").lower()
    return any(x in s for x in _CACHE_DB_MARKERS)


def _is_db_service(service: str) -> bool:
    s = str(service or "").lower()
    return "mongo" in s or s.endswith("db") or "-db" in s


def _is_db_or_cache_service(service: str) -> bool:
    return _is_db_service(service) or _is_cache_service(service)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _default_debug_path(cache_path: Path | None) -> Path | None:
    if cache_path is None:
        return None
    return cache_path.parent / "llm_rca_debug.jsonl"


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
