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
helper services, nodes, app names, or fault-family names.

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
- Never output Kubernetes nodes, app names, namespaces, fault-family names, or helper/proxy/observability services as root cause services.
- If pods are Pending/CrashLooping/unready, nodeName is invalid, scheduling fails, or replicas are unavailable, use infra_failure.
- If auth/unauthorized/credential/permission/MongoDB auth evidence dominates, use auth_failure.
- If target ports, service ports, endpoint/service mismatch, or app config dominates, use config_error.
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
_INFRA_TOKENS = ("pending", "crashloop", "crash", "killed", "kill", "unready", "not ready", "unavailable", "node", "nodename", "schedule", "scheduling", "replica", "oom", "evicted")
_AUTH_TOKENS = ("auth", "unauthorized", "not authorized", "credential", "permission", "forbidden", "access denied", "authentication", "login", "password", "token")
_CONFIG_TOKENS = ("target_port", "target port", "port_misconfig", "misconfig", "wrong bin", "wrong_bin", "config", "configuration", "endpoint", "service port", "connection refused", "bad gateway")
_LATENCY_TOKENS = ("delay", "latency", "slow", "timeout", "timed out", "p99", "p95")
_NETWORK_TOKENS = ("packet", "loss", "network", "unreachable", "reset", "dropped", "drop")
_ERROR_TOKENS = ("error", "exception", "failed", "failure", "unhealthy", "degraded")


class LLMRCASolver:
    """Fixed API-backed RCA solver behind the prompt-policy/controller.

    The solver uses only the redacted compressed state. It reuses the existing
    RCA agent-input builder when available, because that code already knows the
    project schema for service health, log errors, trace summaries, SLA, and
    clusters. The candidate list is not ground truth; it is an evidence-derived
    shortlist used for prompting, debug, and invalid-output recovery.
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
        final = _repair_invalid_prediction(cleaned, valid_services, candidates)

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
            "postprocess_mode": "sanitize_plus_valid_service_guard",
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
        "top_error_services": slim_structured.get("top_error_services", [])[:8],
        "suspicious_trace_edges": slim_structured.get("suspicious_trace_edges", [])[:8],
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

    # Service health is the highest-quality redacted signal.
    service_health = structured.get("service_health", {}) or {}
    if isinstance(service_health, dict):
        for svc, h in service_health.items():
            if not isinstance(h, dict):
                continue
            status = str(h.get("status") or "")
            reasons = [str(x) for x in (h.get("reasons", []) or [])]
            text = " ".join([status] + reasons)
            ft = _suggest_fault_from_text(text) if text.strip() else "infra_failure"
            score = 5.0 + _text_signal_score(text)
            direct_health_rows.append({"service": svc, "status": status, "reasons": reasons[:8], "suggested_fault_type": ft, "root_cause_signal_score": round(score, 3), "source": "agent_input_builder.service_health"})
            add_candidate(svc, ft, score, "degraded_service_health")
            for group, toks in [("infra", _INFRA_TOKENS), ("auth", _AUTH_TOKENS), ("config", _CONFIG_TOKENS), ("latency", _LATENCY_TOKENS), ("network", _NETWORK_TOKENS), ("error", _ERROR_TOKENS)]:
                c = _token_count(text, toks)
                if c:
                    signal(svc, group, c, text)
            if not any(_token_count(text, toks) for toks in [_INFRA_TOKENS, _AUTH_TOKENS, _CONFIG_TOKENS, _LATENCY_TOKENS, _NETWORK_TOKENS]):
                signal(svc, "error", 1, text or "degraded service health")

    # Logs: map dominant error text to the service that emitted it.
    for item in structured.get("top_error_services", []) or []:
        if not isinstance(item, dict):
            continue
        svc = str(item.get("service") or "")
        text = json.dumps(item, sort_keys=True, default=str)
        ft = _suggest_fault_from_text(text)
        cnt = _safe_float(item.get("error_count"), 1.0)
        add_candidate(svc, ft, min(5.0, 1.0 + 0.05 * cnt + _text_signal_score(text) * 0.2), "top_log_error_service")
        group = _fault_group(ft)
        signal(svc, group, max(1.0, min(5.0, cnt)), text)

    # Traces: a failing target can be the unhealthy dependency; source is lower weight.
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
            add_candidate(parts[-1], ft, 2.0, "suspicious_trace_target")
            add_candidate(parts[0], "dependency_failure", 0.75, "suspicious_trace_source")
            signal(parts[-1], _fault_group(ft), 2, text)
            signal(parts[0], "error", 1, text)

    # SLA: unhealthy service names are useful, but lower confidence than direct health.
    for svc in _services_from_sla(structured.get("sla", {})):
        add_candidate(svc, "infra_failure", 1.5, "sla_unhealthy_service")
        signal(svc, "error", 1, "sla unhealthy service")

    # Cluster buckets often identify behavior classes in the compressed state.
    clusters = structured.get("service_clusters", {}) or {}
    if isinstance(clusters, dict):
        for bucket, vals in clusters.items():
            if not isinstance(vals, list):
                continue
            ft = _suggest_fault_from_text(str(bucket))
            for svc in vals:
                add_candidate(str(svc), ft, 1.0, f"cluster:{bucket}")
                signal(str(svc), _fault_group(ft), 1, f"cluster:{bucket}")

    # If a global auth/config hint exists, prefer matching DB/service candidates, not frontends.
    full_text = json.dumps(structured, sort_keys=True, default=str).lower()
    if _has_any(full_text, _AUTH_TOKENS):
        for svc in services:
            if "mongo" in svc.lower() or svc.lower().endswith("db"):
                add_candidate(svc, "auth_failure", 4.0, "global_auth_hint_db_service")
    if _has_any(full_text, _CONFIG_TOKENS):
        for svc in services:
            if "service" in svc.lower() and not _is_helper_service(svc):
                add_candidate(svc, "config_error", 2.5, "global_config_hint_service")

    rows = []
    for row in candidates.values():
        svc = str(row["service"])
        score = float(row["score"])
        if service_set and svc not in service_set:
            score -= 2.0
            row["reasons"].append("not_in_valid_services_penalty")
        if _is_helper_service(svc):
            score -= 6.0
            row["reasons"].append("helper_service_penalty")
        if svc.endswith("-frontend") or svc == "frontend":
            score -= 2.0
            row["reasons"].append("frontend_penalty")
        rows.append({**row, "score": round(score, 6)})
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    signal_list = []
    for row in signal_rows.values():
        row["score"] = round(sum(float(v) for v in row.get("signal_counts", {}).values()), 6)
        signal_list.append(row)
    signal_list.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    return {
        "extractor_version": "agent_input_builder_v1",
        "candidate_root_causes": rows[:60],
        "direct_health_services": sorted(direct_health_rows, key=lambda x: float(x.get("root_cause_signal_score", 0.0)), reverse=True)[:60],
        "signal_by_service": signal_list[:60],
        "service_mention_counts": service_mentions.most_common(80),
        "observed_services_sample": services[:200],
        "top_log_error_services": structured.get("top_error_services", [])[:12],
        "suspicious_trace_edges": structured.get("suspicious_trace_edges", [])[:12],
        "trace_summary": structured.get("trace_summary", {}),
        "sla_excerpt": _slim_sla(structured.get("sla", {})),
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


def _repair_invalid_prediction(prediction: str, valid_services: set[str], candidates: list[dict[str, Any]]) -> str:
    parsed = parse_fault_lines(prediction)
    if parsed:
        kept = []
        for label in parsed:
            svc = label.service
            if valid_services and svc not in valid_services:
                continue
            if _is_non_root_service(svc):
                continue
            kept.append(label.canonical_key())
        if kept:
            return "\n".join(kept)
    for row in candidates or []:
        svc = str(row.get("service") or "")
        ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
        if svc and (not valid_services or svc in valid_services) and not _is_non_root_service(svc):
            return f"{svc}::{ft}"
    return prediction if prediction else "unknown::unknown"


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


def _suggest_fault_from_text(text: str) -> str:
    t = str(text or "").lower()
    scores = {
        "auth_failure": _token_count(t, _AUTH_TOKENS) * 3.0,
        "config_error": _token_count(t, _CONFIG_TOKENS) * 2.5,
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
        + _token_count(t, _CONFIG_TOKENS) * 2.5
        + _token_count(t, _INFRA_TOKENS) * 2.0
        + _token_count(t, _LATENCY_TOKENS) * 1.5
        + _token_count(t, _NETWORK_TOKENS) * 1.5
        + _token_count(t, _ERROR_TOKENS) * 1.0
    )


def _token_count(text: str, tokens: tuple[str, ...]) -> int:
    t = str(text or "").lower()
    return sum(t.count(tok) for tok in tokens)


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    t = str(text or "").lower()
    return any(tok in t for tok in tokens)


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
    deny = {"default", "true", "false", "none", "null", "namespace", "service", "services", "pod", "pods", "metrics", "metric", "logs", "log", "trace", "traces", "output", "status", "health", "error", "warning", "failed", "success", "normal", "unknown", "node", "name", "type", "value", "analysis", "mitigation", "localization"}
    if t in deny:
        return False
    if _is_non_root_service(t):
        return False
    if "-" in t:
        return True
    if t in {"geo", "rate", "profile", "recommendation", "reservation", "search", "frontend"}:
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
