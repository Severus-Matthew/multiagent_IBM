from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import normalize_fault_type, parse_fault_lines
from .llm_rca_solver import (
    _AUTH_TOKENS,
    _CACHE_DB_MARKERS,
    _CONFIG_TOKENS,
    _ERROR_TOKENS,
    _HELPER_SERVICE_PATTERNS,
    _INFRA_TOKENS,
    _LATENCY_TOKENS,
    _NETWORK_TOKENS,
    _NON_ROOT_SERVICE_PATTERNS,
    _build_schema_native_rca_view,
    _clean_fault_type,
    _clean_service,
    _default_debug_path,
    _extract_json,
    _fallback_services_from_text,
    _fault_group,
    _has_any,
    _is_cache_service,
    _is_db_or_cache_service,
    _is_db_service,
    _is_helper_service,
    _is_non_root_service,
    _looks_like_service,
    _safe_float,
    _service_local_text,
    _services_from_sla,
    _slim_sla,
    _specificity_tiebreak,
    _stable_hash,
    _strip_fences,
    _strip_leaky_fields,
    _suggest_fault_from_text,
    _symptom_signature,
    _text_signal_score,
    _token_count,
    sanitize_rca_prediction,
)


BUSINESS_SINGLE_WORD_SERVICES = {
    "geo", "rate", "profile", "recommendation", "reservation", "search", "frontend", "user", "consul",
}


def compact_state_for_llm_v3(compressed_state: dict[str, Any], char_budget: int = 24000) -> dict[str, Any]:
    agent_view = _build_schema_native_rca_view(compressed_state)
    structured = _strip_leaky_fields(agent_view.get("structured_input", {}))
    evidence = high_signal_evidence_summary_v3(compressed_state, structured=structured)
    valid_services = evidence.get("observed_services_sample", []) or structured.get("all_services", []) or _fallback_services_from_text(compressed_state)
    valid_services = sorted({str(s) for s in valid_services if _looks_like_service_v3(str(s)) and not _is_non_root_service(str(s))})

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
        "all_services": valid_services[:180],
        "service_health": structured.get("service_health", {}),
        "top_error_services": structured.get("top_error_services", [])[:16],
        "suspicious_trace_edges": structured.get("suspicious_trace_edges", [])[:16],
        "trace_summary": structured.get("trace_summary", {}),
        "service_clusters": structured.get("service_clusters", {}),
        "sla": _slim_sla(structured.get("sla", {})),
    }
    return {"high_signal_evidence": evidence, "valid_services": valid_services[:220], "rca_agent_structured_evidence": slim_structured}


def high_signal_evidence_summary_v3(compressed_state: dict[str, Any], structured: dict[str, Any] | None = None) -> dict[str, Any]:
    structured = structured or _strip_leaky_fields(_build_schema_native_rca_view(compressed_state).get("structured_input", {}))
    services = sorted({str(s) for s in (structured.get("all_services", []) or []) if _looks_like_service_v3(str(s)) and not _is_non_root_service(str(s))})
    if not services:
        services = [s for s in _fallback_services_from_text(compressed_state) if _looks_like_service_v3(s)]
    service_set = set(services)

    signature = _symptom_signature(compressed_state)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    service_mentions: Counter[str] = Counter()
    signal_rows: dict[str, dict[str, Any]] = {}
    direct_health_rows: list[dict[str, Any]] = []

    top_error_services = [x for x in (structured.get("top_error_services", []) or []) if isinstance(x, dict)]
    top_error_names = [str(x.get("service") or "") for x in top_error_services]
    full_text = json.dumps(structured, sort_keys=True, default=str).lower()
    explicit_auth = _has_any(full_text, _AUTH_TOKENS)
    explicit_config = _has_any(full_text, _CONFIG_TOKENS)
    explicit_infra = _has_any(full_text, _INFRA_TOKENS)
    explicit_latency = _has_any(full_text, _LATENCY_TOKENS)
    explicit_network = _has_any(full_text, _NETWORK_TOKENS)

    def signal(svc: str, group: str, amount: float, evidence: str) -> None:
        svc = str(svc or "").strip()
        if not svc or _is_non_root_service(svc):
            return
        service_mentions[svc] += max(1, int(amount))
        row = signal_rows.setdefault(svc, {"service": svc, "signal_counts": {"infra": 0, "auth": 0, "config": 0, "latency": 0, "network": 0, "error": 0}, "evidence_excerpt": []})
        row["signal_counts"][group] = row["signal_counts"].get(group, 0) + amount
        if evidence and len(row["evidence_excerpt"]) < 8:
            row["evidence_excerpt"].append(str(evidence)[:400])

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

    service_health = structured.get("service_health", {}) or {}
    if isinstance(service_health, dict):
        for svc, h in service_health.items():
            if not isinstance(h, dict) or _is_non_root_service(str(svc)):
                continue
            status = str(h.get("status") or "")
            reasons = [str(x) for x in (h.get("reasons", []) or [])]
            text = " ".join([status] + reasons)
            ft = _suggest_fault_from_text(text)
            has_typed = _has_any(text, _INFRA_TOKENS + _AUTH_TOKENS + _CONFIG_TOKENS + _LATENCY_TOKENS + _NETWORK_TOKENS)
            score = 1.25 if not has_typed else 8.0 + _text_signal_score(text)
            direct_health_rows.append({
                "service": str(svc),
                "status": status,
                "reasons": reasons[:8],
                "suggested_fault_type": ft,
                "root_cause_signal_score": round(score, 3),
                "source": "agent_input_builder.service_health",
            })
            add_candidate(str(svc), ft, score, "typed_service_health" if has_typed else "generic_degraded_service_health")
            for group, toks in [("infra", _INFRA_TOKENS), ("auth", _AUTH_TOKENS), ("config", _CONFIG_TOKENS), ("latency", _LATENCY_TOKENS), ("network", _NETWORK_TOKENS), ("error", _ERROR_TOKENS)]:
                c = _token_count(text, toks)
                if c:
                    signal(str(svc), group, c, text)
            if not has_typed:
                signal(str(svc), "error", 1, text or "generic degraded service health")

    for svc in signature.get("degraded_services", []) or []:
        if str(svc) in service_set and not _is_non_root_service(str(svc)):
            add_candidate(str(svc), "dependency_failure", 0.75, "symptom_signature_degraded")
            signal(str(svc), "error", 1, "symptom_signature_degraded")
    for svc in signature.get("metric_anomaly_services", []) or []:
        if str(svc) in service_set:
            add_candidate(str(svc), "resource_exhaustion", 2.5, "metric_anomaly_service")
            signal(str(svc), "infra", 1, "metric_anomaly_service")

    for item in top_error_services:
        svc = str(item.get("service") or "")
        if not svc or _is_non_root_service(svc):
            continue
        text = json.dumps(item, sort_keys=True, default=str)
        ft = _suggest_fault_from_text(text)
        cnt = _safe_float(item.get("error_count"), 1.0)
        explicit = ft != "dependency_failure"
        score = (2.5 if explicit else 0.5) + min(2.0, 0.02 * cnt) + min(3.0, _text_signal_score(text) * 0.15)
        add_candidate(svc, ft, score, "explicit_top_log_error_service" if explicit else "generic_top_log_error_service")
        signal(svc, _fault_group(ft), max(1.0, min(4.0, cnt)), text)

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
            add_candidate(parts[-1], ft, 1.5, "suspicious_trace_target")
            add_candidate(parts[0], "dependency_failure", 0.25, "suspicious_trace_source")
            signal(parts[-1], _fault_group(ft), 2, text)
            signal(parts[0], "error", 1, text)

    for svc in _services_from_sla(structured.get("sla", {})):
        add_candidate(svc, "dependency_failure", 0.5, "sla_unhealthy_service")
        signal(svc, "error", 1, "sla unhealthy service")

    clusters = structured.get("service_clusters", {}) or {}
    if isinstance(clusters, dict):
        for bucket, vals in clusters.items():
            if not isinstance(vals, list):
                continue
            ft = _suggest_fault_from_text(str(bucket))
            for svc in vals:
                add_candidate(str(svc), ft, 0.2, f"weak_cluster:{bucket}")
                signal(str(svc), _fault_group(ft), 0.25, f"cluster:{bucket}")

    # Strong typed family generation. This does not use oracle labels; it expands
    # plausible typed hypotheses so recoverability is not blocked by missing fault type.
    if explicit_auth:
        for svc in services:
            if _is_db_service(svc):
                local = _service_local_text(structured, svc)
                score = 8.0
                if _has_any(local, _AUTH_TOKENS):
                    score += 4.0
                if svc in top_error_names:
                    score += 2.0
                if service_mentions.get(svc, 0) > 6:
                    score += 0.5
                add_candidate(svc, "auth_failure", score, "v3_auth_db_hypothesis")
                signal(svc, "auth", 3, local[:400] or "global auth evidence")

    if explicit_config:
        for svc in services:
            if _is_db_or_cache_service(svc) or _is_helper_service(svc):
                continue
            local = _service_local_text(structured, svc)
            score = 4.0
            if _has_any(local, _CONFIG_TOKENS):
                score += 4.0
            if svc in top_error_names:
                score += 2.0
            if svc in (signature.get("degraded_services", []) or []):
                score += 1.0
            add_candidate(svc, "config_error", score, "v3_config_service_hypothesis")
            signal(svc, "config", 2, local[:400] or "global config evidence")

    if explicit_infra:
        for svc in services:
            local = _service_local_text(structured, svc)
            if _has_any(local, _INFRA_TOKENS):
                add_candidate(svc, "infra_failure", 4.0 + _text_signal_score(local) * 0.2, "v3_local_infra_hypothesis")
                signal(svc, "infra", 2, local[:400])
            # If only the DB side exposes the text, add the paired business service too.
            business = _paired_business_service(svc, service_set)
            if business and _is_db_or_cache_service(svc):
                local_db = _service_local_text(structured, svc)
                if _has_any(local_db + " " + full_text[:20000], _INFRA_TOKENS):
                    add_candidate(business, "infra_failure", 9.5, "v3_paired_business_over_db_for_infra")
                    signal(business, "infra", 2, local_db[:400] or "paired db infra evidence")

    if explicit_network or explicit_latency:
        ft = "network_failure" if explicit_network else "latency_degradation"
        for svc in services:
            local = _service_local_text(structured, svc)
            if _has_any(local, _NETWORK_TOKENS if explicit_network else _LATENCY_TOKENS):
                add_candidate(svc, ft, 5.0, f"v3_local_{ft}_hypothesis")
                signal(svc, _fault_group(ft), 2, local[:400])
            business = _paired_business_service(svc, service_set)
            if business and _is_db_or_cache_service(svc):
                local_db = _service_local_text(structured, svc)
                if _has_any(local_db + " " + full_text[:20000], _NETWORK_TOKENS if explicit_network else _LATENCY_TOKENS):
                    add_candidate(business, ft, 7.5, f"v3_paired_business_over_db_for_{ft}")
                    signal(business, _fault_group(ft), 2, local_db[:400] or f"paired db {ft} evidence")

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
            score -= 1.5
            reasons.append("frontend_penalty")
        if _is_cache_service(svc) and ft == "dependency_failure":
            score -= 2.5
            reasons.append("cache_dependency_fanout_penalty")
        if _is_db_service(svc) and ft == "infra_failure" and explicit_auth:
            score -= 4.0
            reasons.append("auth_incident_db_infra_penalty")
        if _is_db_or_cache_service(svc) and ft in {"infra_failure", "network_failure", "latency_degradation"}:
            business = _paired_business_service(svc, service_set)
            if business:
                score -= 2.5
                reasons.append("paired_business_preferred_for_non_auth_non_config")
        if _is_db_service(svc) and ft == "dependency_failure" and not explicit_auth:
            score -= 1.5
            reasons.append("generic_db_dependency_penalty")
        if ft == "dependency_failure" and reasons and all(r.startswith("weak_cluster") or r in {"symptom_signature_degraded", "sla_unhealthy_service"} for r in reasons):
            score -= 2.5
            reasons.append("cluster_only_dependency_penalty")
        rows.append({**row, "fault_type": ft, "score": round(score, 6), "reasons": reasons})

    rows.sort(key=lambda r: (float(r.get("score", 0.0)), _specificity_tiebreak_v3(str(r.get("service", "")), str(r.get("fault_type", "")))), reverse=True)

    signal_list = []
    for row in signal_rows.values():
        row["score"] = round(sum(float(v) for v in row.get("signal_counts", {}).values()), 6)
        signal_list.append(row)
    signal_list.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    return {
        "extractor_version": "agent_input_builder_v3_family_aware_candidates",
        "candidate_root_causes": rows[:100],
        "direct_health_services": sorted(direct_health_rows, key=lambda x: float(x.get("root_cause_signal_score", 0.0)), reverse=True)[:80],
        "signal_by_service": signal_list[:80],
        "service_mention_counts": service_mentions.most_common(120),
        "observed_services_sample": services[:220],
        "top_log_error_services": top_error_services[:16],
        "suspicious_trace_edges": structured.get("suspicious_trace_edges", [])[:16],
        "trace_summary": structured.get("trace_summary", {}),
        "sla_excerpt": _slim_sla(structured.get("sla", {})),
        "global_signal_flags": {
            "explicit_auth": explicit_auth,
            "explicit_config": explicit_config,
            "explicit_infra": explicit_infra,
            "explicit_latency": explicit_latency,
            "explicit_network": explicit_network,
        },
        "builder_source": "state_abstraction_full.agent_input_builder.build_rca_agent_input + v3 family-aware candidate expansion",
        "invalid_service_examples": list(_NON_ROOT_SERVICE_PATTERNS),
    }


class EvidenceFirstLLMRCASolver:
    """LLM-backed fixed RCA solver with evidence-first v3 candidate guard.

    The API call is still made, but final output is allowed to follow the v3
    evidence-ranked candidate when the LLM gives a valid-but-weak cascade answer.
    """

    def __init__(self, provider: str = "openai", model: str | None = None, max_tokens: int = 300, temperature: float = 0.0, state_char_budget: int = 24000, cache_path: str | None = None, max_root_causes: int = 1):
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
        compact = compact_state_for_llm_v3(compressed_state, char_budget=self.state_char_budget)
        cache_key = _stable_hash({
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_root_causes": self.max_root_causes,
            "solver": "evidence_first_v3",
            "instruction": instruction,
            "state_hash": _stable_hash(compact),
        })
        if cache_key in self._cache:
            return self._cache[cache_key]

        evidence = compact.get("high_signal_evidence") if isinstance(compact, dict) else {}
        candidates = evidence.get("candidate_root_causes", []) if isinstance(evidence, dict) else []
        valid_services = set(str(x) for x in compact.get("valid_services", []) if x)
        user_prompt = self._build_user_prompt(compact, instruction)
        raw = self.client.call(system_prompt=_SYSTEM_PROMPT_V3, user_prompt=user_prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        sanitized_unlimited = sanitize_rca_prediction(raw, compressed_state, max_root_causes=None)
        cleaned = sanitize_rca_prediction(raw, compressed_state, max_root_causes=self.max_root_causes)
        final, repair_reason = repair_prediction_evidence_first_v3(cleaned, valid_services, candidates)

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
            "valid_services": sorted(valid_services)[:220],
            "instruction": instruction,
            "raw_response": raw,
            "sanitized_unlimited": sanitized_unlimited,
            "sanitized_prediction": cleaned,
            "final_prediction_after_validity_guard": final,
            "postprocess_mode": "evidence_first_candidate_guard_v3",
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
            "candidate_root_causes": (compact_state.get("high_signal_evidence") or {}).get("candidate_root_causes", [])[:25],
            "root_cause_selection_guidance": [
                "Treat the v3 candidate ranking as the primary evidence shortlist.",
                "Do not choose broad fan-out victims or cluster-only dependency candidates.",
                "For auth incidents, choose auth_failure on the DB service with local/top-error evidence.",
                "For target-port/app config incidents, choose config_error on the affected business service.",
                "For killed/unready/scheduling incidents, choose infra_failure on the business service, not the paired DB unless DB has direct evidence.",
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
        row = {"key": key, "provider": self.provider, "model": self.model_name, "max_root_causes": self.max_root_causes, "prediction": prediction, "raw_response": raw_response, "solver": "evidence_first_v3"}
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _append_debug(self, row: dict[str, Any]) -> None:
        if not self.debug_path:
            return
        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def repair_prediction_evidence_first_v3(prediction: str, valid_services: set[str], candidates: list[dict[str, Any]]) -> tuple[str, str]:
    best = _best_valid_candidate_v3(candidates, valid_services)
    parsed = parse_fault_lines(prediction)
    if not best:
        return (prediction or "unknown::unknown", "no_candidate_available")
    best_key = _candidate_key(best)
    best_score = _safe_float(best.get("score"))
    if not parsed:
        return best_key, "fallback_no_parse_to_best_v3_candidate"

    label = parsed[0]
    pred_key = label.canonical_key()
    pred_service = label.service
    pred_ft = normalize_fault_type(label.fault_type)
    if (valid_services and pred_service not in valid_services) or _is_non_root_service(pred_service):
        return best_key, "fallback_invalid_service_to_best_v3_candidate"

    pred_score = _candidate_score_for(pred_service, pred_ft, candidates)
    if pred_key == best_key:
        return pred_key, "llm_agreed_with_best_v3_candidate"
    if pred_score >= max(6.0, best_score - 1.0):
        return pred_key, "kept_high_scoring_llm_candidate"
    return best_key, "overrode_weak_llm_prediction_with_best_v3_candidate"


def _best_valid_candidate_v3(candidates: list[dict[str, Any]], valid_services: set[str]) -> dict[str, Any] | None:
    for row in candidates or []:
        svc = str(row.get("service") or "")
        ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
        if svc and ft != "dependency_failure" and (not valid_services or svc in valid_services) and not _is_non_root_service(svc):
            return row
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


def _looks_like_service_v3(token: str) -> bool:
    if _looks_like_service(token):
        return True
    return str(token or "").strip().lower() in BUSINESS_SINGLE_WORD_SERVICES


def _paired_business_service(service: str, service_set: set[str]) -> str | None:
    s = str(service or "").lower()
    candidates = []
    if s.startswith("mongodb-"):
        candidates.append(s.replace("mongodb-", "", 1))
    if s.startswith("memcached-"):
        candidates.append(s.replace("memcached-", "", 1))
    if s.endswith("-db"):
        candidates.append(s[:-3])
    if s.endswith("-mongo"):
        candidates.append(s[:-6])
    if "-mongodb" in s:
        candidates.append(s.split("-mongodb", 1)[0])
    if "-redis" in s:
        candidates.append(s.split("-redis", 1)[0])
    for c in candidates:
        c = c.replace("hotel-reserv-", "")
        if c in service_set and not _is_db_or_cache_service(c):
            return c
    return None


def _specificity_tiebreak_v3(service: str, fault_type: str) -> float:
    s = service.lower()
    ft = normalize_fault_type(fault_type)
    score = _specificity_tiebreak(service, fault_type)
    if ft != "dependency_failure":
        score += 1.0
    if s in BUSINESS_SINGLE_WORD_SERVICES and ft in {"infra_failure", "network_failure", "latency_degradation", "config_error"}:
        score += 1.5
    if _is_db_service(s) and ft == "auth_failure":
        score += 2.0
    if _is_db_or_cache_service(s) and ft in {"infra_failure", "network_failure", "latency_degradation"}:
        score -= 2.0
    return score


_SYSTEM_PROMPT_V3 = """\
You are a fixed RCA solver for Kubernetes/AIOps incidents.
Use only the redacted evidence. Output only service::fault_type lines.
Prefer the v3 candidate ranking unless there is clear contrary evidence.
Never output explanations, markdown, JSON, app names, nodes, helper services, or namespaces.
"""
