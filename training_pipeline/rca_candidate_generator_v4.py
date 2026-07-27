from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .schemas import normalize_fault_type, parse_fault_lines
from .llm_rca_solver import (
    _AUTH_TOKENS,
    _CONFIG_TOKENS,
    _ERROR_TOKENS,
    _INFRA_TOKENS,
    _LATENCY_TOKENS,
    _NETWORK_TOKENS,
    _fault_group,
    _has_any,
    _is_db_or_cache_service,
    _is_db_service,
    _is_helper_service,
    _is_non_root_service,
    _safe_float,
    _service_local_text,
    _specificity_tiebreak,
    _stable_hash,
    sanitize_rca_prediction,
)
from .rca_candidate_generator_v3 import (
    BUSINESS_SINGLE_WORD_SERVICES,
    EvidenceFirstLLMRCASolver,
    compact_state_for_llm_v3,
    _candidate_key,
    _candidate_score_for,
    _looks_like_service_v3,
    _paired_business_service,
)

# v4 deliberately separates broad/global words from local evidence. In v3,
# global tokens such as "network" or "connection refused" caused network/config
# hypotheses to be added to many unrelated services. These token sets are used
# only for local-text confirmation and conservative backstops.
_CONFIG_STRONG_TOKENS = tuple(dict.fromkeys(_CONFIG_TOKENS + (
    "misconfig", "misconfigured", "wrong binary", "wrong-bin", "wrong_bin",
    "wrong bin", "bad config", "invalid config", "env var", "environment",
)))
_NETWORK_STRONG_TOKENS = (
    "packet loss", "network loss", "network_loss", "loss_", "unreachable",
    "no route", "dns", "reset by peer", "dropped", "drop", "packet",
)
_LATENCY_STRONG_TOKENS = (
    "delay", "latency", "slow", "timeout", "timed out", "p99", "p95", "delay_",
)
_INFRA_STRONG_TOKENS = tuple(dict.fromkeys(_INFRA_TOKENS + (
    "replicas", "replica", "scale", "scaled", "available replicas", "desired replicas",
)))

_SYSTEM_PROMPT_V4 = """\
You are a fixed RCA solver for Kubernetes/AIOps incidents.

You receive redacted RCA evidence and a v4 locally-reranked candidate shortlist.
Generated scenario IDs, ground truth, raw specs, and faulty-service labels are removed.

Output ONLY service::fault_type lines. No prose, markdown, JSON, or bullets.
Use one of the valid services and one canonical fault type.

RCA rules:
- Prefer locally supported v4 candidates over broad cascade symptoms.
- Do not treat every MongoDB/service with the same global token as root cause.
- Auth requires local DB/auth evidence or a DB service with strong service/log support.
- Config requires local target-port/wrong-bin/misconfig evidence, or a business-service config backstop when app-level misconfig is visible.
- Network/latency requires local network-loss or delay evidence; do not convert connection-refused fanout into network root cause.
- Infra requires local pod/replica/unready/scheduling/kill/scale evidence, or a degraded business service during an explicit infra incident.
"""


def compact_state_for_llm_v4(compressed_state: dict[str, Any], char_budget: int = 24000) -> dict[str, Any]:
    compact = compact_state_for_llm_v3(compressed_state, char_budget=char_budget)
    structured = compact.get("rca_agent_structured_evidence", {}) if isinstance(compact, dict) else {}
    evidence = compact.get("high_signal_evidence", {}) if isinstance(compact, dict) else {}
    compact["high_signal_evidence"] = high_signal_evidence_summary_v4(structured, evidence)
    # Ensure v4-added single-word hotel services stay visible.
    valid = set(str(x) for x in compact.get("valid_services", []) if x)
    valid.update(compact["high_signal_evidence"].get("observed_services_sample", []) or [])
    compact["valid_services"] = sorted(s for s in valid if _looks_like_service_v3(s) and not _is_non_root_service(s))[:240]
    return compact


def high_signal_evidence_summary_v4(structured: dict[str, Any], base_evidence: dict[str, Any]) -> dict[str, Any]:
    structured = structured if isinstance(structured, dict) else {}
    base_evidence = base_evidence if isinstance(base_evidence, dict) else {}
    services = sorted({str(s) for s in (base_evidence.get("observed_services_sample") or structured.get("all_services") or []) if _looks_like_service_v3(str(s)) and not _is_non_root_service(str(s))})
    service_set = set(services)
    top_error_services = [x for x in (structured.get("top_error_services", []) or []) if isinstance(x, dict)]
    top_error_names = {str(x.get("service") or "") for x in top_error_services}
    health = structured.get("service_health", {}) if isinstance(structured.get("service_health", {}), dict) else {}
    service_clusters = structured.get("service_clusters", {}) if isinstance(structured.get("service_clusters", {}), dict) else {}
    full_text = json.dumps(structured, sort_keys=True, default=str).lower()

    explicit_auth = _has_any(full_text, _AUTH_TOKENS)
    explicit_config = _has_any(full_text, _CONFIG_STRONG_TOKENS)
    explicit_infra = _has_any(full_text, _INFRA_STRONG_TOKENS)
    explicit_latency = _has_any(full_text, _LATENCY_STRONG_TOKENS)
    explicit_network = _has_any(full_text, _NETWORK_STRONG_TOKENS)

    mention_counts = Counter({str(k): int(v) for k, v in (base_evidence.get("service_mention_counts") or []) if isinstance(k, str)})
    signal_score = {}
    for row in base_evidence.get("signal_by_service", []) or []:
        if isinstance(row, dict) and row.get("service"):
            signal_score[str(row["service"])] = _safe_float(row.get("score"))

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def add_candidate(svc: str, fault_type: str, score: float, reason: str) -> None:
        svc = str(svc or "").strip()
        if not svc or _is_non_root_service(svc):
            return
        ft = normalize_fault_type(fault_type)
        key = (svc, ft)
        row = rows_by_key.setdefault(key, {"service": svc, "fault_type": ft, "score": 0.0, "reasons": []})
        row["score"] = round(float(row.get("score", 0.0)) + float(score), 6)
        if reason and reason not in row["reasons"]:
            row["reasons"].append(reason)

    for row in base_evidence.get("candidate_root_causes", []) or []:
        if not isinstance(row, dict):
            continue
        svc = str(row.get("service") or "").strip()
        ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
        if not svc or _is_non_root_service(svc):
            continue
        rows_by_key[(svc, ft)] = {
            "service": svc,
            "fault_type": ft,
            "score": _safe_float(row.get("score")),
            "reasons": list(row.get("reasons", []) or []),
        }

    degraded = set()
    for svc, item in health.items():
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").lower()
        if status and status not in {"healthy", "unknown", "ok"}:
            degraded.add(str(svc))
    for bucket, vals in service_clusters.items():
        if isinstance(vals, list) and any(tok in str(bucket).lower() for tok in ("unhealthy", "degraded", "infra", "error", "active")):
            degraded.update(str(x) for x in vals if _looks_like_service_v3(str(x)))

    # Local typed candidates. These are high-confidence and should beat broad/global expansions.
    for svc in services:
        local = _service_local_text(structured, svc)
        local_lower = local.lower()
        support = signal_score.get(svc, 0.0) * 0.12 + min(2.0, mention_counts.get(svc, 0) * 0.08)
        in_top_error = svc in top_error_names
        is_degraded = svc in degraded

        if _has_any(local_lower, _AUTH_TOKENS):
            add_candidate(svc, "auth_failure", 12.0 + support + (2.0 if in_top_error else 0.0), "v4_local_auth_evidence")
        if _has_any(local_lower, _CONFIG_STRONG_TOKENS):
            add_candidate(svc, "config_error", 12.0 + support + (2.0 if in_top_error else 0.0), "v4_local_config_evidence")
        if _has_any(local_lower, _INFRA_STRONG_TOKENS):
            add_candidate(svc, "infra_failure", 11.0 + support + (1.5 if is_degraded else 0.0), "v4_local_infra_evidence")
        if _has_any(local_lower, _LATENCY_STRONG_TOKENS):
            add_candidate(svc, "latency_degradation", 9.0 + support, "v4_local_latency_evidence")
        if _has_any(local_lower, _NETWORK_STRONG_TOKENS):
            add_candidate(svc, "network_failure", 9.0 + support, "v4_local_network_evidence")

        # Conservative recoverability backstops. These are intentionally lower
        # than local evidence, but prevent exact fault types from being absent.
        if explicit_infra and not _is_db_or_cache_service(svc) and (is_degraded or in_top_error or svc.endswith("-service") or svc in BUSINESS_SINGLE_WORD_SERVICES):
            add_candidate(svc, "infra_failure", 3.2 + support, "v4_business_infra_backstop")
        if explicit_config and not _is_db_or_cache_service(svc) and (is_degraded or in_top_error or svc.endswith("-service") or svc in BUSINESS_SINGLE_WORD_SERVICES):
            add_candidate(svc, "config_error", 3.0 + support, "v4_business_config_backstop")
        if explicit_latency and not _is_db_or_cache_service(svc) and (is_degraded or in_top_error or svc in BUSINESS_SINGLE_WORD_SERVICES):
            add_candidate(svc, "latency_degradation", 2.8 + support, "v4_business_latency_backstop")
        if explicit_network and not _is_db_or_cache_service(svc) and (is_degraded or in_top_error or svc in BUSINESS_SINGLE_WORD_SERVICES):
            add_candidate(svc, "network_failure", 2.8 + support, "v4_business_network_backstop")

        # Pair DB evidence back to the business service only when the DB has local
        # strong evidence. Do not use the entire global text as evidence.
        business = _paired_business_service(svc, service_set)
        if business and _is_db_or_cache_service(svc):
            if _has_any(local_lower, _INFRA_STRONG_TOKENS):
                add_candidate(business, "infra_failure", 7.5 + support, "v4_paired_local_db_infra_to_business")
            if _has_any(local_lower, _NETWORK_STRONG_TOKENS):
                add_candidate(business, "network_failure", 6.0 + support, "v4_paired_local_db_network_to_business")
            if _has_any(local_lower, _LATENCY_STRONG_TOKENS):
                add_candidate(business, "latency_degradation", 6.0 + support, "v4_paired_local_db_latency_to_business")

    # Hotel app-misconfig often surfaces as DB-side symptoms with little typed
    # config text after redaction. Add a low-rank config candidate for the visible
    # business services so exact candidates are recoverable without leaking oracle.
    hotel_business = [s for s in services if s in BUSINESS_SINGLE_WORD_SERVICES and s not in {"consul"}]
    if hotel_business and any(s.startswith("mongodb-") or s.endswith("-db") for s in services):
        for svc in hotel_business:
            add_candidate(svc, "config_error", 2.2 + min(1.0, mention_counts.get(svc, 0) * 0.05), "v4_hotel_business_config_recoverability_backstop")

    rows = []
    for row in rows_by_key.values():
        svc = str(row.get("service") or "")
        ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
        score = _safe_float(row.get("score"))
        reasons = list(row.get("reasons", []) or [])
        local = _service_local_text(structured, svc).lower()
        is_degraded = svc in degraded
        in_top_error = svc in top_error_names
        support = signal_score.get(svc, 0.0) * 0.05 + min(1.0, mention_counts.get(svc, 0) * 0.04)
        score += support

        if service_set and svc not in service_set:
            score -= 4.0
            reasons.append("v4_not_in_valid_services_penalty")
        if _is_helper_service(svc):
            score -= 10.0
            reasons.append("v4_helper_service_penalty")
        if _is_db_service(svc) and ft == "auth_failure":
            if _has_any(local, _AUTH_TOKENS):
                score += 4.0
                reasons.append("v4_db_local_auth_boost")
            elif in_top_error or signal_score.get(svc, 0.0) >= 4.0:
                score += 1.5
                reasons.append("v4_db_supported_auth_boost")
            else:
                score -= 4.5
                reasons.append("v4_global_only_auth_penalty")
        if ft == "config_error" and "v3_config_service_hypothesis" in reasons and not _has_any(local, _CONFIG_STRONG_TOKENS):
            if in_top_error or is_degraded:
                score -= 1.5
                reasons.append("v4_config_not_local_but_affected_penalty")
            else:
                score -= 5.0
                reasons.append("v4_global_only_config_penalty")
        if ft == "network_failure" and not _has_any(local, _NETWORK_STRONG_TOKENS):
            # Most v3 network candidates came from connection-refused or global
            # network words. Keep true local network-loss candidates high.
            score -= 6.5
            reasons.append("v4_no_local_network_loss_penalty")
        if ft == "latency_degradation" and not _has_any(local, _LATENCY_STRONG_TOKENS):
            score -= 3.5
            reasons.append("v4_no_local_latency_penalty")
        if ft == "infra_failure" and "v3_paired_business_over_db_for_infra" in reasons and not _has_any(local, _INFRA_STRONG_TOKENS) and not is_degraded:
            score -= 6.0
            reasons.append("v4_paired_infra_without_local_business_signal_penalty")
        if _is_db_or_cache_service(svc) and ft in {"infra_failure", "network_failure", "latency_degradation"}:
            business = _paired_business_service(svc, service_set)
            if business:
                score -= 3.0
                reasons.append("v4_prefer_business_over_db_for_non_auth")
        if svc == "frontend" and ft not in {"network_failure", "config_error", "unknown"}:
            score -= 2.0
            reasons.append("v4_frontend_non_specific_penalty")
        if ft == "dependency_failure" and reasons and all(str(r).startswith("weak_cluster") or r in {"symptom_signature_degraded", "sla_unhealthy_service"} for r in reasons):
            score -= 3.0
            reasons.append("v4_cluster_only_dependency_penalty")

        rows.append({"service": svc, "fault_type": ft, "score": round(score, 6), "reasons": reasons})

    rows.sort(key=lambda r: (float(r.get("score", 0.0)), _specificity_tiebreak(str(r.get("service", "")), str(r.get("fault_type", "")))), reverse=True)

    signal_list = list(base_evidence.get("signal_by_service", []) or [])
    direct_health = list(base_evidence.get("direct_health_services", []) or [])
    out = dict(base_evidence)
    out.update({
        "extractor_version": "agent_input_builder_v4_local_evidence_reranking",
        "candidate_root_causes": rows[:120],
        "direct_health_services": direct_health[:100],
        "signal_by_service": signal_list[:100],
        "service_mention_counts": Counter({str(k): int(v) for k, v in mention_counts.items()}).most_common(140),
        "observed_services_sample": services[:240],
        "global_signal_flags": {
            "explicit_auth": explicit_auth,
            "explicit_config": explicit_config,
            "explicit_infra": explicit_infra,
            "explicit_latency": explicit_latency,
            "explicit_network": explicit_network,
            "note": "v4 flags use stronger local-evidence-oriented token sets; connection-refused alone is not network_failure.",
        },
        "builder_source": "state_abstraction_full.agent_input_builder.build_rca_agent_input + v4 local-evidence reranking",
    })
    return out


class EvidenceFirstLLMRCASolverV4(EvidenceFirstLLMRCASolver):
    """LLM-backed fixed RCA solver using v4 local-evidence candidate ranking."""

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        compact = compact_state_for_llm_v4(compressed_state, char_budget=self.state_char_budget)
        cache_key = _stable_hash({
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_root_causes": self.max_root_causes,
            "solver": "evidence_first_v4",
            "instruction": instruction,
            "state_hash": _stable_hash(compact),
        })
        if cache_key in self._cache:
            return self._cache[cache_key]

        evidence = compact.get("high_signal_evidence") if isinstance(compact, dict) else {}
        candidates = evidence.get("candidate_root_causes", []) if isinstance(evidence, dict) else []
        valid_services = set(str(x) for x in compact.get("valid_services", []) if x)
        user_prompt = self._build_user_prompt_v4(compact, instruction)
        raw = self.client.call(system_prompt=_SYSTEM_PROMPT_V4, user_prompt=user_prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        sanitized_unlimited = sanitize_rca_prediction(raw, compressed_state, max_root_causes=None)
        cleaned = sanitize_rca_prediction(raw, compressed_state, max_root_causes=self.max_root_causes)
        final, repair_reason = repair_prediction_evidence_first_v4(cleaned, valid_services, candidates)

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
            "valid_services": sorted(valid_services)[:240],
            "instruction": instruction,
            "raw_response": raw,
            "sanitized_unlimited": sanitized_unlimited,
            "sanitized_prediction": cleaned,
            "final_prediction_after_validity_guard": final,
            "postprocess_mode": "evidence_first_candidate_guard_v4",
            "candidate_repair_reason": repair_reason,
            "high_signal_evidence": evidence,
            "candidate_root_causes": candidates,
        }
        self._cache[cache_key] = final
        self._append_cache(cache_key, final, raw)
        self._append_debug(debug)
        return final

    def _build_user_prompt_v4(self, compact_state: dict[str, Any], instruction: str) -> str:
        payload = {
            "policy_instruction": instruction,
            "root_cause_count_instruction": f"Return at most {self.max_root_causes} root cause line(s). Use fewer if only one upstream cause is supported.",
            "valid_services": compact_state.get("valid_services", []),
            "candidate_root_causes": (compact_state.get("high_signal_evidence") or {}).get("candidate_root_causes", [])[:30],
            "root_cause_selection_guidance": [
                "Treat v4 locally supported candidates as primary; avoid broad global-token candidates.",
                "Do not pick all MongoDB services from global auth/network text; require local or top-error support.",
                "Connection-refused fanout is not by itself network_failure.",
                "For config/target-port/app-misconfig, prefer business-service config candidates when locally supported.",
                "For infra/scale/pod issues, prefer non-DB business-service infra candidates with local/degraded support.",
            ],
            "redacted_rca_evidence": compact_state,
            "output_contract": "Return only service::fault_type lines. No prose.",
        }
        return json.dumps(payload, sort_keys=True, default=str)


def repair_prediction_evidence_first_v4(prediction: str, valid_services: set[str], candidates: list[dict[str, Any]]) -> tuple[str, str]:
    best = _best_valid_candidate_v4(candidates, valid_services)
    parsed = parse_fault_lines(prediction)
    if not best:
        return (prediction or "unknown::unknown", "no_candidate_available")
    best_key = _candidate_key(best)
    best_score = _safe_float(best.get("score"))
    if not parsed:
        return best_key, "fallback_no_parse_to_best_v4_candidate"

    label = parsed[0]
    pred_key = label.canonical_key()
    pred_service = label.service
    pred_ft = normalize_fault_type(label.fault_type)
    if (valid_services and pred_service not in valid_services) or _is_non_root_service(pred_service):
        return best_key, "fallback_invalid_service_to_best_v4_candidate"

    pred_score = _candidate_score_for(pred_service, pred_ft, candidates)
    if pred_key == best_key:
        return pred_key, "llm_agreed_with_best_v4_candidate"
    if pred_score >= max(6.0, best_score - 0.75):
        return pred_key, "kept_high_scoring_llm_candidate_v4"
    return best_key, "overrode_weak_llm_prediction_with_best_v4_candidate"


def _best_valid_candidate_v4(candidates: list[dict[str, Any]], valid_services: set[str]) -> dict[str, Any] | None:
    for row in candidates or []:
        svc = str(row.get("service") or "")
        ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
        if svc and ft not in {"dependency_failure", "unknown"} and (not valid_services or svc in valid_services) and not _is_non_root_service(svc):
            return row
    for row in candidates or []:
        svc = str(row.get("service") or "")
        if svc and (not valid_services or svc in valid_services) and not _is_non_root_service(svc):
            return row
    return None
