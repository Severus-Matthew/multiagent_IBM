from __future__ import annotations

import json
from typing import Any

from .schemas import normalize_fault_type, parse_fault_lines
from .llm_rca_solver import _is_non_root_service, _safe_float, _stable_hash, sanitize_rca_prediction
from .rca_candidate_generator_v3 import _candidate_key
from .rca_candidate_generator_v4 import EvidenceFirstLLMRCASolverV4, compact_state_for_llm_v4


_SYSTEM_PROMPT_V5 = """\
You are a fixed RCA solver for Kubernetes/AIOps incidents.

You receive redacted RCA evidence and a high-recall candidate list. Generated
scenario IDs, ground truth, raw specs, and faulty-service labels are removed.

Output ONLY service::fault_type lines. No prose, markdown, JSON, or bullets.
Use valid services and canonical fault types.

RCA rules:
- Treat the candidate list as a recall set, not as a strict score ranking.
- The true root cause may be lower ranked when cascade symptoms dominate logs.
- Prefer the service with the most direct local evidence for the fault type.
- Do not choose a downstream service only because it has the largest log fanout.
- For auth failures, prefer a MongoDB/DB service with local auth or DB-support evidence.
- For target-port/app-misconfig, prefer the affected business service with config evidence.
- For pod/scale/kill/scheduling issues, prefer the business service over generic DB fanout.
- For wrong-binary/unknown faults, output service::unknown when the service is clear but the canonical type is not.
"""


def compact_state_for_llm_v5(compressed_state: dict[str, Any], char_budget: int = 24000) -> dict[str, Any]:
    compact = compact_state_for_llm_v4(compressed_state, char_budget=char_budget)
    evidence = compact.get("high_signal_evidence") if isinstance(compact, dict) else {}
    if isinstance(evidence, dict):
        evidence = dict(evidence)
        evidence["extractor_version"] = "agent_input_builder_v5_llm_selector_over_v4_recall"
        evidence["selector_note"] = "Candidate scores are not treated as a hard ranking; the LLM may select a lower-ranked candidate if local evidence supports it."
        compact["high_signal_evidence"] = evidence
    return compact


class LLMSelectorRCASolverV5(EvidenceFirstLLMRCASolverV4):
    """LLM-backed RCA solver that keeps valid LLM-selected recall candidates.

    v4 achieved high exact-candidate recall but weak top-k ranking. This v5 guard
    therefore falls back to candidates only for invalid/no-parse output; it does
    not overwrite a valid LLM-selected candidate merely because another candidate
    has a higher heuristic score.
    """

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        compact = compact_state_for_llm_v5(compressed_state, char_budget=self.state_char_budget)
        cache_key = _stable_hash({
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_root_causes": self.max_root_causes,
            "solver": "llm_selector_v5",
            "instruction": instruction,
            "state_hash": _stable_hash(compact),
        })
        if cache_key in self._cache:
            return self._cache[cache_key]

        evidence = compact.get("high_signal_evidence") if isinstance(compact, dict) else {}
        candidates = evidence.get("candidate_root_causes", []) if isinstance(evidence, dict) else []
        valid_services = set(str(x) for x in compact.get("valid_services", []) if x)
        user_prompt = self._build_user_prompt_v5(compact, instruction)
        raw = self.client.call(system_prompt=_SYSTEM_PROMPT_V5, user_prompt=user_prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        sanitized_unlimited = sanitize_rca_prediction(raw, compressed_state, max_root_causes=None)
        cleaned = sanitize_rca_prediction(raw, compressed_state, max_root_causes=self.max_root_causes)
        final, repair_reason = repair_prediction_llm_selector_v5(
            cleaned,
            valid_services,
            candidates,
            max_root_causes=self.max_root_causes,
        )

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
            "postprocess_mode": "llm_selector_candidate_guard_v5",
            "candidate_repair_reason": repair_reason,
            "high_signal_evidence": evidence,
            "candidate_root_causes": candidates,
        }
        self._cache[cache_key] = final
        self._append_cache(cache_key, final, raw)
        self._append_debug(debug)
        return final

    def _build_user_prompt_v5(self, compact_state: dict[str, Any], instruction: str) -> str:
        evidence = compact_state.get("high_signal_evidence") or {}
        payload = {
            "policy_instruction": instruction,
            "root_cause_count_instruction": f"Return at most {self.max_root_causes} root cause line(s). Use fewer if only one upstream cause is supported.",
            "valid_services": compact_state.get("valid_services", []),
            "candidate_root_causes": evidence.get("candidate_root_causes", [])[:80],
            "root_cause_selection_guidance": [
                "Candidate scores are heuristic; do not blindly pick rank 1.",
                "Select the candidate whose service and fault type have the strongest local evidence.",
                "Cascade log errors and connection-refused fanout are weak unless they identify the upstream broken service.",
                "For multi-fault evidence, output multiple lines, up to the allowed root-cause count.",
                "For wrong-binary/unclear application faults, service::unknown is acceptable when the service is clear.",
            ],
            "redacted_rca_evidence": compact_state,
            "output_contract": "Return only service::fault_type lines. No prose.",
        }
        return json.dumps(payload, sort_keys=True, default=str)


def repair_prediction_llm_selector_v5(
    prediction: str,
    valid_services: set[str],
    candidates: list[dict[str, Any]],
    max_root_causes: int = 1,
) -> tuple[str, str]:
    parsed = parse_fault_lines(prediction)
    candidate_keys = {_candidate_key(row) for row in candidates or [] if isinstance(row, dict)}
    candidate_services = {str(row.get("service") or "") for row in candidates or [] if isinstance(row, dict)}

    kept: list[str] = []
    rejected_invalid = 0
    for label in parsed:
        svc = str(label.service or "")
        ft = normalize_fault_type(label.fault_type)
        key = f"{svc}::{ft}"
        if (valid_services and svc not in valid_services) or _is_non_root_service(svc):
            rejected_invalid += 1
            continue
        # Prefer predictions that are in the high-recall candidate set. Allow
        # service::unknown for wrong-binary/unclear application faults even when
        # the candidate generator only produced typed variants.
        if key in candidate_keys or svc in candidate_services or ft == "unknown":
            if key not in kept:
                kept.append(key)
        else:
            rejected_invalid += 1

    if kept:
        return "\n".join(kept[: max(1, int(max_root_causes))]), "kept_valid_llm_selected_candidate_v5"

    fallback = _fallback_candidate_lines_v5(candidates, valid_services, max_root_causes=max_root_causes)
    if fallback:
        reason = "fallback_no_parse_to_v5_candidates" if not parsed else "fallback_invalid_or_out_of_recall_to_v5_candidates"
        if rejected_invalid:
            reason += f"_rejected_{rejected_invalid}"
        return fallback, reason
    return (prediction or "unknown::unknown", "no_valid_prediction_or_candidate_v5")


def _fallback_candidate_lines_v5(candidates: list[dict[str, Any]], valid_services: set[str], max_root_causes: int = 1) -> str:
    out: list[str] = []
    seen_services: set[str] = set()
    for row in candidates or []:
        if not isinstance(row, dict):
            continue
        svc = str(row.get("service") or "")
        ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
        if not svc or _is_non_root_service(svc) or (valid_services and svc not in valid_services):
            continue
        if ft == "dependency_failure":
            continue
        if svc in seen_services and len(out) >= 1:
            continue
        key = f"{svc}::{ft}"
        if key not in out:
            out.append(key)
            seen_services.add(svc)
        if len(out) >= max(1, int(max_root_causes)):
            break
    return "\n".join(out)
