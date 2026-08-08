from __future__ import annotations

from typing import Any

from .llm_rca_solver import _is_non_root_service, _stable_hash
from .rca_candidate_generator_v5 import compact_state_for_llm_v5
from .schemas import normalize_fault_type


class CandidateSweepRCASolver:
    """Deterministic verifier-guided candidate sweep solver.

    This solver is for offline training/debug, not final inference. It consumes
    the high-recall v5 candidate set and returns a different candidate on each
    call for the same redacted state. With GRPO group_size > 1 and
    selection_strategy=best, the verifier can identify whether the correct RCA is
    reachable from the redacted candidate universe and produce useful contrastive
    samples.
    """

    def __init__(self, max_root_causes: int = 1, start_index: int = 0, stride: int = 1, include_dependency: bool = False):
        self.max_root_causes = max(1, int(max_root_causes))
        self.start_index = max(0, int(start_index))
        self.stride = max(1, int(stride))
        self.include_dependency = bool(include_dependency)
        self._state_counts: dict[str, int] = {}

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        compact = compact_state_for_llm_v5(compressed_state)
        evidence = compact.get("high_signal_evidence", {}) if isinstance(compact, dict) else {}
        valid_services = set(str(x) for x in compact.get("valid_services", []) if x) if isinstance(compact, dict) else set()
        candidates = evidence.get("candidate_root_causes", []) if isinstance(evidence, dict) else []
        filtered = self._filter_candidates(candidates, valid_services)
        if not filtered:
            return "unknown::unknown"

        state_key = _stable_hash({"compact_state": compact, "solver": "candidate_sweep_v1"})
        call_index = self._state_counts.get(state_key, 0)
        self._state_counts[state_key] = call_index + 1

        base = (self.start_index + call_index * self.stride) % len(filtered)
        chosen = []
        seen_keys = set()
        for offset in range(len(filtered)):
            row = filtered[(base + offset) % len(filtered)]
            key = self._candidate_key(row)
            if key not in seen_keys:
                chosen.append(key)
                seen_keys.add(key)
            if len(chosen) >= self.max_root_causes:
                break
        return "\n".join(chosen) if chosen else "unknown::unknown"

    def _filter_candidates(self, candidates: list[dict[str, Any]], valid_services: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in candidates or []:
            if not isinstance(row, dict):
                continue
            svc = str(row.get("service") or "").strip()
            ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
            if not svc or _is_non_root_service(svc):
                continue
            if valid_services and svc not in valid_services:
                continue
            if ft == "dependency_failure" and not self.include_dependency:
                continue
            if ft in {"", "dependency_failure"}:
                continue
            rows.append({**row, "service": svc, "fault_type": ft})
        return rows

    @staticmethod
    def _candidate_key(row: dict[str, Any]) -> str:
        return f"{row.get('service')}::{normalize_fault_type(str(row.get('fault_type') or 'unknown'))}"
