from __future__ import annotations

from typing import Any

from .llm_rca_solver import _is_non_root_service, _stable_hash
from .rca_candidate_generator_v5 import compact_state_for_llm_v5
from .schemas import normalize_fault_type


class CandidateSweepRCASolver:
    """Deterministic verifier-guided candidate sweep solver.

    This solver is for offline training/debug, not final inference. It consumes
    the high-recall v5 candidate set and returns a different candidate or
    candidate combination on each call for the same redacted state. With GRPO
    group_size > 1 and selection_strategy=best, the verifier can identify
    whether the correct RCA is reachable from the redacted candidate universe and
    produce useful contrastive samples.

    v2 adds multi-root candidate combinations. This is important because many
    AIOpsLab mixed scenarios contain two faults, while v1 only emitted one
    service::fault_type line and therefore could not succeed on exact-set RCA.
    """

    PAIR_TEMPLATES: tuple[tuple[str, str], ...] = (
        ("config_error", "auth_failure"),
        ("auth_failure", "config_error"),
        ("auth_failure", "latency_degradation"),
        ("latency_degradation", "auth_failure"),
        ("auth_failure", "network_failure"),
        ("network_failure", "auth_failure"),
        ("auth_failure", "dependency_failure"),
        ("dependency_failure", "auth_failure"),
        ("config_error", "latency_degradation"),
        ("latency_degradation", "config_error"),
        ("config_error", "network_failure"),
        ("network_failure", "config_error"),
        ("infra_failure", "auth_failure"),
        ("auth_failure", "infra_failure"),
        ("infra_failure", "config_error"),
        ("config_error", "infra_failure"),
    )

    def __init__(
        self,
        max_root_causes: int = 1,
        start_index: int = 0,
        stride: int = 1,
        include_dependency: bool | None = None,
        pair_pool_per_type: int = 20,
        sliding_pair_pool: int = 40,
    ):
        self.max_root_causes = max(1, int(max_root_causes))
        self.start_index = max(0, int(start_index))
        self.stride = max(1, int(stride))
        # dependency_failure is needed for user_unregistered_mongodb and some
        # hotel multifault cases. Keep it off for single-root sweeps to avoid
        # noisy dependency guesses, but enable it automatically for multi-root.
        self.include_dependency = self.max_root_causes > 1 if include_dependency is None else bool(include_dependency)
        self.pair_pool_per_type = max(1, int(pair_pool_per_type))
        self.sliding_pair_pool = max(2, int(sliding_pair_pool))
        self._state_counts: dict[str, int] = {}

    def solve(self, compressed_state: dict[str, Any], instruction: str) -> str:
        compact = compact_state_for_llm_v5(compressed_state)
        evidence = compact.get("high_signal_evidence", {}) if isinstance(compact, dict) else {}
        valid_services = set(str(x) for x in compact.get("valid_services", []) if x) if isinstance(compact, dict) else set()
        candidates = evidence.get("candidate_root_causes", []) if isinstance(evidence, dict) else []
        filtered = self._filter_candidates(candidates, valid_services)
        if not filtered:
            return "unknown::unknown"

        state_key = _stable_hash({
            "compact_state": compact,
            "solver": "candidate_sweep_v2",
            "max_root_causes": self.max_root_causes,
            "include_dependency": self.include_dependency,
        })
        call_index = self._state_counts.get(state_key, 0)
        self._state_counts[state_key] = call_index + 1

        plan = self._build_plan(filtered)
        if not plan:
            return "unknown::unknown"
        idx = (self.start_index + call_index * self.stride) % len(plan)
        return "\n".join(plan[idx])

    def _build_plan(self, filtered: list[dict[str, Any]]) -> list[list[str]]:
        singles = [self._candidate_key(row) for row in filtered]
        singles = self._dedupe_lines(singles)
        if self.max_root_causes == 1:
            return [[x] for x in singles]

        rows_by_type: dict[str, list[dict[str, Any]]] = {}
        for row in filtered:
            ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
            rows_by_type.setdefault(ft, []).append(row)

        plan: list[list[str]] = []
        seen_sets: set[tuple[str, ...]] = set()

        def add(lines: list[str]) -> None:
            lines = self._dedupe_lines(lines)[: self.max_root_causes]
            if len(lines) < min(2, self.max_root_causes):
                return
            key = tuple(sorted(lines))
            if key in seen_sets:
                return
            seen_sets.add(key)
            plan.append(lines)

        # High-priority typed multifault templates. These cover the synthetic
        # mixed cases: config+auth, auth+network/latency, auth+dependency, etc.
        for ft_a, ft_b in self.PAIR_TEMPLATES:
            rows_a = rows_by_type.get(ft_a, [])[: self.pair_pool_per_type]
            rows_b = rows_by_type.get(ft_b, [])[: self.pair_pool_per_type]
            for a in rows_a:
                for b in rows_b:
                    ka = self._candidate_key(a)
                    kb = self._candidate_key(b)
                    if ka == kb:
                        continue
                    add([ka, kb])

        # Same-fault pairs are lower priority but useful for double DB/auth or
        # repeated infra/data-store failures.
        for ft in ("auth_failure", "config_error", "infra_failure", "network_failure", "latency_degradation", "dependency_failure"):
            rows = rows_by_type.get(ft, [])[: self.pair_pool_per_type]
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    add([self._candidate_key(a), self._candidate_key(b)])

        # Rank-window fallback: enumerate pairs among the top visible candidates
        # so that unusual typed combinations can still be tested.
        top = filtered[: self.sliding_pair_pool]
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                add([self._candidate_key(a), self._candidate_key(b)])

        # Add singles at the end as a fallback. They are not exact for multifault,
        # but help diagnose whether at least one component is recoverable.
        for s in singles:
            add([s, s])  # ignored by add because it dedupes to length 1
        if not plan:
            return [[s] for s in singles]
        return plan

    def _filter_candidates(self, candidates: list[dict[str, Any]], valid_services: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
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
            if ft in {""}:
                continue
            key = f"{svc}::{ft}"
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, "service": svc, "fault_type": ft})
        return rows

    @staticmethod
    def _candidate_key(row: dict[str, Any]) -> str:
        return f"{row.get('service')}::{normalize_fault_type(str(row.get('fault_type') or 'unknown'))}"

    @staticmethod
    def _dedupe_lines(lines: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for line in lines:
            line = str(line).strip()
            if not line or line in seen:
                continue
            seen.add(line)
            out.append(line)
        return out
