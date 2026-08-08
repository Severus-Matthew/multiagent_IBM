from __future__ import annotations

from typing import Any

from .llm_rca_solver import _is_db_or_cache_service, _is_non_root_service, _stable_hash
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

    v2 added multi-root candidate combinations. v4 keeps the v2 high-precision
    ordering first and moves diagnostic fault-family variants to a late fallback
    section. This avoids the v3 regression where network/dependency variants
    displaced the exact config+auth pairs from the sampled window.
    """

    BASE_PAIR_TEMPLATES: tuple[tuple[str, str], ...] = (
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

    DIAGNOSTIC_PAIR_TEMPLATES: tuple[tuple[str, str], ...] = (
        ("auth_failure", "network_failure"),
        ("network_failure", "auth_failure"),
        ("auth_failure", "dependency_failure"),
        ("dependency_failure", "auth_failure"),
        ("auth_failure", "unknown"),
        ("unknown", "auth_failure"),
        ("config_error", "unknown"),
        ("unknown", "config_error"),
        ("latency_degradation", "unknown"),
        ("unknown", "latency_degradation"),
    )

    def __init__(
        self,
        max_root_causes: int = 1,
        start_index: int = 0,
        stride: int = 1,
        include_dependency: bool | None = None,
        pair_pool_per_type: int = 20,
        sliding_pair_pool: int = 40,
        diagnostic_pair_pool_per_type: int = 32,
        diagnostic_sliding_pair_pool: int = 80,
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
        self.diagnostic_pair_pool_per_type = max(1, int(diagnostic_pair_pool_per_type))
        self.diagnostic_sliding_pair_pool = max(2, int(diagnostic_sliding_pair_pool))
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
            "solver": "candidate_sweep_v4",
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

        base_rows_by_type = self._rows_by_type(filtered)

        # Stage 1: exact high-precision v2 ordering over the original candidate
        # set only. This is what produced 28/37 on multifaults, so keep it first.
        self._add_typed_pairs(
            add=add,
            rows_by_type=base_rows_by_type,
            templates=self.BASE_PAIR_TEMPLATES,
            pool_per_type=self.pair_pool_per_type,
        )
        self._add_same_fault_pairs(
            add=add,
            rows_by_type=base_rows_by_type,
            pool_per_type=self.pair_pool_per_type,
            fault_types=("auth_failure", "config_error", "infra_failure", "network_failure", "latency_degradation", "dependency_failure"),
        )
        self._add_rank_window_pairs(add=add, rows=filtered[: self.sliding_pair_pool])

        # Stage 2: diagnostic fallback variants. These are useful for the v11
        # miss families, but they are deliberately after the original high-
        # precision plan so they cannot crowd out exact config+auth samples.
        diagnostic_rows = self._diagnostic_variant_rows(filtered)
        if diagnostic_rows:
            diagnostic_rows_by_type = self._rows_by_type(diagnostic_rows)
            merged_rows_by_type = self._rows_by_type(filtered + diagnostic_rows)
            self._add_typed_pairs(
                add=add,
                rows_by_type=merged_rows_by_type,
                templates=self.DIAGNOSTIC_PAIR_TEMPLATES,
                pool_per_type=self.diagnostic_pair_pool_per_type,
            )
            self._add_same_fault_pairs(
                add=add,
                rows_by_type=diagnostic_rows_by_type,
                pool_per_type=self.diagnostic_pair_pool_per_type,
                fault_types=("network_failure", "dependency_failure", "unknown", "latency_degradation"),
            )
            self._add_rank_window_pairs(add=add, rows=(filtered + diagnostic_rows)[: self.diagnostic_sliding_pair_pool])

        if not plan:
            return [[s] for s in singles]
        return plan

    def _filter_candidates(self, candidates: list[dict[str, Any]], valid_services: set[str]) -> list[dict[str, Any]]:
        rows_by_key: dict[str, dict[str, Any]] = {}

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
            old = rows_by_key.get(key)
            if old is None or self._score(row) > self._score(old):
                rows_by_key[key] = {**row, "service": svc, "fault_type": ft}

        rows = list(rows_by_key.values())
        rows.sort(key=self._score, reverse=True)
        return rows

    def _diagnostic_variant_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        variants_by_key: dict[str, dict[str, Any]] = {}

        def add_variant(row: dict[str, Any], fault_type: str, score_delta: float, reason: str) -> None:
            svc = str(row.get("service") or "").strip()
            ft = normalize_fault_type(fault_type)
            if not svc or _is_non_root_service(svc):
                return
            if ft == "dependency_failure" and not self.include_dependency:
                return
            key = f"{svc}::{ft}"
            reasons = list(row.get("reasons", []) or [])
            if reason not in reasons:
                reasons.append(reason)
            new_row = {**row, "service": svc, "fault_type": ft, "score": self._score(row) + score_delta, "reasons": reasons}
            old = variants_by_key.get(key)
            if old is None or self._score(new_row) > self._score(old):
                variants_by_key[key] = new_row

        for row in rows:
            svc = str(row.get("service") or "").strip()
            ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
            if not svc or _is_non_root_service(svc):
                continue

            # network_loss may appear latency/config/infra-like in redacted
            # telemetry. Add same-service network variants only as fallback.
            if not _is_db_or_cache_service(svc) and ft in {"latency_degradation", "config_error", "infra_failure"}:
                add_variant(row, "network_failure", -0.35, "sweep_v4_late_network_loss_variant")
            if not _is_db_or_cache_service(svc) and ft == "network_failure":
                add_variant(row, "latency_degradation", -0.35, "sweep_v4_late_network_delay_variant")

            # wrong_bin_usage is normalized as unknown, while symptoms can look
            # config-like. Add unknown variants late.
            if not _is_db_or_cache_service(svc) and ft in {"config_error", "infra_failure", "latency_degradation", "network_failure"}:
                add_variant(row, "unknown", -0.50, "sweep_v4_late_wrong_bin_unknown_variant")

            # user_unregistered_mongodb is dependency_failure; revoke_auth is
            # auth_failure. Add both DB interpretations late.
            if _is_db_or_cache_service(svc) and ft == "auth_failure":
                add_variant(row, "dependency_failure", -0.25, "sweep_v4_late_db_dependency_variant")
            if _is_db_or_cache_service(svc) and ft == "dependency_failure":
                add_variant(row, "auth_failure", -0.25, "sweep_v4_late_db_auth_variant")

        variants = list(variants_by_key.values())
        variants.sort(key=self._score, reverse=True)
        return variants

    def _add_typed_pairs(
        self,
        *,
        add,
        rows_by_type: dict[str, list[dict[str, Any]]],
        templates: tuple[tuple[str, str], ...],
        pool_per_type: int,
    ) -> None:
        for ft_a, ft_b in templates:
            rows_a = rows_by_type.get(ft_a, [])[:pool_per_type]
            rows_b = rows_by_type.get(ft_b, [])[:pool_per_type]
            for a in rows_a:
                for b in rows_b:
                    ka = self._candidate_key(a)
                    kb = self._candidate_key(b)
                    if ka == kb:
                        continue
                    add([ka, kb])

    def _add_same_fault_pairs(
        self,
        *,
        add,
        rows_by_type: dict[str, list[dict[str, Any]]],
        pool_per_type: int,
        fault_types: tuple[str, ...],
    ) -> None:
        for ft in fault_types:
            rows = rows_by_type.get(ft, [])[:pool_per_type]
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    add([self._candidate_key(a), self._candidate_key(b)])

    def _add_rank_window_pairs(self, *, add, rows: list[dict[str, Any]]) -> None:
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                add([self._candidate_key(a), self._candidate_key(b)])

    @staticmethod
    def _rows_by_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            ft = normalize_fault_type(str(row.get("fault_type") or "unknown"))
            out.setdefault(ft, []).append(row)
        return out

    @staticmethod
    def _candidate_key(row: dict[str, Any]) -> str:
        return f"{row.get('service')}::{normalize_fault_type(str(row.get('fault_type') or 'unknown'))}"

    @staticmethod
    def _score(row: dict[str, Any]) -> float:
        try:
            return float(row.get("score", 0.0) or 0.0)
        except Exception:
            return 0.0

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
