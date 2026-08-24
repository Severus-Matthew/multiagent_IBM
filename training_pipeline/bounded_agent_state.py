from __future__ import annotations

"""Deterministic semantic projection for trainable RCA/Action policy inputs.

The processed AIOpsLab state is already redacted, but some scenarios still contain
>100k tokenizer tokens because it carries dense per-service metric series, repeated
system detail, log inventories, and observability provenance.  Passing that whole
object to a LoRA policy is unnecessary and makes single-GPU backward impractical.

This module builds a *new JSON object* from the redacted state.  It never slices a
rendered prompt/token sequence.  Instead it preserves compact health/dependency/SLA
context, summarizes time-series-like lists, ranks service detail only by observable
signals, and converts file inventories/provenance to counts.  No oracle/private
fields are consulted.
"""

from dataclasses import dataclass
import json
import math
from statistics import fmean
from typing import Any, Iterable

from .agent_input_safety import agent_input_safety_report


@dataclass(frozen=True)
class BoundedAgentStateConfig:
    # Character limits are tokenizer-agnostic guards.  The Qwen audit separately
    # measures exact chat-template token counts.
    max_serialized_chars: int = 42_000
    max_system_services: int = 24
    max_metric_services: int = 24
    max_log_services: int = 20
    max_list_examples: int = 4
    max_string_chars: int = 320
    max_dependency_edges: int = 16

    def validate(self) -> None:
        if self.max_serialized_chars < 8_000:
            raise ValueError("max_serialized_chars must be >= 8000")
        for name in ("max_system_services", "max_metric_services", "max_log_services", "max_list_examples"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_string_chars < 64:
            raise ValueError("max_string_chars must be >= 64")


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _finite_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        value = float(x)
        return value if math.isfinite(value) else None
    return None


def _round_number(x: float) -> int | float:
    if float(x).is_integer():
        return int(x)
    return round(float(x), 6)


def _numeric_summary(values: Iterable[Any]) -> dict[str, Any] | None:
    nums = [_finite_number(x) for x in values]
    nums = [x for x in nums if x is not None]
    if not nums:
        return None
    return {
        "count": len(nums),
        "min": _round_number(min(nums)),
        "max": _round_number(max(nums)),
        "mean": _round_number(fmean(nums)),
        "first": _round_number(nums[0]),
        "last": _round_number(nums[-1]),
    }


def _compact_string(value: str, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)] + "...[truncated]"


def _representative_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [n - 1]
    if k == 2:
        return [0, n - 1]
    # Preserve beginning/end plus approximately uniform interior observations.
    out = {0, n - 1}
    for i in range(1, k - 1):
        out.add(round(i * (n - 1) / (k - 1)))
    return sorted(out)[:k]


def _compact_value(
    value: Any,
    *,
    list_examples: int,
    string_chars: int,
    depth: int = 0,
    max_depth: int = 5,
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _compact_string(value, string_chars)
    if depth >= max_depth:
        if isinstance(value, dict):
            return {"summary": "nested_dict", "keys": sorted(str(k) for k in value.keys())[:12], "num_keys": len(value)}
        if isinstance(value, (list, tuple)):
            return {"summary": "nested_list", "count": len(value)}
        return _compact_string(str(value), string_chars)

    if isinstance(value, dict):
        # Key order is deterministic; keep all keys at shallow depths because
        # service/metric dictionaries often encode distinct diagnostic features.
        return {
            str(k): _compact_value(
                v,
                list_examples=list_examples,
                string_chars=string_chars,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(value, (list, tuple)):
        values = list(value)
        numeric = _numeric_summary(values)
        if numeric is not None and len(numeric) == 6 and len(values) == numeric["count"]:
            return {"numeric_series": numeric}
        if not values:
            return []
        idx = _representative_indices(len(values), list_examples)
        examples = [
            _compact_value(
                values[i],
                list_examples=list_examples,
                string_chars=string_chars,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for i in idx
        ]
        if len(values) <= len(examples):
            return examples
        return {
            "count": len(values),
            "representative_examples": examples,
            "omitted": len(values) - len(examples),
        }

    return _compact_string(str(value), string_chars)


def _observable_service_priority(state: dict[str, Any]) -> list[str]:
    """Rank services using only agent-visible anomaly evidence."""
    score: dict[str, float] = {}
    order_bonus: dict[str, float] = {}

    def bump(service: Any, amount: float) -> None:
        name = str(service or "").strip()
        if name:
            score[name] = score.get(name, 0.0) + float(amount)

    # Explicit compact health summaries are the strongest non-oracle signal.
    service_health = state.get("service_health", {}) or {}
    if isinstance(service_health, dict):
        for service, info in service_health.items():
            text = _json(info).lower()
            s = 0.0
            for marker, weight in (
                ("unready", 4.0), ("pending", 4.0), ("crash", 4.0),
                ("error", 3.0), ("fail", 3.0), ("degraded", 3.0),
                ("unavailable", 3.0), ("timeout", 2.0), ("restart", 2.0),
            ):
                if marker in text:
                    s += weight
            # Positive numeric health flags/counters add weak evidence.
            if isinstance(info, dict):
                for v in info.values():
                    n = _finite_number(v)
                    if n is not None and n > 0:
                        s += min(2.0, abs(n)) * 0.1
            bump(service, s + 0.5)

    llm_view = state.get("llm_view", {}) or {}
    top_logs = llm_view.get("top_log_error_services", []) if isinstance(llm_view, dict) else []
    if isinstance(top_logs, list):
        for rank, item in enumerate(top_logs):
            service = item.get("service") if isinstance(item, dict) else None
            if service:
                bump(service, max(1.0, 8.0 - 0.35 * rank))
                order_bonus[str(service)] = -float(rank)

    # Scan per-service system health for observable failure language.
    system = state.get("system", {}) or {}
    if isinstance(system, dict):
        for service, info in system.items():
            text = _json(info).lower()
            amount = 0.0
            for marker in ("unready", "pending", "crash", "error", "failed", "degraded", "unavailable"):
                if marker in text:
                    amount += 1.0
            bump(service, amount)

    # Ensure every known service remains rankable; alphabetic tie-break makes the
    # projection deterministic and does not encode hidden labels.
    services = state.get("services", []) or []
    for service in services:
        bump(service, 0.0)
    for mapping_key in ("system", "metrics", "logs"):
        mapping = state.get(mapping_key, {}) or {}
        if isinstance(mapping, dict):
            for service in mapping.keys():
                bump(service, 0.0)

    return sorted(score, key=lambda s: (-score[s], order_bonus.get(s, 0.0), s))


def _select_service_map(
    mapping: Any,
    priority: list[str],
    *,
    limit: int,
    list_examples: int,
    string_chars: int,
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    chosen: list[str] = []
    seen: set[str] = set()
    for service in priority:
        if service in mapping and service not in seen:
            chosen.append(service)
            seen.add(service)
            if len(chosen) >= limit:
                break
    if len(chosen) < limit:
        for service in sorted(str(k) for k in mapping.keys()):
            if service not in seen:
                chosen.append(service)
                seen.add(service)
                if len(chosen) >= limit:
                    break
    out = {
        service: _compact_value(
            mapping[service],
            list_examples=list_examples,
            string_chars=string_chars,
        )
        for service in chosen
    }
    omitted = sorted(str(k) for k in mapping.keys() if str(k) not in seen)
    if omitted:
        out["__projection_summary__"] = {
            "total_services": len(mapping),
            "detailed_services": len(chosen),
            "omitted_service_count": len(omitted),
            # Names remain available globally in `services`; only a small preview
            # is included here to keep section metadata bounded.
            "omitted_service_preview": omitted[:12],
        }
    return out


def _compact_observability_metadata(value: Any, cfg: BoundedAgentStateConfig) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for section, data in sorted(value.items(), key=lambda kv: str(kv[0])):
        if not isinstance(data, dict):
            out[str(section)] = _compact_value(
                data,
                list_examples=cfg.max_list_examples,
                string_chars=cfg.max_string_chars,
            )
            continue
        section_out: dict[str, Any] = {}
        for key, child in sorted(data.items(), key=lambda kv: str(kv[0])):
            key_s = str(key)
            if key_s in {"files_seen", "empty_files"} and isinstance(child, list):
                section_out[f"{key_s}_count"] = len(child)
                continue
            if key_s == "dependency_error_edges_from_logs" and isinstance(child, list):
                idx = _representative_indices(len(child), min(cfg.max_dependency_edges, len(child)))
                section_out[key_s] = {
                    "count": len(child),
                    "representative_edges": [
                        _compact_value(
                            child[i],
                            list_examples=cfg.max_list_examples,
                            string_chars=cfg.max_string_chars,
                        )
                        for i in idx
                    ],
                }
                continue
            section_out[key_s] = _compact_value(
                child,
                list_examples=cfg.max_list_examples,
                string_chars=cfg.max_string_chars,
            )
        out[str(section)] = section_out
    return out


def build_bounded_agent_state(
    sanitized_state: dict[str, Any],
    *,
    config: BoundedAgentStateConfig | None = None,
) -> dict[str, Any]:
    """Return a bounded semantic projection of an already-redacted agent state."""
    cfg = config or BoundedAgentStateConfig()
    cfg.validate()
    if not isinstance(sanitized_state, dict):
        raise TypeError("sanitized_state must be a dict")
    safety_before = agent_input_safety_report(sanitized_state)
    if not safety_before.get("safe_for_training_agent"):
        raise ValueError(f"input to bounded projection is not training-safe: {safety_before}")

    priority = _observable_service_priority(sanitized_state)
    out: dict[str, Any] = {
        "projection": {
            "version": "bounded_agent_state_v1_observable_semantic_projection",
            "source_safe_for_training_agent": True,
            "service_detail_ranking": "observable_health_and_log_error_signals_only",
            "raw_prompt_token_truncation_used": False,
        }
    }

    # Preserve small structural/global context directly (with recursive list/string
    # compaction for robustness across future dataset revisions).
    for key in (
        "timestamp", "workload", "services", "clusters", "graph", "traces",
        "sla", "service_health", "model_table", "redaction",
    ):
        if key in sanitized_state:
            out[key] = _compact_value(
                sanitized_state[key],
                list_examples=max(cfg.max_list_examples, 8),
                string_chars=cfg.max_string_chars,
            )

    if "llm_view" in sanitized_state:
        out["llm_view"] = _compact_value(
            sanitized_state["llm_view"],
            list_examples=max(cfg.max_list_examples, 8),
            string_chars=cfg.max_string_chars,
        )

    out["system"] = _select_service_map(
        sanitized_state.get("system"), priority,
        limit=cfg.max_system_services,
        list_examples=cfg.max_list_examples,
        string_chars=cfg.max_string_chars,
    )
    out["metrics"] = _select_service_map(
        sanitized_state.get("metrics"), priority,
        limit=cfg.max_metric_services,
        list_examples=cfg.max_list_examples,
        string_chars=cfg.max_string_chars,
    )
    out["logs"] = _select_service_map(
        sanitized_state.get("logs"), priority,
        limit=cfg.max_log_services,
        list_examples=cfg.max_list_examples,
        string_chars=cfg.max_string_chars,
    )
    out["observability_metadata"] = _compact_observability_metadata(
        sanitized_state.get("observability_metadata"), cfg
    )

    out["projection"]["source_serialized_chars"] = len(_json(sanitized_state))
    out["projection"]["projected_serialized_chars"] = len(_json(out))
    out["projection"]["priority_service_preview"] = priority[:12]

    # A hard structural guard catches schema drift.  We deliberately fail rather
    # than silently slice the rendered JSON/prompt, because arbitrary token
    # truncation can destroy diagnostic semantics and exact replay provenance.
    chars = len(_json(out))
    if chars > cfg.max_serialized_chars:
        raise ValueError(
            f"bounded semantic projection still exceeds character budget: {chars} > {cfg.max_serialized_chars}; "
            "tighten semantic section limits explicitly rather than truncating prompt tokens"
        )

    safety_after = agent_input_safety_report(out)
    if not safety_after.get("safe_for_training_agent"):
        raise AssertionError(f"bounded projection introduced an unsafe agent field: {safety_after}")
    return out
