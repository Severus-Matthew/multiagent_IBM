from __future__ import annotations

"""Deterministic semantic projection for trainable RCA/Action policy inputs.

The processed AIOpsLab state is already redacted, but several fields are still far
larger than is useful for policy learning.  In particular, per-service metric
``groups`` duplicate compact ``flat_summary`` values, system objects carry verbose
Kubernetes deployment/event payloads, log objects repeat long evidence strings, and
observability metadata contains file inventories/provenance.

This module builds a new JSON object from the already-sanitized state.  It never
truncates rendered prompt token IDs.  Every reduction is schema-aware and uses only
observable agent-facing information.  All services remain represented at a summary
tier; richer detail is reserved for services ranked by observable anomaly signals.
No oracle/private fields are consulted.
"""

from dataclasses import dataclass
import json
import math
from typing import Any

from .agent_input_safety import agent_input_safety_report


@dataclass(frozen=True)
class BoundedAgentStateConfig:
    # Character limit is a tokenizer-independent structural guard.  The companion
    # audit checks the exact Qwen chat-template token count.
    max_serialized_chars: int = 42_000
    # These are rich-detail limits, not service-visibility limits.  Every system
    # service keeps a compact health summary; every selected metric/log service
    # keeps its compact aggregate signal.  Rich deployment/event or text evidence
    # is restricted to the most anomalous services using observable signals only.
    max_system_services: int = 12
    max_metric_services: int = 64
    max_log_services: int = 8
    max_list_examples: int = 2
    max_string_chars: int = 220
    max_dependency_edges: int = 8

    def validate(self) -> None:
        if self.max_serialized_chars < 8_000:
            raise ValueError("max_serialized_chars must be >= 8000")
        for name in (
            "max_system_services",
            "max_metric_services",
            "max_log_services",
            "max_list_examples",
            "max_dependency_edges",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_string_chars < 64:
            raise ValueError("max_string_chars must be >= 64")


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _compact_string(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    suffix = "...[truncated]"
    return text[: max(0, limit - len(suffix))] + suffix


def _compact_value(
    value: Any,
    *,
    list_examples: int,
    string_chars: int,
    depth: int = 0,
    max_depth: int = 3,
) -> Any:
    """Small generic compactor used only for already-small structural sections."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _compact_string(value, string_chars)
    if depth >= max_depth:
        if isinstance(value, dict):
            return {
                "summary": "nested_dict",
                "num_keys": len(value),
                "keys": sorted(str(k) for k in value.keys())[:8],
            }
        if isinstance(value, (list, tuple)):
            return {"summary": "nested_list", "count": len(value)}
        return _compact_string(value, string_chars)
    if isinstance(value, dict):
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
        if not values:
            return []
        examples = [
            _compact_value(
                v,
                list_examples=list_examples,
                string_chars=string_chars,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for v in values[:list_examples]
        ]
        if len(values) <= len(examples):
            return examples
        return {
            "count": len(values),
            "first_examples": examples,
            "omitted": len(values) - len(examples),
        }
    return _compact_string(value, string_chars)


_DIAGNOSTIC_PATH_MARKERS = (
    "health",
    "status",
    "ready",
    "replica",
    "available",
    "unavailable",
    "restart",
    "pending",
    "crash",
    "error",
    "fail",
    "warning",
    "condition",
    "reason",
    "endpoint",
    "address",
    "latency",
    "timeout",
    "cpu",
    "memory",
    "request",
    "node",
    "pod",
    "container",
)


def _diagnostic_scalars(value: Any, *, limit: int, string_chars: int) -> dict[str, Any]:
    """Flatten a verbose object to a small deterministic set of diagnostic leaves."""
    rows: list[tuple[int, str, Any]] = []

    def walk(obj: Any, path: str, depth: int) -> None:
        if len(rows) >= 256 or depth > 5:
            return
        if obj is None or isinstance(obj, (bool, int, float, str)):
            path_l = path.lower()
            priority = sum(1 for marker in _DIAGNOSTIC_PATH_MARKERS if marker in path_l)
            if isinstance(obj, str):
                obj = _compact_string(obj, string_chars)
            rows.append((priority, path or "value", obj))
            return
        if isinstance(obj, dict):
            for key, child in sorted(obj.items(), key=lambda kv: str(kv[0])):
                child_path = f"{path}.{key}" if path else str(key)
                walk(child, child_path, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            # Lists are inventories/events in these schemas.  Record their size;
            # detailed examples are handled explicitly by the caller when useful.
            rows.append((1, f"{path}.count" if path else "count", len(obj)))
            return
        rows.append((0, path or "value", _compact_string(obj, string_chars)))

    walk(value, "", 0)
    rows.sort(key=lambda row: (-row[0], row[1]))
    out: dict[str, Any] = {}
    for _, path, scalar in rows:
        if path not in out:
            out[path] = scalar
        if len(out) >= limit:
            break
    return out


def _observable_service_priority(state: dict[str, Any]) -> list[str]:
    """Rank services using only agent-visible health/log anomaly evidence."""
    score: dict[str, float] = {}
    log_rank: dict[str, int] = {}

    def bump(service: Any, amount: float) -> None:
        name = str(service or "").strip()
        if name:
            score[name] = score.get(name, 0.0) + float(amount)

    service_health = state.get("service_health", {}) or {}
    if isinstance(service_health, dict):
        for service, info in service_health.items():
            text = _json(info).lower()
            amount = 0.5
            for marker, weight in (
                ("unready", 4.0),
                ("pending", 4.0),
                ("crash", 4.0),
                ("error", 3.0),
                ("fail", 3.0),
                ("degraded", 3.0),
                ("unavailable", 3.0),
                ("timeout", 2.0),
                ("restart", 2.0),
            ):
                if marker in text:
                    amount += weight
            bump(service, amount)

    llm_view = state.get("llm_view", {}) or {}
    ranked_logs = llm_view.get("top_log_error_services", []) if isinstance(llm_view, dict) else []
    if isinstance(ranked_logs, list):
        for rank, item in enumerate(ranked_logs):
            service = item.get("service") if isinstance(item, dict) else None
            if service:
                log_rank[str(service)] = rank
                bump(service, max(1.0, 8.0 - 0.35 * rank))

    system = state.get("system", {}) or {}
    if isinstance(system, dict):
        for service, info in system.items():
            text = _json((info or {}).get("health", info) if isinstance(info, dict) else info).lower()
            amount = sum(
                1.0
                for marker in ("unready", "pending", "crash", "error", "failed", "degraded", "unavailable")
                if marker in text
            )
            bump(service, amount)

    for service in state.get("services", []) or []:
        bump(service, 0.0)
    for mapping_key in ("system", "metrics", "logs"):
        mapping = state.get(mapping_key, {}) or {}
        if isinstance(mapping, dict):
            for service in mapping:
                bump(service, 0.0)

    return sorted(score, key=lambda s: (-score[s], log_rank.get(s, 10**9), s))


def _priority_subset(mapping: Any, priority: list[str], limit: int) -> list[str]:
    if not isinstance(mapping, dict):
        return []
    chosen: list[str] = []
    seen: set[str] = set()
    for service in priority:
        if service in mapping and service not in seen:
            chosen.append(service)
            seen.add(service)
            if len(chosen) >= limit:
                return chosen
    for service in sorted(str(k) for k in mapping):
        if service not in seen:
            chosen.append(service)
            seen.add(service)
            if len(chosen) >= limit:
                break
    return chosen


def _compact_metric_map(mapping: Any, priority: list[str], cfg: BoundedAgentStateConfig) -> dict[str, Any]:
    """Keep metric aggregates; drop the ~7.5k-char duplicated groups payload."""
    if not isinstance(mapping, dict):
        return {}
    chosen = _priority_subset(mapping, priority, min(cfg.max_metric_services, len(mapping)))
    out: dict[str, Any] = {}
    for service in chosen:
        raw = mapping.get(service)
        if not isinstance(raw, dict):
            out[service] = _compact_value(
                raw, list_examples=1, string_chars=cfg.max_string_chars, max_depth=2
            )
            continue
        entry: dict[str, Any] = {}
        if "metric_signal_present" in raw:
            entry["metric_signal_present"] = raw.get("metric_signal_present")
        if "flat_summary" in raw:
            entry["flat_summary"] = _compact_value(
                raw.get("flat_summary"),
                list_examples=1,
                string_chars=160,
                max_depth=2,
            )
        groups = raw.get("groups")
        if isinstance(groups, dict):
            entry["group_names"] = sorted(str(k) for k in groups.keys())
        out[service] = entry
    if len(chosen) < len(mapping):
        out["__projection_summary__"] = {
            "total_services": len(mapping),
            "services_with_metric_summary": len(chosen),
            "omitted_metric_service_count": len(mapping) - len(chosen),
        }
    return out


def _compact_system_map(mapping: Any, priority: list[str], cfg: BoundedAgentStateConfig) -> dict[str, Any]:
    """Represent every service by health; add rich K8s detail only to top anomalies."""
    if not isinstance(mapping, dict):
        return {}
    detail_services = set(_priority_subset(mapping, priority, min(cfg.max_system_services, len(mapping))))
    out: dict[str, Any] = {}
    for service in sorted(str(k) for k in mapping):
        raw = mapping.get(service)
        if not isinstance(raw, dict):
            out[service] = {"summary": _compact_string(raw, 120)}
            continue
        entry: dict[str, Any] = {
            "health": _diagnostic_scalars(raw.get("health", {}), limit=8, string_chars=120)
        }
        if service in detail_services:
            if "deployment" in raw:
                entry["deployment_signals"] = _diagnostic_scalars(
                    raw.get("deployment"), limit=10, string_chars=140
                )
            if "endpoints" in raw:
                entry["endpoint_signals"] = _diagnostic_scalars(
                    raw.get("endpoints"), limit=8, string_chars=140
                )
            events = raw.get("events_top")
            if events:
                entry["event_signals"] = _diagnostic_scalars(
                    events, limit=8, string_chars=160
                )
            inventory: dict[str, Any] = {}
            for key in ("pods", "nodes", "containers", "images"):
                value = raw.get(key)
                if isinstance(value, (list, tuple)):
                    inventory[f"{key}_count"] = len(value)
                    if value:
                        inventory[f"{key}_example"] = _compact_string(value[0], 120)
            if inventory:
                entry["inventory"] = inventory
        out[service] = entry
    out["__projection_summary__"] = {
        "total_services": len(mapping),
        "health_summary_services": len(mapping),
        "rich_detail_services": len(detail_services),
        "rich_detail_service_names": sorted(detail_services),
    }
    return out


def _first_text_examples(value: Any, *, count: int, string_chars: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value[:count]:
        if isinstance(item, str):
            out.append(_compact_string(item, string_chars))
        elif isinstance(item, dict):
            out.append(_diagnostic_scalars(item, limit=6, string_chars=string_chars))
        else:
            out.append(_compact_string(item, string_chars))
    return out


def _compact_logs_map(mapping: Any, priority: list[str], cfg: BoundedAgentStateConfig) -> dict[str, Any]:
    """Keep counts/signals for every log service; bound raw text to top anomalies."""
    if not isinstance(mapping, dict):
        return {}
    detail_services = set(_priority_subset(mapping, priority, min(cfg.max_log_services, len(mapping))))
    out: dict[str, Any] = {}
    for service in sorted(str(k) for k in mapping):
        raw = mapping.get(service)
        if not isinstance(raw, dict):
            out[service] = {"summary": _compact_string(raw, 120)}
            continue
        entry: dict[str, Any] = {}
        for key in ("signal", "dependency_error_counts", "error_families", "severity_counts"):
            if key in raw:
                entry[key] = _compact_value(
                    raw.get(key), list_examples=1, string_chars=140, max_depth=2
                )
        if service in detail_services:
            templates = _first_text_examples(
                raw.get("error_templates_top"), count=1, string_chars=180
            )
            evidence = _first_text_examples(
                raw.get("evidence_lines_top"), count=1, string_chars=180
            )
            if templates:
                entry["error_template_example"] = templates[0]
            if evidence:
                entry["evidence_line_example"] = evidence[0]
        out[service] = entry
    out["__projection_summary__"] = {
        "total_services": len(mapping),
        "signal_summary_services": len(mapping),
        "text_evidence_services": len(detail_services),
        "text_evidence_service_names": sorted(detail_services),
    }
    return out


def _compact_observability_metadata(value: Any, cfg: BoundedAgentStateConfig) -> dict[str, Any]:
    """Convert file/provenance inventories to counts and preserve compact signals."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for section, data in sorted(value.items(), key=lambda kv: str(kv[0])):
        section_s = str(section)
        if not isinstance(data, dict):
            out[section_s] = _compact_value(
                data, list_examples=1, string_chars=cfg.max_string_chars, max_depth=2
            )
            continue
        section_out: dict[str, Any] = {}
        for key, child in sorted(data.items(), key=lambda kv: str(kv[0])):
            key_s = str(key)
            if key_s in {"files_seen", "empty_files", "files_used", "pods_wide", "pod_to_service"}:
                if isinstance(child, (list, dict)):
                    section_out[f"{key_s}_count"] = len(child)
                continue
            if key_s == "dependency_error_edges_from_logs" and isinstance(child, list):
                examples = [
                    _diagnostic_scalars(edge, limit=6, string_chars=140)
                    if isinstance(edge, dict)
                    else _compact_string(edge, 140)
                    for edge in child[: cfg.max_dependency_edges]
                ]
                section_out[key_s] = {
                    "count": len(child),
                    "first_edges": examples,
                    "omitted": max(0, len(child) - len(examples)),
                }
                continue
            if key_s == "unhealthy_services" and isinstance(child, list):
                # This is observable system state, not an oracle label.  Preserve
                # names because it is compact and useful for RCA.
                section_out[key_s] = [str(x) for x in child[:64]]
                continue
            if key_s in {"file_metas"} and isinstance(child, list):
                section_out[f"{key_s}_count"] = len(child)
                continue
            if key_s == "parse_errors" and isinstance(child, list):
                section_out["parse_error_count"] = len(child)
                if child:
                    section_out["parse_error_example"] = _compact_string(child[0], 160)
                continue
            section_out[key_s] = _compact_value(
                child,
                list_examples=1,
                string_chars=160,
                max_depth=2,
            )
        out[section_s] = section_out
    return out


def _compact_llm_view(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, child in sorted(value.items(), key=lambda kv: str(kv[0])):
        key_s = str(key)
        if key_s == "top_log_error_services" and isinstance(child, list):
            rows: list[Any] = []
            for item in child[:8]:
                if isinstance(item, dict):
                    compact = _diagnostic_scalars(item, limit=8, string_chars=140)
                    if "service" in item:
                        compact = {"service": str(item.get("service")), **compact}
                    rows.append(compact)
                else:
                    rows.append(_compact_string(item, 140))
            out[key_s] = {"count": len(child), "top_entries": rows}
        else:
            out[key_s] = _compact_value(
                child, list_examples=2, string_chars=160, max_depth=2
            )
    return out


def _compact_model_table(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _diagnostic_scalars(v, limit=5, string_chars=120)
            if isinstance(v, (dict, list, tuple))
            else _compact_string(v, 120)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, list):
        rows = []
        for item in value[:64]:
            rows.append(
                _diagnostic_scalars(item, limit=5, string_chars=120)
                if isinstance(item, (dict, list, tuple))
                else _compact_string(item, 120)
            )
        return {"count": len(value), "rows": rows}
    return _compact_value(value, list_examples=1, string_chars=120, max_depth=2)


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
            "version": "bounded_agent_state_v2_schema_aware_observable_projection",
            "source_safe_for_training_agent": True,
            "service_detail_ranking": "observable_health_and_log_error_signals_only",
            "raw_prompt_token_truncation_used": False,
            "metric_policy": "flat_summary_plus_signal_drop_verbose_groups",
            "system_policy": "all_service_health_plus_observable_top_service_detail",
            "log_policy": "all_service_counts_plus_observable_top_service_text_evidence",
            "observability_policy": "provenance_inventories_to_counts",
        }
    }

    # Small structural/global context.
    for key in ("timestamp", "workload", "services", "clusters", "graph", "traces", "sla", "redaction"):
        if key in sanitized_state:
            out[key] = _compact_value(
                sanitized_state[key],
                list_examples=4,
                string_chars=cfg.max_string_chars,
                max_depth=3,
            )

    if "service_health" in sanitized_state:
        out["service_health"] = _compact_value(
            sanitized_state["service_health"],
            list_examples=2,
            string_chars=160,
            max_depth=2,
        )
    if "model_table" in sanitized_state:
        out["model_table"] = _compact_model_table(sanitized_state["model_table"])
    if "llm_view" in sanitized_state:
        out["llm_view"] = _compact_llm_view(sanitized_state["llm_view"])

    out["system"] = _compact_system_map(sanitized_state.get("system"), priority, cfg)
    out["metrics"] = _compact_metric_map(sanitized_state.get("metrics"), priority, cfg)
    out["logs"] = _compact_logs_map(sanitized_state.get("logs"), priority, cfg)
    out["observability_metadata"] = _compact_observability_metadata(
        sanitized_state.get("observability_metadata"), cfg
    )

    source_chars = len(_json(sanitized_state))
    projected_chars_before_metadata = len(_json(out))
    out["projection"].update(
        {
            "source_serialized_chars": source_chars,
            "projected_serialized_chars_before_projection_metadata": projected_chars_before_metadata,
            "system_total_services": len(sanitized_state.get("system", {}) or {}),
            "system_rich_detail_services": min(
                cfg.max_system_services, len(sanitized_state.get("system", {}) or {})
            ),
            "metric_total_services": len(sanitized_state.get("metrics", {}) or {}),
            "metric_summary_services": min(
                cfg.max_metric_services, len(sanitized_state.get("metrics", {}) or {})
            ),
            "log_total_services": len(sanitized_state.get("logs", {}) or {}),
            "log_text_evidence_services": min(
                cfg.max_log_services, len(sanitized_state.get("logs", {}) or {})
            ),
            "priority_service_preview": priority[:12],
        }
    )

    chars = len(_json(out))
    out["projection"]["projected_serialized_chars"] = chars
    chars = len(_json(out))
    if chars > cfg.max_serialized_chars:
        section_sizes = {
            key: len(_json(value))
            for key, value in out.items()
            if key != "projection"
        }
        section_sizes = dict(sorted(section_sizes.items(), key=lambda kv: kv[1], reverse=True))
        raise ValueError(
            f"bounded semantic projection still exceeds character budget: {chars} > {cfg.max_serialized_chars}; "
            f"section_chars={section_sizes}. Tighten semantic section/detail limits explicitly rather than "
            "truncating prompt tokens"
        )

    safety_after = agent_input_safety_report(out)
    if not safety_after.get("safe_for_training_agent"):
        raise AssertionError(f"bounded projection introduced an unsafe agent field: {safety_after}")
    return out
