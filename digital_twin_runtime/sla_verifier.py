from __future__ import annotations

from typing import Any

from .telemetry_comparator import symptom_signature


SLA_FIELDS = [
    "degraded_services",
    "failed_edges",
    "top_error_services",
    "trace_sources",
    "trace_targets",
    "metric_anomaly_services",
]


def sla_verdict_from_signature(signature: dict[str, Any]) -> dict[str, Any]:
    """Compute a simple SLA-style verdict from an observable symptom signature.

    This is intentionally based on redacted telemetry symptoms, not oracle labels.
    It gives the action reward an explicit before/after SLA target while the live
    Kubernetes SLA verifier is not yet wired into the offline pipeline.
    """
    counts = {field: _count(signature.get(field, [])) for field in SLA_FIELDS}
    hard_violations = (
        counts["degraded_services"]
        + counts["failed_edges"]
        + counts["top_error_services"]
    )
    soft_violations = counts["trace_sources"] + counts["trace_targets"] + counts["metric_anomaly_services"]
    total_violations = hard_violations + 0.25 * soft_violations
    return {
        "sla_restored": hard_violations == 0,
        "hard_violations": int(hard_violations),
        "soft_violations": int(soft_violations),
        "weighted_violations": round(float(total_violations), 4),
        "counts": counts,
    }


def sla_verdict_from_state(state: dict[str, Any]) -> dict[str, Any]:
    sig = symptom_signature(state)
    return {
        "signature_summary": signature_summary(sig),
        **sla_verdict_from_signature(sig),
    }


def symptom_reduction(before: dict[str, Any], after: dict[str, Any]) -> float:
    b = float(before.get("weighted_violations", 0.0) or 0.0)
    a = float(after.get("weighted_violations", 0.0) or 0.0)
    if b <= 0.0:
        return 1.0 if a <= 0.0 else 0.0
    return round(max(0.0, min(1.0, (b - a) / b)), 6)


def signature_summary(signature: dict[str, Any], max_items: int = 10) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in SLA_FIELDS + ["affected_services"]:
        vals = signature.get(field, []) or []
        if isinstance(vals, list):
            out[f"num_{field}"] = len(vals)
            out[f"sample_{field}"] = vals[:max_items]
        else:
            out[f"num_{field}"] = 0
            out[f"sample_{field}"] = []
    return out


def _count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, set):
        return len(value)
    if isinstance(value, tuple):
        return len(value)
    return 0
