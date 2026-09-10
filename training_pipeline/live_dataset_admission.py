from __future__ import annotations

"""Evidence gates for using historical incidents as live-Twin RL targets.

The preferred contract is generator-produced ``injection_evidence`` on every
fault instance. Legacy AIOpsLab captures predate that contract, so their known
injector scope is represented by a versioned provider adapter here—not inside
the generic Twin planner/verifier. A future ITBench importer supplies its own
evidence and does not inherit these AIOpsLab service rules.

Two independent questions are answered per record:

1. Source faithfulness: did the upstream injector actually mutate the labeled
   service? Several upstream injectors ignore their target argument, so a label
   can name a service that was never touched. Such a capture is a healthy system
   under a fault label and cannot be reproduced by any Twin.
2. Comparable evidence: does the capture carry a symptom signature, in a channel
   the Twin comparison scores, for the labeled mechanism? Trace collection failed
   for a fraction of the corpus (header-only Jaeger exports), but structural
   mechanisms such as scale-to-zero or unschedulable pods leave their signature in
   Deployment/endpoint state, which every capture records. Those incidents are
   still fully comparable; the comparator skips the channel the capture never
   observed (see ``telemetry_comparator.observed_channels``).

Evidence strength is reported so a caller can choose the strict default
(structural or trace evidence at the labeled target) or opt into "weak" evidence
(service-health flags and log error counts only), which the corpus shows to be
noisy: several datastore services carry those flags in every capture.
"""

import re
from typing import Any

from digital_twin_runtime.live_capabilities import LIVE_MECHANISM_CAPABILITIES
from digital_twin_runtime.telemetry_comparator import (
    _norm_service,
    _service_aliases,
    observed_channels,
    symptom_signature,
)

from .ground_truth import labels_from_full_state
from .schemas import FaultLabel, normalize_fault_mechanism


_LEGACY_AIOPSLAB_EFFECTIVE_TARGETS: dict[str, set[str]] = {
    "mongodb_auth_missing": {"url-shorten-mongodb"},
    "mongodb_auth_revoked": {"mongodb-rate", "mongodb-geo"},
    "mongodb_user_unregistered": {"mongodb-rate", "mongodb-geo"},
    "application_config_misconfig": {"geo"},
    "wrong_binary": {"profile"},
}

LEGACY_NOOP_REASON = "legacy_aiopslab_injector_did_not_mutate_labeled_target"

EVIDENCE_STRONG = "strong"
EVIDENCE_WEAK = "weak"
EVIDENCE_NONE = "none"


def has_collected_trace_edges(compressed_state: dict[str, Any]) -> bool:
    return observed_channels(compressed_state)["traces"]


def legacy_label_faithful(label: FaultLabel) -> bool:
    """Whether the upstream AIOpsLab injector mutated this label's service."""
    allowed = _LEGACY_AIOPSLAB_EFFECTIVE_TARGETS.get(label.fault_mechanism)
    return allowed is None or label.service in allowed


def source_injection_evidence(full_state: dict[str, Any]) -> dict[str, Any]:
    root_evidence = full_state.get("injection_evidence") or {}
    root_rows = root_evidence.get("faults", []) if isinstance(root_evidence, dict) else []
    if root_rows:
        failures = [
            row for row in root_rows
            if not isinstance(row, dict) or not row.get("mutated") or not row.get("manifested")
        ]
        return {
            "valid": bool(root_evidence.get("verified")) and not failures,
            "mode": "generator_recorded_mutation_and_manifestation_evidence",
            "failures": failures,
        }
    context = full_state.get("fault_context", {}) or {}
    instances = context.get("fault_instances") or []
    explicit = [row.get("injection_evidence") for row in instances if isinstance(row, dict)]
    if explicit and any(row is not None for row in explicit):
        failures = [row for row in explicit if not isinstance(row, dict) or not row.get("mutated")]
        return {
            "valid": not failures,
            "mode": "generator_recorded_injection_evidence",
            "failures": failures,
        }

    failures = []
    for label in labels_from_full_state(full_state):
        if not legacy_label_faithful(label):
            failures.append({
                "mechanism": label.fault_mechanism,
                "labeled_service": label.service,
                "effective_targets": sorted(_LEGACY_AIOPSLAB_EFFECTIVE_TARGETS[label.fault_mechanism]),
                "reason": LEGACY_NOOP_REASON,
            })
    correction = full_state.get("label_correction") or {}
    return {
        "valid": not failures,
        "mode": (
            "legacy_aiopslab_source_scope_v1_with_label_correction"
            if correction else "legacy_aiopslab_source_scope_v1"
        ),
        "failures": failures,
    }


def _system_entry(compressed_state: dict[str, Any], service: str) -> dict[str, Any] | None:
    """The ``system`` entry for ``service``, preferring an exact name match.

    ``_service_aliases`` maps ``mongodb-<x>`` to ``<x>`` so the alias-based
    fallback can find a service recorded under a slightly different spelling.
    But a hotel capture lists ``rate`` and ``mongodb-rate`` as separate,
    genuinely different services, and ``mongodb-rate``'s aliases include
    ``rate`` — matching on alias overlap alone can therefore return the
    datastore's entry for an app-service label (or vice versa). An exact key
    always wins before alias overlap is considered.
    """
    system = compressed_state.get("system") or {}
    if not isinstance(system, dict):
        return None
    if isinstance(system.get(service), dict):
        return system[service]
    target = _norm_service(service)
    for name, info in system.items():
        if isinstance(info, dict) and _norm_service(name) == target:
            return info
    aliases = _service_aliases(service)
    for name, info in system.items():
        if isinstance(info, dict) and _service_aliases(name) & aliases:
            return info
    return None


def _structural_signature(info: dict[str, Any]) -> dict[str, Any]:
    """Deployment/endpoint tokens for one service and whether they are anomalous."""
    deployment = info.get("deployment", {}) or {}
    endpoints = info.get("endpoints", {}) or {}
    health = info.get("health", {}) or {}
    tokens: dict[str, Any] = {}
    for key in (
        "replicas_desired", "replicas_current", "replicas_ready",
        "replicas_available", "replicas_unavailable",
    ):
        if key in deployment:
            tokens[key] = deployment.get(key)
    if "ready_endpoint_count" in endpoints:
        tokens["ready_endpoint_count"] = endpoints.get("ready_endpoint_count")
    if health.get("status"):
        tokens["health_status"] = health.get("status")

    def _num(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    desired = _num(tokens.get("replicas_desired"))
    ready = _num(tokens.get("replicas_ready"))
    unavailable = _num(tokens.get("replicas_unavailable"))
    endpoints_ready = _num(tokens.get("ready_endpoint_count"))
    status = str(tokens.get("health_status") or "").lower()
    anomalous = bool(
        (unavailable or 0) > 0
        or (desired is not None and ready is not None and ready < desired)
        or (desired is not None and desired == 0)
        or (endpoints_ready is not None and endpoints_ready == 0)
        or (status and status not in {"healthy", "unknown"})
        or (_num(health.get("pods_unready")) or 0) > 0
        or (_num(health.get("crashloop_count")) or 0) > 0
    )
    return {"tokens": tokens, "anomalous": anomalous, "recorded": bool(tokens)}


_SCALE_VARIANT_RE = re.compile(r"^scale_(\d+)$")


def _expected_scale_replicas(variant_name: str) -> int | None:
    match = _SCALE_VARIANT_RE.fullmatch(str(variant_name or ""))
    return int(match.group(1)) if match else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def comparable_symptom_evidence(
    compressed_state: dict[str, Any], labels: list[FaultLabel]
) -> dict[str, Any]:
    """Classify the evidence a capture holds for its labeled faults.

    ``strong``: the trace channel was collected, or the mechanism is structural
    and the labeled target's Deployment/endpoint state is recorded (and anomalous,
    except for replica-count variants whose signature is the count itself).
    ``weak``: only service-health flags or log error counts at the target.
    ``none``: nothing at the target in any scored channel.
    """
    channels = observed_channels(compressed_state)
    signature = symptom_signature(compressed_state)
    degraded = {x for s in signature["degraded_services"] for x in _service_aliases(s)}
    log_errors = {x for s in signature["top_error_services"] for x in _service_aliases(s)}
    per_label: list[dict[str, Any]] = []
    strengths: list[str] = []
    for label in labels:
        mechanism = normalize_fault_mechanism(label.fault_mechanism) or str(label.fault_mechanism)
        capability = LIVE_MECHANISM_CAPABILITIES.get(mechanism)
        mechanism_channels = capability.evidence_channels if capability else ("traces", "logs")
        aliases = _service_aliases(label.service)
        entry = _system_entry(compressed_state, label.service)
        structural = _structural_signature(entry) if entry else {"tokens": {}, "anomalous": False, "recorded": False}
        # Replica-count variants (scale_2, scale_3) are not "anomalous" in
        # isolation; their signature is the recorded count matching what the
        # labeled variant actually requested, not merely that some Deployment
        # tokens were recorded (which is true of nearly every service).
        expected_replicas = _expected_scale_replicas(label.variant_name)
        recorded_replicas = structural["tokens"].get("replicas_desired")
        count_variant = (
            mechanism == "scale_replicas_zero"
            and expected_replicas is not None
            and recorded_replicas is not None
            and _safe_int(recorded_replicas) == expected_replicas
        )
        structural_ok = (
            "structural" in mechanism_channels
            and channels["system"]
            and (structural["anomalous"] or count_variant)
        )
        weak_ok = bool(aliases & degraded) or bool(aliases & log_errors)
        if channels["traces"]:
            strength, basis = EVIDENCE_STRONG, "collected_trace_edges"
        elif structural_ok:
            strength, basis = EVIDENCE_STRONG, "structural_target_state"
        elif weak_ok:
            strength, basis = EVIDENCE_WEAK, "service_health_or_log_errors_only"
        else:
            strength, basis = EVIDENCE_NONE, "no_target_evidence_in_scored_channels"
        strengths.append(strength)
        per_label.append({
            "service": label.service,
            "mechanism": mechanism,
            "mechanism_evidence_channels": list(mechanism_channels),
            "strength": strength,
            "basis": basis,
            "structural": structural,
            "degraded_flag": bool(aliases & degraded),
            "log_error_flag": bool(aliases & log_errors),
        })
    order = {EVIDENCE_STRONG: 2, EVIDENCE_WEAK: 1, EVIDENCE_NONE: 0}
    overall = (
        min(strengths, key=lambda s: order[s]) if strengths else EVIDENCE_NONE
    )
    return {
        "strength": overall,
        "observed_channels": channels,
        "per_label": per_label,
        "policy": "structural_or_trace_evidence_at_labeled_target_v1",
    }


def assess_record_for_live_reward(
    record: Any, *, admit_weak_evidence: bool = False
) -> dict[str, Any]:
    source = source_injection_evidence(record.full_state)
    labels = labels_from_full_state(record.full_state)
    evidence = comparable_symptom_evidence(record.compressed_state, labels)
    traced = evidence["observed_channels"]["traces"]
    reasons = []
    if not source["valid"]:
        reasons.append("source_injection_not_faithful_to_label")
    if evidence["strength"] == EVIDENCE_NONE:
        reasons.append("historical_capture_has_no_comparable_symptom_evidence")
    elif evidence["strength"] == EVIDENCE_WEAK and not admit_weak_evidence:
        reasons.append("historical_capture_has_only_weak_symptom_evidence")
    return {
        "scenario_id": str(record.scenario_id),
        "eligible": not reasons,
        "reasons": reasons,
        "source_injection": source,
        "has_collected_trace_edges": traced,
        "comparable_evidence": evidence,
        "label_correction": (record.full_state.get("label_correction") or None),
    }
