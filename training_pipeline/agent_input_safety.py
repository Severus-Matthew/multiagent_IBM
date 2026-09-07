from __future__ import annotations

import json
import re
from typing import Any


# Fields that must never be shown to a trainable RCA policy/solver.  This list is
# intentionally broader than the known old leak fields because new state builders
# may add similarly named helper fields later.
BANNED_AGENT_KEYS = {
    "scenario_id",
    "ground_truth",
    "ground_truth_summary",
    "fault_context",
    "fault_instances",
    "faulty_service",
    "faulty_services",
    "primary_fault",
    "known_fault_hypotheses",
    "raw_spec",
    "problem_description",
    "candidate_root_causes",
    "root_cause_candidates",
    "candidate_repair_reason",
    "candidate_root_cause",
    "valid_services",
    "observed_services_sample",
    "all_services",
    # Copied from fault_context in historical compressed files. It is the
    # AIOpsLab problem phase, not telemetry, and must not enter policy prompts.
    "task",
    # build_state.py records wall-clock processing time, not an observed incident
    # timestamp. Dataset generation/processing order can correlate with fault
    # families, so exposing this unique per-record value creates a memorization
    # side channel and makes an identical raw capture produce a different prompt.
    "timestamp",
    # Raw Kubernetes objects (e.g. Endpoints) carry metadata.namespace set to
    # the real source namespace. The opaque Twin namespace is surfaced to
    # agents separately, deliberately, outside this sanitizer; the real one
    # must never reach a policy prompt through a nested telemetry field.
    "namespace",
}

# Generated AIOpsLab IDs encode the hidden family, target, and variant. Nested
# strings such as telemetry file paths often embed the whole ID even after the
# scenario_id key itself has been stripped.
_DESCRIPTIVE_SCENARIO_ID_RE = re.compile(
    r"gen_(?:multifault__|[a-z0-9_]+-(?:detection|localization|analysis|mitigation)-)[A-Za-z0-9_.-]*",
    re.IGNORECASE,
)
_REDACTED_SCENARIO = "[redacted_scenario]"

# In-cluster K8s service DNS names (log/trace targets, dependency edges) take
# the form <service>.<namespace>.svc.cluster.local and embed the real source
# namespace as a literal label. Redact only that label — the service name and
# the "this is a K8s service reference" shape are legitimate diagnostic signal.
_K8S_NAMESPACE_FQDN_RE = re.compile(r"\.[A-Za-z0-9-]+(\.svc\.cluster\.local\b)", re.IGNORECASE)
_REDACTED_NAMESPACE_LABEL = ".[redacted_namespace]"

BANNED_KEY_FRAGMENTS = (
    "ground_truth",
    "fault_context",
    "fault_instance",
    "faulty_service",
    "known_fault",
    "candidate_root",
    "root_cause_candidate",
)

# Strings that indicate a candidate menu rather than raw telemetry.  These are
# allowed in private evaluator files but not in agent-facing input.
BANNED_TEXT_MARKERS = (
    "candidate_root_causes",
    "root_cause_candidates",
    "known_fault_hypotheses",
    "fault_instances",
    "fault_context",
    "faulty_service",
    "ground_truth",
)


def sanitize_agent_state(obj: Any, *, mode: str = "training_safe") -> Any:
    """Return an agent-facing state object with candidate/oracle menus removed.

    The sanitizer preserves redacted telemetry, service health, logs, traces,
    metrics, and graph evidence, but removes oracle labels and generated
    service::fault_type candidate menus.  It is intentionally schema-agnostic so
    it can be reused for AIOpsLab now and ITBench-style environments later.
    """
    if mode == "legacy":
        return obj
    if mode != "training_safe":
        raise ValueError(f"unknown agent input mode {mode!r}; use legacy or training_safe")
    return _sanitize(obj)


def _sanitize_text(text: str) -> str:
    text = _DESCRIPTIVE_SCENARIO_ID_RE.sub(_REDACTED_SCENARIO, text)
    text = _K8S_NAMESPACE_FQDN_RE.sub(lambda m: _REDACTED_NAMESPACE_LABEL + m.group(1), text)
    return text


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            key_s = str(key)
            key_l = key_s.lower()
            if key_l in BANNED_AGENT_KEYS:
                continue
            if any(fragment in key_l for fragment in BANNED_KEY_FRAGMENTS):
                continue
            # high_signal_evidence is allowed only after recursive stripping; it
            # can contain useful aggregate telemetry but must not contain the
            # candidate menu.
            # Dict keys are data too (e.g. dependency-error-count maps keyed by
            # a K8s FQDN target) — sanitize the key text the same way as values.
            out[_sanitize_text(key_s)] = _sanitize(value)
        return out
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, str):
        return _sanitize_text(obj)
    return obj


def agent_input_safety_report(obj: Any) -> dict[str, Any]:
    """Return a lightweight leak/candidate-menu audit for agent-facing input."""
    found_keys: list[str] = []

    def walk(x: Any, path: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                k_s = str(k)
                k_l = k_s.lower()
                p = f"{path}.{k_s}" if path else k_s
                if k_l in BANNED_AGENT_KEYS or any(fragment in k_l for fragment in BANNED_KEY_FRAGMENTS):
                    found_keys.append(p)
                walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x[:1000]):
                walk(v, f"{path}[{i}]")

    walk(obj)
    text = json.dumps(obj, sort_keys=True, default=str)
    text_l = text.lower()
    found_markers = [m for m in BANNED_TEXT_MARKERS if m in text_l]
    found_scenario_ids = sorted({m.group(0) for m in _DESCRIPTIVE_SCENARIO_ID_RE.finditer(text)})
    return {
        "safe_for_training_agent": not found_keys and not found_markers and not found_scenario_ids,
        "banned_key_paths": found_keys[:50],
        "banned_text_markers": found_markers,
        "descriptive_scenario_id_values": found_scenario_ids[:20],
        "serialized_chars": len(text_l),
        "sanitizer_version": "agent_input_safety_v2_no_oracle_no_candidate_menu_no_descriptive_ids",
    }
