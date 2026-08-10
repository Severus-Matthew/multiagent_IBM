from __future__ import annotations

import json
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
}

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
            out[key_s] = _sanitize(value)
        return out
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
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
    text = json.dumps(obj, sort_keys=True, default=str).lower()
    found_markers = [m for m in BANNED_TEXT_MARKERS if m in text]
    return {
        "safe_for_training_agent": not found_keys and not found_markers,
        "banned_key_paths": found_keys[:50],
        "banned_text_markers": found_markers,
        "serialized_chars": len(text),
        "sanitizer_version": "agent_input_safety_v1_no_oracle_no_candidate_menu",
    }
