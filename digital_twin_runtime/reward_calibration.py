"""Versioned live controls for single faults, variants, and joint hypotheses.

No threshold is built in. Controls are collected independently of policy training;
entries are recomputed from their evidence whenever a manifest is loaded.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from training_pipeline.schemas import FaultLabel
from .targeted_telemetry import MEASUREMENT_CONTRACT

FORMAT = "matched_live_reward_controls_v1"
MIN_MATCHED_INCIDENTS = 3


def canonical_variant(mechanism: str, variant: str | None) -> str:
    value = variant or "default"
    if value == "default":
        return {"scale_replicas_zero": "scale_0", "network_delay": "delay_1000ms"}.get(mechanism, value)
    return value


def application_key(state: dict[str, Any]) -> str:
    services = sorted(set(str(s) for s in state.get("services", []) if s))
    if not services:
        raise ValueError("application calibration needs an observable service inventory")
    return hashlib.sha256(json.dumps(services).encode()).hexdigest()


def hypothesis_key(labels: list[FaultLabel], app_key: str) -> str:
    if not labels or any(not f.is_injectible() for f in labels):
        raise ValueError("calibration requires injectible mechanism/variant hypotheses")
    # Retain multiplicity: two roots with the same mechanism are a joint fault.
    mechanisms = sorted((f.fault_mechanism, canonical_variant(f.fault_mechanism, f.variant_name)) for f in labels)
    return json.dumps([app_key, mechanisms], separators=(",", ":"))


def required_controls(num_faults: int, mechanisms=()) -> set[str]:
    required = {"positive", "no_fault", "wrong_service", "wrong_mechanism", "extra_root"}
    if set(mechanisms) & {"scale_replicas_zero", "network_delay", "network_loss"}:
        required.add("wrong_variant")
    return required | ({"missing_root"} if num_faults > 1 else set())


def derive_entries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities = set()
    for row in rows:
        key = str(row["calibration_key"])
        identity = (key, row["scenario_id"], row["control"], row.get("repeat", 0))
        if identity in identities:
            raise ValueError("duplicate calibration control identity")
        identities.add(identity)
        score = float(row.get("score", float("nan")))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("calibration score must be finite and in [0, 1]")
        grouped[key].append(row)
    # Any admitted hypothesis must reject every measured negative for this
    # application, including negatives generated under another mechanism key.
    # Otherwise a wrong B hypothesis on an A incident could be compared against
    # B's unrelated, weaker negative threshold.
    application_negatives: dict[str, list[float]] = defaultdict(list)
    for key, controls in grouped.items():
        for row in controls:
            if row["control"] != "positive":
                application_negatives[json.loads(key)[0]].append(float(row["score"]))
    entries = {}
    for key, controls in grouped.items():
        parsed = json.loads(key)
        required = required_controls(len(parsed[1]), [m for m, _ in parsed[1]])
        by_incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in controls:
            by_incident[str(row["scenario_id"])].append(row)
        reasons = []
        environments = {r.get("environment_sha256") for r in controls}
        if None in environments or len(environments) != 1:
            reasons.append("missing_or_changed_reference_environment")
        if len(by_incident) < MIN_MATCHED_INCIDENTS:
            reasons.append("insufficient_matched_incidents")
        for incident, group in by_incident.items():
            if not required.issubset({r["control"] for r in group}):
                reasons.append("missing_matched_controls:" + incident)
            if any(r.get("measurement_contract") != MEASUREMENT_CONTRACT or
                   not r.get("lifecycle_passed") or not r.get("telemetry_complete") for r in group):
                reasons.append("unqualified_control_lifecycle:" + incident)
        positive = [float(r["score"]) for r in controls if r["control"] == "positive"]
        low = min(positive, default=0.0)
        high = max(application_negatives[parsed[0]], default=1.0)
        if low <= high:
            reasons.append("positive_negative_scores_not_separated")
        entries[key] = {"eligible": not reasons, "reasons": reasons,
                        "threshold": (low + high) / 2 if not reasons else None,
                        "minimum_positive": low, "maximum_negative": high,
                        "matched_incidents": len(by_incident), "control_count": len(controls),
                        "environment_sha256": next(iter(environments)) if len(environments) == 1 else None,
                        "required_controls": sorted(required)}
    return entries


def write_calibration(rows: list[dict[str, Any]], path: str | Path, *, dataset_sha256: str) -> dict[str, Any]:
    manifest = {"format": FORMAT, "measurement_contract": MEASUREMENT_CONTRACT,
                "dataset_sha256": dataset_sha256, "controls": rows, "entries": derive_entries(rows)}
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_calibration(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != FORMAT or manifest.get("measurement_contract") != MEASUREMENT_CONTRACT:
        raise ValueError("live controls must be recollected with the current measurement contract")
    if manifest.get("entries") != derive_entries(manifest.get("controls") or []):
        raise ValueError("calibration entries disagree with their matched controls")
    return manifest, hashlib.sha256(raw).hexdigest()


def assess_calibration(labels: list[FaultLabel], path: str | None, state: dict[str, Any] | None) -> dict[str, Any]:
    if not path or not state:
        return {"eligible": False, "reason": "current_live_calibration_manifest_required"}
    manifest, digest = load_calibration(path)
    key = hypothesis_key(labels, application_key(state))
    entry = manifest["entries"].get(key)
    if not entry or not entry["eligible"]:
        return {"eligible": False, "reason": "hypothesis_has_no_qualified_matched_controls",
                "calibration_key": key, "manifest_sha256": digest,
                "qualification": entry}
    return {**entry, "eligible": True, "calibration_key": key,
            "manifest_sha256": digest, "measurement_contract": MEASUREMENT_CONTRACT}
