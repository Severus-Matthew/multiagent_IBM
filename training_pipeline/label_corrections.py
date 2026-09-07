from __future__ import annotations

"""Evidence-based ground-truth corrections for legacy multi-fault captures.

Several upstream AIOpsLab injectors silently ignore their target argument (see
``live_dataset_admission._LEGACY_AIOPSLAB_EFFECTIVE_TARGETS``). A multi-fault
scenario that combined one such no-op component with a faithful component
therefore recorded a *single-fault* incident: the telemetry reflects only the
component that was actually applied. Keeping both labels teaches the RCA policy
to invent a second root cause that no evidence supports, and makes exact-set
matching unattainable for an incident the Twin can reproduce faithfully.

This module corrects the private evaluator labels (``full_state.fault_context``)
to the components the injector provably applied. It never touches the
agent-visible compressed state, and it is opt-in: a correction manifest is
built once, reviewed, and then passed explicitly to the data loader. Records
whose every component was a no-op are not correctable and are listed for
re-recording instead.

Applying a correction makes the record a single-fault incident. Whether that is
preferable to re-recording the scenario with target-honouring injectors is a
dataset-design decision; the manifest records enough provenance for either.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .data_loader import iter_scenarios
from .ground_truth import labels_from_fault_context, labels_from_full_state
from .live_dataset_admission import (
    LEGACY_NOOP_REASON,
    _LEGACY_AIOPSLAB_EFFECTIVE_TARGETS,
    legacy_label_faithful,
)
from .split_utils import read_scenario_ids

CORRECTION_FORMAT = "legacy_noop_component_label_correction_v1"


def _instance_matches(instance: dict[str, Any], service: str, family: str) -> bool:
    svc = str(instance.get("faulty_service") or instance.get("service") or "")
    fam = str(instance.get("fault_family") or "")
    return svc == service and fam == family


def correction_for_record(record: Any) -> dict[str, Any] | None:
    """Return a correction entry, or None when the labels need no change."""
    context = record.full_state.get("fault_context", {}) or {}
    if record.full_state.get("injection_evidence") or any(
        isinstance(row, dict) and row.get("injection_evidence") is not None
        for row in (context.get("fault_instances") or [])
    ):
        # Generator-recorded evidence supersedes the legacy provider rules.
        return None
    labels = labels_from_fault_context(context)
    if len(labels) < 2:
        return None
    dropped = [label for label in labels if not legacy_label_faithful(label)]
    retained = [label for label in labels if legacy_label_faithful(label)]
    if not dropped or not retained:
        return None
    return {
        "scenario_id": str(record.scenario_id),
        "format": CORRECTION_FORMAT,
        "reason": LEGACY_NOOP_REASON,
        "original_labels": [label.to_dict() for label in labels],
        "dropped": [
            {
                **label.to_dict(),
                "effective_targets": sorted(
                    _LEGACY_AIOPSLAB_EFFECTIVE_TARGETS.get(label.fault_mechanism, set())
                ),
            }
            for label in dropped
        ],
        "retained": [label.to_dict() for label in retained],
        "becomes_single_fault": len(retained) == 1,
    }


def apply_label_correction(full_state: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Return a corrected deep copy of ``full_state``; never mutates the input.

    Refuses a record that carries generator-recorded injection evidence. The
    manifest is built from legacy captures where the injector's target scope is
    inferred from a static mechanism table (``correction_for_record`` applies
    the same check when building it); a regenerated capture already knows,
    per-component, what the injector actually mutated, so silently applying a
    manifest entry meant for the historical capture to a re-recorded one with
    the same scenario id could strip a label the regeneration made faithful.
    """
    context = full_state.get("fault_context", {}) or {}
    if full_state.get("injection_evidence") or any(
        isinstance(row, dict) and row.get("injection_evidence") is not None
        for row in (context.get("fault_instances") or [])
    ):
        raise ValueError(
            f"{entry['scenario_id']}: record carries generator-recorded injection evidence; "
            "label corrections apply only to legacy captures without it"
        )
    corrected = deepcopy(full_state)
    context = corrected.get("fault_context", {}) or {}
    instances = list(context.get("fault_instances") or [])
    retained_keys = {(row["service"], row["fault_family"]) for row in entry["retained"]}
    kept = [
        row for row in instances
        if isinstance(row, dict)
        and any(_instance_matches(row, svc, fam) for svc, fam in retained_keys)
    ]
    if len(kept) != len(entry["retained"]):
        raise ValueError(
            f"{entry['scenario_id']}: fault_instances no longer match the correction manifest"
        )
    context["fault_instances"] = kept
    context["is_multifault"] = len(kept) > 1
    primary = kept[0]
    context["primary_fault"] = primary
    context["faulty_service"] = primary.get("faulty_service")
    context["expected_faulty_services"] = [
        str(row.get("faulty_service")) for row in kept if row.get("faulty_service")
    ]
    if len(kept) == 1:
        context["fault_family"] = primary.get("fault_family", context.get("fault_family"))
        context["variant"] = primary.get("variant", context.get("variant"))
        context["variant_name"] = primary.get("variant_name", context.get("variant_name"))
        context["variant_params"] = primary.get("variant_params", context.get("variant_params"))
    corrected["fault_context"] = context
    corrected["label_correction"] = {
        "format": entry["format"],
        "reason": entry["reason"],
        "dropped": entry["dropped"],
        "original_label_count": len(entry["original_labels"]),
    }
    if labels_from_full_state(corrected) != [
        label for label in labels_from_full_state(full_state) if legacy_label_faithful(label)
    ]:
        raise ValueError(f"{entry['scenario_id']}: corrected labels do not equal the faithful subset")
    return corrected


def build_manifest(processed_states: str, scenario_ids: set[str] | None) -> dict[str, Any]:
    corrections: dict[str, dict[str, Any]] = {}
    uncorrectable: list[dict[str, Any]] = []
    scanned = 0
    for record in iter_scenarios(processed_states, allowed_ids=scenario_ids):
        scanned += 1
        entry = correction_for_record(record)
        if entry is not None:
            corrections[record.scenario_id] = entry
            continue
        labels = labels_from_full_state(record.full_state)
        if labels and not any(legacy_label_faithful(label) for label in labels):
            uncorrectable.append({
                "scenario_id": record.scenario_id,
                "reason": "every_component_was_a_legacy_noop_injection",
                "labels": [label.to_dict() for label in labels],
            })
    return {
        "format": CORRECTION_FORMAT,
        "policy": (
            "drop multi-fault components whose upstream injector provably did not "
            "mutate the labeled service; retain the faithful components"
        ),
        "scanned_records": scanned,
        "corrections": corrections,
        "uncorrectable": uncorrectable,
    }


def load_manifest(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).expanduser().read_text())
    if payload.get("format") != CORRECTION_FORMAT:
        raise ValueError(f"unsupported label correction manifest format: {payload.get('format')!r}")
    return dict(payload.get("corrections") or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", action="append", default=[],
                    help="id list file; may be repeated (train and test)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ids: set[str] | None = None
    for path in args.scenario_ids:
        ids = (ids or set()) | (read_scenario_ids(path) or set())
    manifest = build_manifest(args.processed_states, ids)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    single = sum(1 for row in manifest["corrections"].values() if row["becomes_single_fault"])
    print(json.dumps({
        "output": str(out),
        "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "scanned_records": manifest["scanned_records"],
        "correctable": len(manifest["corrections"]),
        "become_single_fault": single,
        "uncorrectable": len(manifest["uncorrectable"]),
    }, indent=2))


if __name__ == "__main__":
    main()
