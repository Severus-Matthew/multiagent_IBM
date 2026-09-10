"""Content-verified, explicit train/calibration/test admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .agent_input_safety import agent_input_safety_report
from .ground_truth import labels_from_full_state
from .split_utils import read_scenario_ids

ABSTRACTION_CONTRACT = "raw_spans_public_state_metric_units_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(manifest_path: str | Path, processed_states: str | Path, selected_ids: set[str], *,
                     split: str = "train", calibration_path: str | None = None) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    root = path.parent
    digest = file_sha256(path)
    declared = (root / "manifest.sha256").read_text().split()[0]
    if declared != digest:
        raise ValueError("frozen dataset manifest hash mismatch")
    if Path(processed_states).resolve() != root / "processed_states":
        raise ValueError("processed_states must be the manifest's frozen content directory")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "frozen_aiops_dataset_v1":
        raise ValueError("unsupported dataset format")
    memberships: dict[str, set[str]] = {"train": set(), "calibration": set(), "test": set()}
    ids_seen: set[str] = set()
    fingerprints: dict[str, str] = {}
    families: dict[str, str] = {}
    for record in manifest.get("records", []):
        sid, side = str(record["scenario_id"]), str(record["split"])
        if side not in memberships or sid in ids_seen or Path(sid).name != sid:
            raise ValueError("duplicate, invalid, or overlapping dataset membership")
        ids_seen.add(sid); memberships[side].add(sid)
        listed = set()
        for entry in record["files"]:
            raw_path = root / entry["path"]
            actual = raw_path.resolve()
            if not actual.is_relative_to(root / "processed_states" / sid) or raw_path.is_symlink():
                raise ValueError("dataset content escapes its incident directory")
            if actual.stat().st_size != entry["bytes"] or file_sha256(actual) != entry["sha256"]:
                raise ValueError("frozen content hash mismatch: " + entry["path"])
            listed.add(actual.name)
        if not {"state_abstraction.json", "state_abstraction_compressed.json"}.issubset(listed):
            raise ValueError("manifest does not bind both incident views")
        directory = root / "processed_states" / sid
        public = json.loads((directory / "state_abstraction_compressed.json").read_text())
        if public.get("abstraction_contract") != ABSTRACTION_CONTRACT:
            raise ValueError("rebuild abstractions with current raw-span/public-state/metric contracts")
        if not agent_input_safety_report(public)["safe_for_training_agent"]:
            raise ValueError("standalone compressed state fails the public-input boundary")
        fingerprint = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
        if fingerprint in fingerprints and fingerprints[fingerprint] != side:
            raise ValueError("identical observable capture appears in different splits")
        fingerprints[fingerprint] = side
        full = json.loads((directory / "state_abstraction.json").read_text())
        # Task and parameter variants of a service/mechanism combination stay
        # together. Sharing a component across distinct joint faults is allowed.
        labels = labels_from_full_state(full)
        family = json.dumps(sorted((f.service, f.fault_mechanism) for f in labels))
        if labels and family in families and families[family] != side:
            raise ValueError("incident mechanism/task/parameter variants cross split boundaries")
        if labels:
            families[family] = side
    if not selected_ids or not selected_ids.issubset(memberships.get(split, set())):
        raise ValueError("explicit scenario selection must be a nonempty subset of the " + split + " split")
    if manifest.get("counts", {}).get("total") != len(ids_seen):
        raise ValueError("dataset manifest count mismatch")
    calibration_digest = None
    if calibration_path:
        from digital_twin_runtime.reward_calibration import load_calibration
        calibration, calibration_digest = load_calibration(calibration_path)
        if calibration.get("dataset_sha256") != digest:
            raise ValueError("calibration belongs to a different frozen dataset")
        control_ids = {str(r["scenario_id"]) for r in calibration["controls"]}
        if not control_ids or not control_ids.issubset(memberships["calibration"]):
            raise ValueError("live controls must use only the held-out calibration split")
    return {"dataset_sha256": digest, "calibration_sha256": calibration_digest,
            "counts": {k: len(v) for k, v in memberships.items()},
            "selection_count": len(selected_ids), "split": split,
            "selection_sha256": hashlib.sha256(json.dumps(sorted(selected_ids)).encode()).hexdigest(),
            "abstraction_contract": ABSTRACTION_CONTRACT}


def validate_training_inputs(args: Any) -> dict[str, Any]:
    if args.twin_mode in {"live", "hybrid"} and not args.allow_uncalibrated_live_reward and not args.reward_calibration:
        raise ValueError("production live training requires --reward_calibration from matched held-out controls")
    if args.temperature != 1.0 or args.top_p != 1.0:
        raise ValueError("GRPO raw_softmax_v1 requires --temperature 1 --top_p 1")
    if args.label_corrections:
        raise ValueError("apply label corrections before freezing; training cannot mutate frozen labels")
    report = validate_dataset(args.dataset_manifest, args.processed_states,
                              read_scenario_ids(args.scenario_ids) or set(),
                              calibration_path=args.reward_calibration)
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        candidates = [checkpoint.parent / "run_manifest.json", checkpoint.parent.parent / "run_manifest.json"]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ValueError("resume requires the checkpoint's run manifest")
        previous = json.loads(source.read_text())
        if previous.get("sampling_contract") != "raw_softmax_v1" or previous.get("dataset_integrity") != report:
            raise ValueError("checkpoint sampling/dataset contract differs; start a new qualified run")
    return report
