from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from .data_loader import read_json
from .label_corrections import apply_label_correction, load_manifest
from .split_utils import read_scenario_ids


REQUIRED_FILES = ("state_abstraction.json", "state_abstraction_compressed.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Create an immutable-by-copy dataset version")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--test_ids", required=True)
    ap.add_argument("--calibration_ids", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--admission_report", required=True)
    ap.add_argument("--override_processed_states", action="append", default=[],
                    help="processed_states root(s) whose records replace the base copy for the same "
                         "scenario id (e.g. regenerated captures); may be repeated, first match wins")
    ap.add_argument("--label_corrections", default=None,
                    help="apply a label-correction manifest to the frozen private states")
    args = ap.parse_args()

    source = Path(args.processed_states).resolve()
    output = Path(args.output_dir).resolve()
    states = output / "processed_states"
    if output.exists():
        raise FileExistsError(f"dataset version already exists and will not be overwritten: {output}")
    states.mkdir(parents=True)

    train = sorted(read_scenario_ids(args.train_ids) or set())
    test = sorted(read_scenario_ids(args.test_ids) or set())
    calibration = sorted(read_scenario_ids(args.calibration_ids) or set())
    if not train or not test or not calibration:
        raise ValueError("train, calibration and test splits must all be nonempty")
    if set(train) & set(test) or set(train) & set(calibration) or set(test) & set(calibration):
        raise ValueError("train/calibration/test overlap")
    corrections = load_manifest(args.label_corrections)
    corrected_ids: list[str] = []
    overrides = [Path(p).resolve() for p in args.override_processed_states]
    provenance: dict[str, str] = {}
    records = []
    for split, ids in (("train", train), ("calibration", calibration), ("test", test)):
        for scenario_id in ids:
            src = source / scenario_id
            for override in overrides:
                candidate = override / scenario_id
                if candidate.is_dir() and all((candidate / name).is_file() for name in REQUIRED_FILES):
                    src = candidate
                    break
            provenance[scenario_id] = str(src.parent)
            if not src.is_dir() or any(not (src / name).is_file() for name in REQUIRED_FILES):
                raise FileNotFoundError(f"incomplete source record: {src}")
            dst = states / scenario_id
            shutil.copytree(src, dst, copy_function=shutil.copy2)
            if scenario_id in corrections and provenance[scenario_id] == str(source):
                # Corrections describe the historical capture; a regenerated
                # record carries generator evidence and its own labels.
                # The frozen private state carries the corrected labels so every
                # consumer of this version sees one ground truth.
                full = read_json(dst / "state_abstraction.json", {})
                corrected = apply_label_correction(full, corrections[scenario_id])
                (dst / "state_abstraction.json").write_text(
                    json.dumps(corrected, indent=2, sort_keys=True) + "\n"
                )
                corrected_ids.append(scenario_id)
            files = []
            for path in sorted(p for p in dst.rglob("*") if p.is_file()):
                files.append({
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
            records.append({"scenario_id": scenario_id, "split": split, "files": files,
                            "source_root": provenance[scenario_id]})

    (output / "train_ids.txt").write_text("\n".join(train) + "\n")
    (output / "calibration_ids.txt").write_text("\n".join(calibration) + "\n")
    (output / "test_ids.txt").write_text("\n".join(test) + "\n")
    shutil.copy2(Path(args.admission_report).resolve(), output / "admission_report.json")
    manifest = {
        "format": "frozen_aiops_dataset_v1",
        "version": args.version,
        "created_unix": time.time(),
        "source": str(source),
        "counts": {"train": len(train), "calibration": len(calibration), "test": len(test), "total": len(records)},
        "train_test_overlap": False,
        "copy_policy": "independent_files_no_symlinks_or_hardlinks",
        "override_roots": [str(p) for p in overrides],
        "records_from_override": sorted(sid for sid, root in provenance.items() if root != str(source)),
        "label_corrections": {
            "manifest": str(Path(args.label_corrections).resolve()) if args.label_corrections else None,
            "manifest_sha256": _sha256(Path(args.label_corrections).resolve()) if args.label_corrections else None,
            "applied_scenario_ids": sorted(corrected_ids),
        },
        "records": records,
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(rendered)
    (output / "manifest.sha256").write_text(
        hashlib.sha256(rendered.encode()).hexdigest() + "  manifest.json\n"
    )
    from .dataset_integrity import validate_dataset
    validate_dataset(output / "manifest.json", states, set(train))
    # Make accidental in-place mutation fail for the normal training user. A new
    # version is created for every salvage/regeneration stage.
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
