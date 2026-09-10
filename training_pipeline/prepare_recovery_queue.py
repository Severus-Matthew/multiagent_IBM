from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .split_utils import read_scenario_ids


def _write(path: Path, values: set[str]) -> str:
    rendered = "\n".join(sorted(values)) + "\n"
    path.write_text(rendered)
    return hashlib.sha256(rendered.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a versioned raw-telemetry recovery queue")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--admitted_train", required=True)
    ap.add_argument("--admitted_test", required=True)
    ap.add_argument("--telemetry_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--version", default="recovered-telemetry-v2")
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"queue version already exists: {out}")
    out.mkdir(parents=True)
    train = read_scenario_ids(args.train) or set()
    test = read_scenario_ids(args.test) or set()
    admitted_train = read_scenario_ids(args.admitted_train) or set()
    admitted_test = read_scenario_ids(args.admitted_test) or set()
    rejected_train = train - admitted_train
    rejected_test = test - admitted_test
    rejected = rejected_train | rejected_test
    telemetry = Path(args.telemetry_dir).resolve()
    available = {p.name for p in telemetry.iterdir() if p.is_dir() and (p / "DONE.json").is_file()}
    recoverable = rejected & available
    missing = rejected - available
    hashes = {
        "rejected_train.txt": _write(out / "rejected_train.txt", rejected_train),
        "rejected_test.txt": _write(out / "rejected_test.txt", rejected_test),
        "reprocess_ids.txt": _write(out / "reprocess_ids.txt", recoverable),
        "missing_raw_telemetry.txt": _write(out / "missing_raw_telemetry.txt", missing),
    }
    manifest = {
        "format": "raw_telemetry_recovery_queue_v1",
        "version": args.version,
        "created_unix": time.time(),
        "inputs": {
            "train": str(Path(args.train).resolve()),
            "test": str(Path(args.test).resolve()),
            "admitted_train": str(Path(args.admitted_train).resolve()),
            "admitted_test": str(Path(args.admitted_test).resolve()),
            "telemetry_dir": str(telemetry),
        },
        "counts": {
            "rejected_train": len(rejected_train),
            "rejected_test": len(rejected_test),
            "rejected_unique": len(rejected),
            "raw_telemetry_available": len(recoverable),
            "raw_telemetry_missing": len(missing),
        },
        "selection_sha256": hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
