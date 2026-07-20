from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

LEAK_KEYS = ("faulty_service", "fault_instances", "raw_spec")

@dataclass
class ScenarioRecord:
    scenario_id: str
    scenario_dir: Path
    full_state_path: Path
    compressed_state_path: Path
    full_state: dict[str, Any]
    compressed_state: dict[str, Any]


def read_json(path: str | Path, default=None):
    p = Path(path).expanduser()
    if not p.exists():
        return default
    with open(p, "r") as f:
        return json.load(f)


def assert_redacted_state_safe(compressed: dict[str, Any], scenario_id: str = "") -> None:
    redaction = compressed.get("redaction", {}) or {}
    if not redaction.get("safe_for_rca_agent"):
        raise ValueError(f"{scenario_id}: compressed state is not marked safe_for_rca_agent")
    if compressed.get("fault_context") not in (None, {}, ""):
        raise ValueError(f"{scenario_id}: compressed state contains fault_context")
    blob = json.dumps(compressed, sort_keys=True)
    leaks = [k for k in LEAK_KEYS if k in blob]
    if leaks:
        raise ValueError(f"{scenario_id}: compressed state may leak keys: {leaks}")
    if '"ground_truth":' in blob:
        raise ValueError(f"{scenario_id}: compressed state contains ground_truth object")


def iter_scenarios(processed_states_dir: str | Path, limit: int | None = None,
                   require_safe_redaction: bool = True) -> Iterator[ScenarioRecord]:
    root = Path(processed_states_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir()
                  and (d / "state_abstraction.json").exists()
                  and (d / "state_abstraction_compressed.json").exists())
    if limit is not None:
        dirs = dirs[:limit]
    for d in dirs:
        full = read_json(d / "state_abstraction.json", {})
        comp = read_json(d / "state_abstraction_compressed.json", {})
        sid = full.get("scenario_id") or comp.get("scenario_id") or d.name
        if require_safe_redaction:
            assert_redacted_state_safe(comp, sid)
        yield ScenarioRecord(sid, d, d / "state_abstraction.json",
                             d / "state_abstraction_compressed.json", full, comp)


def summarize_dataset(processed_states_dir: str | Path) -> dict[str, Any]:
    total = safe = 0
    unsafe = []
    for rec in iter_scenarios(processed_states_dir, require_safe_redaction=False):
        total += 1
        try:
            assert_redacted_state_safe(rec.compressed_state, rec.scenario_id)
            safe += 1
        except Exception as e:
            unsafe.append({"scenario_id": rec.scenario_id, "error": str(e)})
    return {"processed_states_dir": str(Path(processed_states_dir).expanduser()),
            "num_scenarios": total, "num_safe_for_rca": safe,
            "num_unsafe": len(unsafe), "unsafe_examples": unsafe[:20]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Stage 0 dataset check")
    ap.add_argument("--processed_states", required=True)
    args = ap.parse_args()
    print(json.dumps(summarize_dataset(args.processed_states), indent=2))

if __name__ == "__main__":
    main()
