from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Exact keys that must never appear anywhere inside an agent-visible compressed state.
# Keep these as exact-key checks rather than naive substring checks so that allowed
# redaction markers such as `ground_truth_removed` and `fault_context_removed` do not
# false-positive.
FORBIDDEN_EXACT_KEYS = {
    "fault_context",
    "faulty_service",
    "fault_instances",
    "expected_faulty_services",
    "known_fault_hypotheses",
    "primary_fault",
    "raw_spec",
    "problem_description",
    "ground_truth",
}

# Value-side markers that indicate an oracle/weak-label side channel even when the
# dangerous value is not stored under an obvious key.
FORBIDDEN_VALUE_MARKERS = (
    "scenario_fault_context",
    "generated_fault_context",
    "oracle_ground_truth",
    "oracle_ground_truth_fault",
    "oracle_neighbor_of_",
    "ranked_by_rca_context_or_weak_signals",
    "suspect_silent_failure",
)


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


def _scan_for_redaction_leaks(obj: Any, path: str = "$") -> list[str]:
    leaks: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            child_path = f"{path}.{key_s}"
            if key_s in FORBIDDEN_EXACT_KEYS:
                leaks.append(f"forbidden_key:{child_path}")
            leaks.extend(_scan_for_redaction_leaks(value, child_path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            leaks.extend(_scan_for_redaction_leaks(value, f"{path}[{i}]"))
    elif isinstance(obj, str):
        low = obj.lower()
        for marker in FORBIDDEN_VALUE_MARKERS:
            if marker in low:
                leaks.append(f"forbidden_value:{path}:{marker}")
    return leaks


def assert_redacted_state_safe(compressed: dict[str, Any], scenario_id: str = "") -> None:
    redaction = compressed.get("redaction", {}) or {}
    if not redaction.get("safe_for_rca_agent"):
        raise ValueError(f"{scenario_id}: compressed state is not marked safe_for_rca_agent")

    leaks = _scan_for_redaction_leaks(compressed)
    if leaks:
        raise ValueError(f"{scenario_id}: compressed state has redaction leaks: {leaks[:10]}")


def iter_scenarios(processed_states_dir: str | Path, limit: int | None = None,
                   require_safe_redaction: bool = True,
                   allowed_ids: set[str] | None = None) -> Iterator[ScenarioRecord]:
    root = Path(processed_states_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir()
                  and (d / "state_abstraction.json").exists()
                  and (d / "state_abstraction_compressed.json").exists())
    # Scenario IDs in the processed corpus are directory names. Filter before
    # loading multi-megabyte state JSON, then still verify the embedded ID below.
    if allowed_ids is not None:
        dirs = [d for d in dirs if d.name in allowed_ids]
    if limit is not None:
        dirs = dirs[:limit]
    for d in dirs:
        full = read_json(d / "state_abstraction.json", {})
        comp = read_json(d / "state_abstraction_compressed.json", {})
        sid = full.get("scenario_id") or comp.get("scenario_id") or d.name
        if allowed_ids is not None and str(sid) not in allowed_ids:
            continue
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
