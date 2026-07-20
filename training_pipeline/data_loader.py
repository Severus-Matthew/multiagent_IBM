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
        raise ValueError(f"{scenario_id}: compressed state contains