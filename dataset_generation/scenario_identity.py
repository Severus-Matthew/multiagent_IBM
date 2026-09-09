"""Stable scenario identity including every subfault's parameters."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_spec(spec: dict[str, Any]) -> dict[str, Any]:
    # Preserve execution order: overlapping mutations need not commute.
    fields = ("fault_family", "task", "faulty_service", "app_name", "app", "mode",
              "deployment", "folder_name", "py_file_name", "class_name", "variant")
    out = {k: spec[k] for k in fields if k in spec}
    if spec.get("subproblems"):
        out["subproblems"] = [canonical_spec(s) for s in spec["subproblems"]]
    return out


def scenario_spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(canonical_spec(spec), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def attach_scenario_identity(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    digest = scenario_spec_hash(spec)
    out["scenario_spec_sha256"] = digest
    if spec.get("is_multifault"):
        out["problem_id"] = str(spec["problem_id"]) + "--" + digest[:24]
    return out


def unique_scenarios(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {}
    output = []
    for spec in specs:
        pid, digest = spec["problem_id"], scenario_spec_hash(spec)
        if pid in seen:
            if seen[pid] != digest:
                raise ValueError("distinct scenario specifications share problem_id=" + pid)
            continue
        seen[pid] = digest
        output.append(spec)
    return output
