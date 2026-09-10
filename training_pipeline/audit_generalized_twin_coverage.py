from __future__ import annotations

"""Deterministic, train-only coverage audit for generalized sparse-Twin planning."""

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
from typing import Any

from digital_twin_runtime.application_topology import discover_application_topology
from digital_twin_runtime.live_capabilities import assess_live_capability
from digital_twin_runtime.twin_spec_builder import build_sparse_live_twin_spec

from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .split_utils import read_scenario_ids


def _application(services: set[str]) -> str:
    if "nginx-thrift" in services or any(name.endswith("-service") for name in services):
        return "socialNetwork"
    if "frontend" in services and ("consul" in services or any(name.startswith("mongodb-") for name in services)):
        return "hotelReservation"
    if "frontend-proxy" in services or "adservice" in services:
        return "astronomy-shop"
    if "flower" in services:
        return "flower"
    return "unknown"


def _planner_state(state: dict[str, Any], applications_root: Path) -> tuple[dict[str, Any], str]:
    result = copy.deepcopy(state)
    services = {str(item) for item in result.get("services", []) or [] if item}
    application = _application(services)
    source = applications_root / application
    topology = discover_application_topology(source, services)
    graph = result.setdefault("graph", {})
    edges = list(graph.get("edges", []) or [])
    seen = {(str(row.get("src")), str(row.get("dst"))) for row in edges if isinstance(row, dict)}
    for src, dst in topology.edges:
        if (src, dst) not in seen:
            edges.append({"src": src, "dst": dst, "source": topology.source_mode})
    for entrypoint in topology.entrypoints:
        if ("ROOT", entrypoint) not in seen:
            edges.append({"src": "ROOT", "dst": entrypoint, "source": topology.source_mode})
    graph["edges"] = edges
    return result, application


def _select(records: list[Any], count: int, seed: int) -> list[Any]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        labels = labels_from_full_state(record.full_state)
        key = "+".join(sorted({label.fault_mechanism or "unmapped" for label in labels}))
        buckets[key].append(record)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[Any] = []
    keys = sorted(buckets)
    while len(selected) < min(count, len(records)) and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_states", required=True)
    parser.add_argument("--train_ids", required=True)
    parser.add_argument("--applications_root", default="AIOpsLab/aiopslab-applications")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    train_ids = read_scenario_ids(args.train_ids) or set()
    records = [
        record for record in iter_scenarios(args.processed_states, allowed_ids=train_ids)
        if labels_from_full_state(record.full_state)
    ]
    selected = _select(records, args.count, args.seed)
    rows = []
    summary = Counter()
    for record in selected:
        labels = labels_from_full_state(record.full_state)
        planner_state, application = _planner_state(
            record.compressed_state, Path(args.applications_root).resolve()
        )
        spec = build_sparse_live_twin_spec(planner_state, labels)
        roots = {label.service for label in labels}
        kept = set(spec.services_to_keep)
        capabilities = [assess_live_capability(label) for label in labels]
        planning_passed = bool(kept) and roots <= kept and not spec.resource_summary.get("invalid_topology")
        adapter_ready = all(row["supported"] for row in capabilities)
        summary[f"application:{application}"] += 1
        summary["multi_fault" if len(labels) > 1 else "single_fault"] += 1
        summary["planning_passed" if planning_passed else "planning_failed"] += 1
        summary["adapter_ready" if adapter_ready else "adapter_missing"] += 1
        for label in labels:
            summary[f"mechanism:{label.fault_mechanism or 'unmapped'}"] += 1
        rows.append({
            "scenario_id": record.scenario_id, "application": application,
            "num_faults": len(labels), "faults": [label.to_dict() for label in labels],
            "planning_passed": planning_passed, "adapter_ready": adapter_ready,
            "capabilities": capabilities, "services_kept": spec.services_to_keep,
            "resource_summary": spec.resource_summary,
        })
    report = {
        "format": "generalized_twin_coverage_audit_v1", "seed": args.seed,
        "selection_scope": "training_split_only", "requested": args.count,
        "selected": len(selected), "summary": dict(sorted(summary.items())), "cases": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("selected", "summary")}, indent=2))


if __name__ == "__main__":
    main()
