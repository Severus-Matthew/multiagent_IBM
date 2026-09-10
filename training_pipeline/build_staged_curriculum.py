from __future__ import annotations

"""Build the versioned 816-record curriculum without scenario-id allowlists.

Stage membership is derived from mechanism capability and an audit of the
original AIOpsLab injector's effective domain. Runtime replay remains generic;
these source-domain rules exist only to prevent training against mislabeled
historical telemetry and disappear once Stage C is regenerated.
"""

import argparse
import json
from pathlib import Path
from typing import Any


STAGE_A_MECHANISMS = {
    "assign_to_non_existent_node", "scale_replicas_zero", "target_port_misconfig",
}


def _source_issue(fault: dict[str, Any]) -> str | None:
    mechanism = str(fault.get("fault_mechanism") or "")
    service = str(fault.get("service") or "")
    if mechanism == "mongodb_auth_missing" and service != "url-shorten-mongodb":
        return "original_tls_injector_noop_for_target"
    if mechanism in {"mongodb_auth_revoked", "mongodb_user_unregistered"} and service not in {
        "mongodb-rate", "mongodb-geo",
    }:
        return "original_mongodb_user_role_injector_noop_for_target"
    if mechanism == "wrong_binary" and service != "profile":
        return "original_wrong_binary_mutation_noop_for_target_command"
    if mechanism == "container_stop":
        return "docker_flower_runtime_requires_separate_twin_adapter"
    if not mechanism:
        return "healthy_noop_control_not_fault_injection"
    return None


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage_report", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    report = json.loads(Path(args.coverage_report).read_text(encoding="utf-8"))
    cases = list(report.get("cases") or [])
    if len(cases) != 816:
        raise ValueError(f"expected 816 training cases, found {len(cases)}")

    stage_a: list[str] = []
    stage_b: list[str] = []
    stage_c: list[str] = []
    audit: list[dict[str, Any]] = []
    for case in cases:
        scenario_id = str(case["scenario_id"])
        faults = list(case.get("faults") or [])
        issues = [issue for fault in faults if (issue := _source_issue(fault))]
        mechanisms = {str(fault.get("fault_mechanism") or "") for fault in faults}
        if issues:
            stage = "C"
            stage_c.append(scenario_id)
        elif mechanisms and mechanisms <= STAGE_A_MECHANISMS:
            stage = "A"
            stage_a.append(scenario_id)
        else:
            stage = "B"
            stage_b.append(scenario_id)
        audit.append({
            "scenario_id": scenario_id, "stage": stage,
            "mechanisms": sorted(mechanisms), "source_issues": sorted(set(issues)),
        })

    expected = (216, 264, 336)
    actual = (len(stage_a), len(stage_b), len(stage_c))
    if actual != expected:
        raise AssertionError(f"unexpected curriculum counts: {actual}, expected {expected}")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_ids(out / "stage_a_216.txt", stage_a)
    _write_ids(out / "stage_b_add_264.txt", stage_b)
    _write_ids(out / "stage_c_add_336.txt", stage_c)
    _write_ids(out / "cumulative_ab_480.txt", stage_a + stage_b)
    _write_ids(out / "full_816.txt", stage_a + stage_b + stage_c)
    manifest = {
        "format": "aiops_staged_curriculum_v1",
        "counts": {"stage_a": 216, "stage_b_add": 264, "stage_c_add": 336, "full": 816},
        "schedule": [
            "stage_a_once", "stage_b_new_records_once", "stage_c_new_records_once",
            "full_816_epoch_2", "full_816_epoch_3",
        ],
        "stage_a_mechanism_classes": sorted(STAGE_A_MECHANISMS),
        "note": "Stage B is 264, not 236; the requested counts otherwise omitted 28 records.",
        "records": audit,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
