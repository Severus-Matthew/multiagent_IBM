"""Collect current live controls, then derive thresholds from actual measurements."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from digital_twin_runtime.live_capabilities import LIVE_MECHANISM_CAPABILITIES
from digital_twin_runtime.reward_calibration import application_key, hypothesis_key, write_calibration, canonical_variant
from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
from digital_twin_runtime.targeted_telemetry import MEASUREMENT_CONTRACT
from digital_twin_runtime.telemetry_comparator import score_resolution
from .data_loader import iter_scenarios
from .dataset_integrity import validate_dataset
from .ground_truth import labels_from_full_state
from .schemas import INJECTIBLE_FAULT_MECHANISMS, FaultLabel
from .split_utils import read_scenario_ids


def control_hypotheses(labels, services):
    """Explicit evaluator-only control construction; never a policy candidate menu."""
    targets = {f.service for f in labels}
    alternatives = sorted(set(services) - targets)
    if not alternatives:
        raise ValueError("no distinct service available for matched controls")
    first = labels[0]
    yield "positive", labels, 0
    for root_index, root in enumerate(labels):
        variant_values = LIVE_MECHANISM_CAPABILITIES[root.fault_mechanism].supported_variants
        for index, variant in enumerate(variant_values):
            if canonical_variant(root.fault_mechanism, variant) != canonical_variant(root.fault_mechanism, root.variant_name):
                changed = list(labels); changed[root_index] = replace(root, variant_name=variant, metadata={})
                yield "wrong_variant", changed, root_index * 100 + index
        changed = list(labels); changed[root_index] = replace(root, service=alternatives[0], metadata={})
        yield "wrong_service", changed, root_index
    # Test every implemented alternative mechanism, not a single convenient
    # negative. Unsupported object requirements remain failed control evidence.
    for root_index, root in enumerate(labels):
        for index, mechanism in enumerate(sorted(LIVE_MECHANISM_CAPABILITIES)):
            if mechanism == root.fault_mechanism:
                continue
            wrong = FaultLabel(service=root.service, fault_type=INJECTIBLE_FAULT_MECHANISMS[mechanism],
                               fault_mechanism=mechanism)
            changed = list(labels); changed[root_index] = wrong
            yield "wrong_mechanism", changed, root_index * 100 + index
    yield "extra_root", [*labels, replace(first, service=alternatives[0])], 0
    if len(labels) > 1:
        for index in range(len(labels)):
            yield "missing_root", labels[:index] + labels[index + 1:], index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--dataset_manifest", required=True)
    ap.add_argument("--scenario_ids", required=True, help="Calibration-only IDs")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--source_namespace", required=True, help="Healthy reference application namespace")
    ap.add_argument("--application_source_root", required=True)
    ap.add_argument("--state_abstraction_root", default="state_abstraction_full")
    args = ap.parse_args()
    ids = read_scenario_ids(args.scenario_ids) or set()
    integrity = validate_dataset(args.dataset_manifest, args.processed_states, ids, split="calibration")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (output / "controls.jsonl").open("w") as log:
        for record in iter_scenarios(args.processed_states, allowed_ids=ids):
            labels = labels_from_full_state(record.full_state)
            key = hypothesis_key(labels, application_key(record.compressed_state))
            verifier = SparseLiveTwinVerifier(SparseLiveVerifierConfig(
                source_namespace=args.source_namespace, application_source_root=args.application_source_root,
                state_abstraction_root=str(Path(args.state_abstraction_root).resolve()),
                require_reward_calibration=False, reproduction_threshold=0.0,
                artifact_root=str(output / "telemetry")))
            verifier.prepare_scenario({}, record.compressed_state)
            scope = verifier._incident_spec.services_to_keep
            for control, hypothesis, repeat in control_hypotheses(labels, scope):
                verifier.begin_trajectory(f"control-{record.scenario_id}-{control}-{repeat}")
                row = {"scenario_id": record.scenario_id, "calibration_key": key,
                       "control": control, "repeat": repeat, "measurement_contract": MEASUREMENT_CONTRACT,
                       "environment_sha256": verifier.environment_sha256,
                       "hypothesis": [f.to_dict() for f in hypothesis],
                       "score": 0.0, "telemetry_complete": False, "lifecycle_passed": False}
                try:
                    result = verifier.validate_rca_prediction({}, record.compressed_state, hypothesis)
                    row["result"] = result
                    row["score"] = float(result.get("reproduction_score", 0.0))
                    if not result.get("predicted_fault_injection_checked"):
                        raise RuntimeError(result.get("reason") or "fault injection/observation failed")
                    for handle in reversed(verifier.handles):
                        handle.restore()
                    recovery = verifier.session.wait_for_clean_baseline()
                    recovered = verifier._capture_phase("control_recovered", require_trace_coverage=True)
                    row["telemetry_complete"] = True
                    row["lifecycle_passed"] = bool(recovery.ready and recovered["workload"].completed
                        and not recovered["workload"].failed and not recovered["workload"].application_failures)
                    row["recovery"] = score_resolution(verifier.before_state, recovered["state"])
                    if control == "positive":
                        row["lifecycle_passed"] &= bool(result.get("counterfactual_evidence_gate"))
                        no_fault = {**row, "control": "no_fault", "hypothesis": [],
                                    "score": float(result["clean_reproduction_score"]),
                                    "repeat": 0, "result": {"source": "same_incident_clean_control"}}
                        rows.append(no_fault); log.write(json.dumps(no_fault) + "\n"); log.flush()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    verifier.end_trajectory()
                rows.append(row); log.write(json.dumps(row) + "\n"); log.flush()
    manifest = write_calibration(rows, output / "reward_calibration.json", dataset_sha256=integrity["dataset_sha256"])
    print(json.dumps({"manifest": str(output / "reward_calibration.json"),
                      "qualified_entries": sum(e["eligible"] for e in manifest["entries"].values()),
                      "total_entries": len(manifest["entries"])}))


if __name__ == "__main__":
    main()
