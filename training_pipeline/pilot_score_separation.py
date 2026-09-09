"""Pilot diagnostic: do positive, clean and wrong hypotheses separate on fresh captures?

Evaluator-only. For every recorded incident it runs a small, explicit subset of
the calibration controls on the corrected live verifier (positive = private
label, one wrong-service and one wrong-mechanism hypothesis; the clean control
is the Twin's own clean baseline) and reports the reproduction scores with their
per-channel overlaps. It also compares the capture's own clean phase with its
incident phase offline when the recorder kept both. It qualifies nothing: full
calibration needs collect_live_calibration on a frozen calibration split.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from digital_twin_runtime.live_capabilities import LIVE_MECHANISM_CAPABILITIES
from digital_twin_runtime.telemetry_comparator import compare_symptoms_scoped, symptom_signature
from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .schemas import INJECTIBLE_FAULT_MECHANISMS, FaultLabel

WRONG_MECHANISM_PREFERENCE = ("scale_replicas_zero", "network_delay", "pod_kill", "container_kill", "target_port_misconfig")
SCORE_KEYS = ("reproduction_score", "clean_reproduction_score", "counterfactual_evidence_gate", "rca_twin_verified",
              "predicted_fault_injection_checked", "positive_incident_evidence", "incident_scope_coverage_complete",
              "score_reason", "deployment_state_overlap", "degraded_service_overlap", "trace_edge_overlap",
              "log_error_service_overlap", "active_channel_weights", "telemetry_incomplete", "reason")


def pilot_controls(labels: list[FaultLabel], request_path_targets: list[str], scope: list[str],
                   trace_endpoints: list[str] | None = None) -> list[tuple[str, list[FaultLabel]]]:
    """positive, one wrong-service, one wrong-mechanism; deterministic and label-derived.

    The wrong-service control prefers another symptomatic request-path target,
    then a service the incident's traces observed (so the wrong hypothesis is
    at least exercised by the workload), then any scoped service.
    """
    root = labels[0]
    controls: list[tuple[str, list[FaultLabel]]] = [("positive", list(labels))]
    others = ([s for s in request_path_targets if s != root.service]
              or [s for s in sorted(trace_endpoints or []) if s != root.service and s in scope and s != "ROOT"]
              or [s for s in scope if s != root.service])
    if others:
        controls.append(("wrong_service", [replace(root, service=others[0], metadata={})]))
    for mechanism in WRONG_MECHANISM_PREFERENCE:
        if mechanism != root.fault_mechanism and mechanism in LIVE_MECHANISM_CAPABILITIES and mechanism in INJECTIBLE_FAULT_MECHANISMS:
            controls.append(("wrong_mechanism", [FaultLabel(service=root.service, fault_type=INJECTIBLE_FAULT_MECHANISMS[mechanism],
                                                            fault_mechanism=mechanism)]))
            break
    return controls


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: result.get(k) for k in SCORE_KEYS if k in result}
    out["manifested"] = [m.get("manifested") for m in result.get("manifestations", [])]
    out["outside"] = {k: v for k, v in (result.get("unexplained_original_symptoms_outside_scope") or {}).items() if v}
    out["unattributed"] = {k: v[:4] for k, v in (result.get("unattributed_original_symptom_names") or {}).items() if v}
    out["resources_valid"] = (result.get("measured_resources") or {}).get("valid")
    return out


def self_consistency(incident_state: dict[str, Any], clean_state: dict[str, Any], scope: list[str], attributable: list[str]) -> dict[str, Any]:
    """Does the capture's own clean phase score below its incident phase? Offline, no Twin."""
    same = compare_symptoms_scoped(incident_state, incident_state, scope, target_services=scope, attributable_services=attributable)
    clean = compare_symptoms_scoped(incident_state, clean_state, scope, target_services=scope, attributable_services=attributable)
    sig = symptom_signature(incident_state)
    return {"incident_vs_itself": same["reproduction_score"], "incident_vs_own_clean_phase": clean["reproduction_score"],
            "incident_degraded": sig["degraded_services"][:8], "incident_failed_edges": sig["failed_edges"][:8],
            "incident_error_services": len(sig["top_error_services"]), "clean_reason": clean["score_reason"],
            "incident_coverage_complete": same["incident_scope_coverage_complete"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--processed_phases", default=None, help="Recorder's processed_phases/ (clean/recovered abstractions)")
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--aiopslab_applications", default="AIOpsLab/aiopslab-applications")
    ap.add_argument("--state_abstraction_root", default="state_abstraction_full")
    ap.add_argument("--workload_rate", type=int, default=10)
    ap.add_argument("--workload_duration_seconds", type=int, default=150)
    ap.add_argument("--controls", default="positive,wrong_service,wrong_mechanism")
    args = ap.parse_args()
    from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    wanted = None
    if args.scenario_ids:
        wanted = {l.strip() for l in Path(args.scenario_ids).read_text().splitlines() if l.strip()}
    selected_controls = set(args.controls.split(","))
    log = (output / "log.jsonl").open("a")
    rows: list[dict[str, Any]] = []
    for record in iter_scenarios(args.processed_states, allowed_ids=wanted):
        labels = labels_from_full_state(record.full_state)
        app = "hotel" if "hotel" in record.scenario_id else "social"
        cfg = SparseLiveVerifierConfig(
            source_namespace="test-hotel-reservation" if app == "hotel" else "test-social-network",
            application_source_root=str(Path(args.aiopslab_applications, "hotelReservation" if app == "hotel" else "socialNetwork").resolve()),
            state_abstraction_root=str(Path(args.state_abstraction_root).resolve()), require_reward_calibration=False,
            reproduction_threshold=0.0, workload_rate=args.workload_rate, workload_duration_seconds=args.workload_duration_seconds,
            artifact_root=str(output / "twin_artifacts" / record.scenario_id))
        verifier = SparseLiveTwinVerifier(cfg)
        row: dict[str, Any] = {"scenario_id": record.scenario_id, "labels": [l.to_dict() for l in labels], "controls": {}}
        try:
            verifier.prepare_scenario({}, record.compressed_state)
            summary = verifier._incident_spec.resource_summary
            row["scope"] = {"kept": len(verifier._incident_spec.services_to_keep), "deployable": len(summary["deployable_services"]),
                            "request_path_targets": summary["incident_request_path_targets"],
                            "trace_observable_targets": summary["incident_trace_observable_targets"],
                            "unattributed": summary["unattributed_symptom_names"], "reduction_percent": summary.get("service_reduction_percent")}
            if args.processed_phases:
                clean_path = Path(args.processed_phases, record.scenario_id, "clean", "state_abstraction_compressed.json")
                if clean_path.is_file():
                    row["self_consistency"] = self_consistency(verifier._incident_state, json.loads(clean_path.read_text()),
                                                               verifier._incident_spec.services_to_keep, summary["deployable_services"])
            trace_endpoints = sorted({str(x) for e in ((verifier._incident_state.get("traces") or {}).get("per_edge") or {}).values()
                                      if isinstance(e, dict) for x in (e.get("source"), e.get("target")) if x})
            for name, hypothesis in pilot_controls(labels, summary["incident_request_path_targets"],
                                                   verifier._incident_spec.services_to_keep, trace_endpoints):
                if name not in selected_controls:
                    continue
                verifier.begin_trajectory(f"pilot-{record.scenario_id}-{name}"); started = time.time()
                try:
                    result = verifier.validate_rca_prediction({}, record.compressed_state, hypothesis)
                    row["controls"][name] = {"hypothesis": [h.to_dict() for h in hypothesis], "seconds": round(time.time() - started, 1), **_summary(result)}
                except Exception as exc:  # noqa: BLE001 - one control must not kill the pilot
                    row["controls"][name] = {"hypothesis": [h.to_dict() for h in hypothesis], "error": f"{type(exc).__name__}: {exc}"}
                finally:
                    verifier.end_trajectory()
                print(json.dumps({"scenario": record.scenario_id[:60], "control": name, **{k: row["controls"][name].get(k) for k in ("reproduction_score", "clean_reproduction_score", "counterfactual_evidence_gate", "error")}}), flush=True)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"; row["traceback"] = traceback.format_exc()[-1500:]
            try:
                verifier.end_trajectory()
            except Exception:  # noqa: BLE001
                pass
        rows.append(row); log.write(json.dumps(row, default=str) + "\n"); log.flush()
    (output / "separation.json").write_text(json.dumps(rows, indent=2, default=str))
    lines = ["| scenario | control | score | clean | gate | dep | degraded | edges | logs | reason |", "|---|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        for name, c in row.get("controls", {}).items():
            lines.append(f"| {row['scenario_id'][:48]} | {name} | {c.get('reproduction_score')} | {c.get('clean_reproduction_score')} | "
                         f"{c.get('counterfactual_evidence_gate')} | {c.get('deployment_state_overlap')} | {c.get('degraded_service_overlap')} | "
                         f"{c.get('trace_edge_overlap')} | {c.get('log_error_service_overlap')} | {c.get('score_reason') or c.get('error')} |")
    (output / "separation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
