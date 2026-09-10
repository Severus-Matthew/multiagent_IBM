"""Real lifecycle evidence for the corrected verifier on this cluster.

Evaluator-only: the incident's private label is used to build the positive
control and a wrong-service control, exactly as collect_live_calibration does.
No policy is trained and nothing is written to the source namespaces.
"""
import json, sys, time, traceback
from dataclasses import replace
from pathlib import Path
ROOT = sys.argv[1]; sys.path.insert(0, ROOT)
SID = sys.argv[2]; OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
from training_pipeline.data_loader import iter_scenarios
from training_pipeline.ground_truth import labels_from_full_state
from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
log = (OUT / "log.jsonl").open("a")
def emit(event, **kw):
    row = {"t": round(time.time(), 1), "event": event, **kw}
    log.write(json.dumps(row, default=str) + "\n"); log.flush(); print(json.dumps(row, default=str)[:900], flush=True)
def summary(result):
    keys = ("reproduction_score", "clean_reproduction_score", "counterfactual_evidence_gate", "rca_twin_verified",
            "predicted_fault_injection_checked", "positive_incident_evidence", "incident_scope_coverage_complete",
            "score_reason", "decision_threshold", "telemetry_incomplete", "reason", "services_selected",
            "service_reduction_percent", "live_reward_calibrated", "uncalibrated_reward_override_used")
    out = {k: result.get(k) for k in keys if k in result}
    out["manifested"] = [m.get("manifested") for m in result.get("manifestations", [])]
    out["outside"] = {k: v for k, v in (result.get("unexplained_original_symptoms_outside_scope") or result.get("original_symptoms_outside_scope") or {}).items() if v}
    out["unattributed"] = {k: v[:4] for k, v in (result.get("unattributed_original_symptom_names") or {}).items() if v}
    out["resources_valid"] = (result.get("measured_resources") or {}).get("valid")
    out["resources_reason"] = (result.get("measured_resources") or {}).get("invalid_reason")
    out["clean_resources_valid"] = (result.get("clean_measured_resources") or {}).get("valid")
    out["actionable"] = result.get("actionable_fault_resources")
    return out
apps = Path("/home/ubuntu/multiagent_IBM/AIOpsLab/aiopslab-applications")
app_root = apps / ("hotelReservation" if "hotel" in SID and (apps / "hotelReservation").is_dir() else "socialNetwork")
cfg = SparseLiveVerifierConfig(
    source_namespace="test-hotel-reservation" if "hotel" in SID else "test-social-network",
    application_source_root=str(app_root.resolve()),
    state_abstraction_root=str(Path(ROOT, "state_abstraction_full").resolve()),
    require_reward_calibration=False, reproduction_threshold=0.0,
    workload_rate=10, workload_duration_seconds=150, artifact_root=str(OUT / "twin_artifacts"))
rec = next(iter_scenarios("/mnt/aiops-training/datasets/full-622-49-v1/processed_states", allowed_ids={SID}))
labels = labels_from_full_state(rec.full_state)
emit("start", scenario=SID, labels=[l.to_dict() for l in labels], workload_seconds=150)
v = SparseLiveTwinVerifier(cfg)
final = {"scenario": SID}
try:
    t0 = time.time(); v.prepare_scenario({}, rec.compressed_state)
    rs = v._incident_spec.resource_summary
    emit("scope", secs=round(time.time()-t0,1), scrape_interval=v.scrape_interval_seconds, keep=len(v._incident_spec.services_to_keep),
         total=rs.get("total_application_services"), targets=rs.get("incident_request_path_targets"),
         trace_targets=rs.get("incident_trace_observable_targets"), unattributed=rs.get("unattributed_symptom_names"),
         undeployable=rs.get("undeployable_inventory_names"), environment_sha256=v.environment_sha256)
    # ---- positive control + independent action ----
    v.begin_trajectory("smoke-positive"); t0 = time.time()
    v.prepare_incident_twin(rec.compressed_state)
    clean = v._clean_capture
    emit("clean_phase", secs=round(time.time()-t0,1), namespace=v.session.namespace, workload=clean["workload"].to_dict() if hasattr(clean["workload"], "to_dict") else str(clean["workload"]),
         channels=clean["channels"], coverage=clean["coverage"], resources=clean["collection"]["resources"], errors=clean["collection"].get("errors"))
    t0 = time.time(); pos = v.validate_rca_prediction({}, rec.compressed_state, labels)
    final["positive"] = summary(pos); emit("positive", secs=round(time.time()-t0,1), **final["positive"])
    if pos.get("rca_twin_verified"):
        try:
            t0 = time.time(); gate = v.prepare_action_attempt(labels)
            cmds = []
            for res in (pos.get("actionable_fault_resources") or []):
                if isinstance(res, dict) and res.get("kind") and res.get("name"):
                    cmds.append(f"kubectl delete {res['kind']} {res['name']} -n {v.session.namespace}")
            if not cmds:
                svc = labels[0].service; ns = v.session.namespace; mech = labels[0].fault_mechanism
                if mech == "scale_replicas_zero": cmds = [f"kubectl scale deployment/{svc} --replicas=1 -n {ns}"]
                elif mech == "assign_to_non_existent_node": cmds = [f"kubectl patch deployment {svc} -n {ns} --type=json -p='[{{\"op\":\"remove\",\"path\":\"/spec/template/spec/nodeSelector\"}}]'"]
            emit("action_commands", commands=cmds)
            if cmds:
                act = v.apply_commands_and_score({}, labels, {"service": labels[0].service}, cmds, compressed_state=rec.compressed_state)
                final["action"] = {k: act.get(k) for k in ("resolved", "sla_condition_satisfied", "sla_restored", "sla_restoration_applicable",
                                   "symptom_reduction", "telemetry_incomplete", "reason", "repair_validation") if k in act}
                final["action"]["resources_valid"] = (act.get("measured_resources") or {}).get("valid")
                final["action"]["verified_repair_plan"] = bool(act.get("verified_repair_plan"))
                final["action"]["repair_export_error"] = act.get("repair_export_error")
                final["action"]["execution"] = act.get("execution")
                emit("action", secs=round(time.time()-t0,1), **{k: v_ for k, v_ in final["action"].items() if k != "execution"})
        except Exception as exc:
            final["action_error"] = f"{type(exc).__name__}: {exc}"; emit("action_error", error=final["action_error"], tb=traceback.format_exc()[-1500:])
    v.end_trajectory()
    # ---- wrong-service control ----
    keep = [s for s in v._incident_spec.services_to_keep if s != labels[0].service and s in set(rs.get("incident_request_path_targets") or [])]
    other = keep[0] if keep else next(s for s in v._incident_spec.services_to_keep if s != labels[0].service)
    wrong = [replace(labels[0], service=other, metadata={})]
    v.begin_trajectory("smoke-wrong-service"); t0 = time.time()
    neg = v.validate_rca_prediction({}, rec.compressed_state, wrong)
    final["wrong_service"] = {"service": other, **summary(neg)}; emit("wrong_service", secs=round(time.time()-t0,1), **final["wrong_service"])
    v.end_trajectory()
except Exception as exc:
    final["error"] = f"{type(exc).__name__}: {exc}"; emit("error", error=final["error"], tb=traceback.format_exc()[-2500:])
    try: v.end_trajectory()
    except Exception: pass
(OUT / "summary.json").write_text(json.dumps(final, indent=2, default=str))
emit("done", **{k: (v_ if not isinstance(v_, dict) else {kk: v_[kk] for kk in list(v_)[:6]}) for k, v_ in final.items()})
