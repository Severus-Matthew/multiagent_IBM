"""Recovery-machinery evidence: clean -> injected -> repair -> recovered on the corrected verifier.

The evaluator applies the exact repair for the injected fault directly with
apply_commands_and_score. This deliberately bypasses prepare_action_attempt's
evidence gate (which rejected the positive control on this incident) and is
therefore evidence that the recovery lifecycle works, NOT a qualification.
"""
import json, sys, time, traceback
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
apps = Path("/home/ubuntu/multiagent_IBM/AIOpsLab/aiopslab-applications")
app_root = apps / ("hotelReservation" if "hotel" in SID else "socialNetwork")
cfg = SparseLiveVerifierConfig(
    source_namespace="test-hotel-reservation" if "hotel" in SID else "test-social-network",
    application_source_root=str(app_root.resolve()),
    state_abstraction_root=str(Path(ROOT, "state_abstraction_full").resolve()),
    require_reward_calibration=False, reproduction_threshold=0.0,
    workload_rate=10, workload_duration_seconds=150, artifact_root=str(OUT / "twin_artifacts"))
rec = next(iter_scenarios("/mnt/aiops-training/datasets/full-622-49-v1/processed_states", allowed_ids={SID}))
labels = labels_from_full_state(rec.full_state)
v = SparseLiveTwinVerifier(cfg); final = {"scenario": SID, "note": "evidence gate bypassed for machinery evidence"}
try:
    v.prepare_scenario({}, rec.compressed_state)
    v.begin_trajectory("smoke-action"); t0 = time.time()
    v.prepare_incident_twin(rec.compressed_state)
    emit("clean_phase", secs=round(time.time()-t0,1), namespace=v.session.namespace, resources_valid=v._clean_capture["collection"]["resources"].get("valid"))
    t0 = time.time(); pos = v.validate_rca_prediction({}, rec.compressed_state, labels)
    final["positive"] = {k: pos.get(k) for k in ("reproduction_score", "clean_reproduction_score", "counterfactual_evidence_gate", "rca_twin_verified", "predicted_fault_injection_checked")}
    emit("positive", secs=round(time.time()-t0,1), **final["positive"], actionable=pos.get("actionable_fault_resources"))
    cmds = [f"kubectl delete {r['kind']} {r['name']} -n {v.session.namespace}" for r in (pos.get("actionable_fault_resources") or []) if isinstance(r, dict) and r.get("kind") and r.get("name")]
    emit("action_commands", commands=cmds)
    before_sla = (v.before_state.get("sla") or {}).get("global_sla")
    t0 = time.time(); act = v.apply_commands_and_score({}, labels, {"service": labels[0].service}, cmds, compressed_state=rec.compressed_state)
    final["action"] = {k: act.get(k) for k in ("resolved", "twin_resolved", "sla_condition_satisfied", "sla_restored", "sla_restoration_applicable",
                       "sla_transition_restored", "symptom_reduction", "global_symptom_reduction", "telemetry_incomplete", "reason", "repair_validation", "repair_export_error") if k in act}
    final["action"]["execution"] = {k: (act.get("execution") or {}).get(k) for k in ("executed", "reasons", "commands")} if isinstance(act.get("execution"), dict) else act.get("execution")
    final["action"]["recovery_ready"] = (act.get("recovery") or {}).get("ready") if isinstance(act.get("recovery"), dict) else None
    final["action"]["resources_valid"] = (act.get("measured_resources") or {}).get("valid")
    final["action"]["before_sla"] = before_sla; final["action"]["after_sla"] = (act.get("after_sla") or {}).get("global_sla") if isinstance(act.get("after_sla"), dict) else None
    final["action"]["verified_repair_plan"] = bool(act.get("verified_repair_plan"))
    emit("action", secs=round(time.time()-t0,1), **{k: v_ for k, v_ in final["action"].items()})
    v.end_trajectory()
except Exception as exc:
    final["error"] = f"{type(exc).__name__}: {exc}"; emit("error", error=final["error"], tb=traceback.format_exc()[-2500:])
    try: v.end_trajectory()
    except Exception: pass
(OUT / "summary.json").write_text(json.dumps(final, indent=2, default=str)); emit("done")
