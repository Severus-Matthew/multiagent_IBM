"""Record incidents in the source application under the Twin's measurement contract.

The fault is injected with the AIOpsLab problem definition (plus the reviewed
target-honouring injector fixes), but every observation is taken exactly the way
the Twin takes it: the same wrk2 payload/endpoint selection, rate and duration,
an explicit ``ObservationWindow`` per phase, and ``collect_targeted_telemetry``
(phase-bounded Jaeger, Prometheus at phase end with scrape coverage, timestamped
log windows, pod inventory). Phases per scenario:

    clean      healthy source application, before injection
    incident   after injection and a manifestation settle; becomes the record
    recovered  after the problem's own recovery (optional evidence)

The incident phase is abstracted into a dataset record whose private label comes
from ``spec.json``; ``ground_truth.json``, ``fault_timing.json`` and
``injection_evidence.json`` follow the generator's formats. The recorder refuses
to run while ``aiops-twin-*`` namespaces exist (Twins clone the source), refuses
an unclean source, refuses to write into an existing scenario directory, and
stops the whole run if the source cannot be returned to a clean state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from digital_twin_runtime.targeted_telemetry import (  # noqa: E402
    MEASUREMENT_CONTRACT, ObservationWindow, TelemetryCollectionError, capture_pod_inventory,
    collect_targeted_telemetry, discover_prometheus_scrape_interval, hold_phase_window,
    require_phase_window_covers_scrapes,
)
from digital_twin_runtime.targeted_workload import run_targeted_wrk  # noqa: E402

CAPTURE_FORMAT = "compatible_incident_capture_v1"
PRIVATE_FILES = ("spec.json", "problem_desc.txt", "ground_truth.json", "fault_timing.json", "injection_evidence.json")
APP_ROOTS = {"hotel": "hotelReservation", "social": "socialNetwork"}
SOURCE_NAMESPACES = {"hotel": "test-hotel-reservation", "social": "test-social-network"}


@dataclass
class RecorderConfig:
    aiopslab_root: Path
    output_dir: Path
    state_abstraction_root: Path
    workload_rate: int = 10
    workload_duration_seconds: int = 150
    manifestation_settle_seconds: float = 45.0
    telemetry_settle_seconds: float = 5.0
    recovery_timeout_seconds: float = 300.0
    capture_recovered: bool = True
    abstraction_python: str = sys.executable
    app_roots: dict[str, str] = field(default_factory=dict)
    source_namespaces: dict[str, str] = field(default_factory=dict)


class SourceSession:
    """The minimal session view the Twin collector needs, for the source namespace."""

    def __init__(self, namespace: str, objects: list[dict[str, Any]]):
        self.namespace = namespace
        # The source application is already deployed; the Twin helpers assert
        # these lifecycle flags before creating workload Jobs in the namespace.
        self.created = True
        self.applied = True
        controllers = [o for o in objects if o.get("kind") in {"Deployment", "StatefulSet"}]
        self.bundle = SimpleNamespace(
            objects=objects,
            object_refs=[{"kind": o["kind"], "name": o["metadata"]["name"]} for o in controllers],
        )
        self.controller_names = sorted(o["metadata"]["name"] for o in controllers)


def _kubectl_json(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(["kubectl", *args, "-o", "json"], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout or "{}")


def load_source_session(namespace: str) -> SourceSession:
    items = _kubectl_json(["get", "deployments,statefulsets,services", "-n", namespace]).get("items", [])
    for item in items:
        item.setdefault("kind", "")
    return SourceSession(namespace, items)


def twin_namespaces_present() -> list[str]:
    proc = subprocess.run(["kubectl", "get", "ns", "--no-headers", "-o", "custom-columns=:metadata.name"],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60)
    return sorted(n for n in proc.stdout.split() if n.startswith("aiops-twin-"))


def spec_targets(spec: dict[str, Any]) -> list[str]:
    if spec.get("is_multifault") or spec.get("mode") == "multifault":
        raise ValueError("multifault specs are not supported by the recorder yet")
    target = spec.get("faulty_service") or spec.get("fault_service") or spec.get("service")
    if not target:
        raise ValueError(f"spec {spec.get('problem_id')} names no faulty service; application-level faults are not supported")
    return [str(target)]


def injection_evidence(problem_id: str, journal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Same admission rule and format as dataset_generation/regenerate.py."""
    applied = [m for m in journal_rows if m.get("applied")]
    verified = bool(applied) and all(m.get("applied") and m.get("manifested") for m in journal_rows)
    return {
        "format": "source_injection_evidence_v1",
        "problem_id": problem_id,
        "verified": verified,
        "faults": [{**m, "mutated": bool(m.get("applied")), "manifested": bool(m.get("manifested"))} for m in journal_rows],
    }


def select_workloads(verifier: Any, targets: list[str]) -> list[dict[str, Any]]:
    """The Twin's payload/endpoint choice for each target, deduplicated like the Twin."""
    distinct: dict[tuple[str, str], dict[str, Any]] = {}
    for service in targets:
        script, endpoint = verifier._workload(service)
        key = (str(script), str(endpoint))
        distinct.setdefault(key, {"service": service, "payload_script": str(script), "endpoint": str(endpoint),
                                  "payload_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest()})
    return list(distinct.values())


def run_phase(session: SourceSession, profile: Any, workloads: list[dict[str, Any]], phase: str,
              out_dir: Path, *, scrape_interval: float, cfg: RecorderConfig,
              wrk: Callable[..., Any] = run_targeted_wrk,
              collect: Callable[..., Any] = collect_targeted_telemetry,
              inventory: Callable[[Any], dict[str, Any]] = capture_pod_inventory,
              sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """One measured phase: inventory, workloads, window, settle, collect. Mirrors the verifier."""
    initial = inventory(session)
    started = clock()
    rows = []
    for load in workloads:
        rows.append(wrk(session, payload_script=Path(load["payload_script"]), endpoint=load["endpoint"],
                        rate=cfg.workload_rate, duration_seconds=cfg.workload_duration_seconds,
                        required_service=load["service"], frontend_service=profile.frontend_service,
                        frontend_container=profile.frontend_container, frontend_port=profile.frontend_port))
    # A workload that dies under the fault must not shorten the phase (same
    # rule as the Twin's _capture_phase).
    window = ObservationWindow(started, hold_phase_window(started, cfg.workload_duration_seconds, clock=clock, sleep=sleep), phase)
    sleep(max(0.0, cfg.telemetry_settle_seconds))
    primary = rows[0]
    collection = collect(session, out_dir, window=window, workload=primary, initial_pod_inventory=initial,
                         scrape_interval_seconds=scrape_interval)
    contract = [{"service": w["service"], "rate": cfg.workload_rate, "duration_seconds": cfg.workload_duration_seconds,
                 "payload_sha256": w["payload_sha256"], "endpoint": w["endpoint"].replace(session.namespace, "application-namespace")}
                for w in workloads]
    healthy = all(bool(getattr(r, "completed", False)) and not getattr(r, "failed", True)
                  and not getattr(r, "application_failures", 1) and not getattr(r, "non_success_responses", 1)
                  and (getattr(r, "total_requests", 0) or 0) > 0 for r in rows)
    phase_record = {
        "phase": phase, "measurement_contract": MEASUREMENT_CONTRACT, "window": window.to_dict(),
        "workload_contract": contract,
        "workload_contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
        "workload_healthy": healthy,
        "workloads": [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in rows],
        "collection": collection.to_dict() if hasattr(collection, "to_dict") else dict(collection),
    }
    (out_dir / "phase.json").write_text(json.dumps(phase_record, indent=2, default=str))
    return phase_record


def abstract_phase(phase_dir: Path, output_dir: Path, cfg: RecorderConfig) -> dict[str, Any]:
    proc = subprocess.run([cfg.abstraction_python, "run_pipeline.py", "--run_dir", str(phase_dir),
                           "--output_dir", str(output_dir), "--skip_simulator"],
                          cwd=str(cfg.state_abstraction_root), text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=900)
    ok = proc.returncode == 0 and (output_dir / "state_abstraction_compressed.json").is_file()
    return {"ok": ok, "returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:] if not ok else ""}


def record_scenario(spec: dict[str, Any], *, cfg: RecorderConfig, generator: Any, problem_factory: Callable[[], Any],
                    session: SourceSession, verifier: Any, journal: list[dict[str, Any]],
                    scrape_interval: float, is_clean: Callable[[str], tuple[bool, dict[str, Any]]],
                    wait_clean: Callable[[str, float], tuple[bool, dict[str, Any]]],
                    phase_runner: Callable[..., dict[str, Any]] = run_phase,
                    abstractor: Callable[..., dict[str, Any]] = abstract_phase,
                    sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    problem_id = str(spec["problem_id"])
    scenario_dir = cfg.output_dir / "raw" / problem_id
    if scenario_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing capture: {scenario_dir}")
    targets = spec_targets(spec)
    row: dict[str, Any] = {"problem_id": problem_id, "scenario_spec_sha256": spec.get("scenario_spec_sha256"),
                           "targets": targets, "namespace": session.namespace, "phases": {}}
    ok, report = is_clean(session.namespace)
    if not ok:
        raise RuntimeError(f"source namespace is not clean before recording: {report}")
    scenario_dir.mkdir(parents=True)
    workloads = select_workloads(verifier, targets)
    row["workloads"] = workloads
    problem = problem_factory()
    generator.write_json(scenario_dir / "spec.json", spec)
    try:
        desc = problem.get_task_description()
    except Exception:  # noqa: BLE001 - description is provenance only
        desc = f"{spec.get('fault_family')} on {targets} in {session.namespace} ({spec.get('task')})"
    (scenario_dir / "problem_desc.txt").write_text(str(desc))
    generator.save_ground_truth(problem, spec, scenario_dir)

    row["phases"]["clean"] = phase_runner(session, verifier.runtime_profile, workloads, "clean", scenario_dir / "clean",
                                          scrape_interval=scrape_interval, cfg=cfg)
    journal_start = len(journal)
    injected_at = datetime.now(timezone.utc).isoformat()
    problem.inject_fault()
    evidence: dict[str, Any] = {"verified": False}
    try:
        sleep(max(0.0, cfg.manifestation_settle_seconds))
        mutations = list(journal[journal_start:])
        evidence = injection_evidence(problem_id, mutations)
        row["injection_verified"] = evidence["verified"]
        (scenario_dir / "injection_evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n")
        row["phases"]["incident"] = phase_runner(session, verifier.runtime_profile, workloads, "incident",
                                                 scenario_dir / "incident", scrape_interval=scrape_interval, cfg=cfg)
        collected_at = datetime.now(timezone.utc).isoformat()
        generator.save_fault_timing(scenario_dir, injected_at, collected_at)
        for name in PRIVATE_FILES:
            if (scenario_dir / name).exists():
                shutil.copy2(scenario_dir / name, scenario_dir / "incident" / name)
        for helper in ("build_topology_and_graph", "validate_unready_pods"):
            fn = getattr(generator, helper, None)
            if callable(fn):
                try:
                    fn(session.namespace, scenario_dir / "incident") if helper == "build_topology_and_graph" \
                        else fn(session.namespace, spec, scenario_dir / "incident")
                    row[helper] = "ok"
                except Exception as exc:  # noqa: BLE001 - optional corpus-parity artifacts
                    row[helper] = f"{type(exc).__name__}: {exc}"
    finally:
        # Whatever happened after injection, the source application is
        # returned to a clean state (the next scenario and every Twin clone it).
        try:
            problem.recover_fault()
        finally:
            recovered, report = wait_clean(session.namespace, cfg.recovery_timeout_seconds)
            row["recovered_clean"] = recovered
            row["recovery_report"] = report
    if not recovered:
        raise RuntimeError(f"source namespace did not return to a clean state after recovery: {report}")
    if cfg.capture_recovered:
        row["phases"]["recovered"] = phase_runner(session, verifier.runtime_profile, workloads, "recovered",
                                                  scenario_dir / "recovered", scrape_interval=scrape_interval, cfg=cfg)
    capture = {"format": CAPTURE_FORMAT, "measurement_contract": MEASUREMENT_CONTRACT, "problem_id": problem_id,
               "namespace_role": "source_application", "scrape_interval_seconds": scrape_interval,
               "workload_rate": cfg.workload_rate, "workload_duration_seconds": cfg.workload_duration_seconds,
               "manifestation_settle_seconds": cfg.manifestation_settle_seconds,
               "injected_at": injected_at, "collected_at": collected_at,
               "phases": {name: {k: v for k, v in rec.items() if k != "workloads"} for name, rec in row["phases"].items()},
               "injection_verified": evidence["verified"]}
    (scenario_dir / "recorder_capture.json").write_text(json.dumps(capture, indent=2, default=str))
    processed = cfg.output_dir / "processed_states" / problem_id
    row["abstraction"] = abstractor(scenario_dir / "incident", processed, cfg)
    row["phase_abstractions"] = {}
    for name in ("clean", "recovered"):
        if name in row["phases"]:
            row["phase_abstractions"][name] = abstractor(scenario_dir / name, cfg.output_dir / "processed_phases" / problem_id / name, cfg)
    # The collector raises on any incomplete phase, so reaching here means every
    # phase was fully observed. Acceptance additionally needs a verified
    # injection, a healthy clean-phase workload (the source served traffic
    # before the fault), a recovered source and a usable abstraction; the
    # incident-phase workload may legitimately fail (that is the symptom).
    row["accepted"] = bool(evidence["verified"] and row["phases"]["clean"]["workload_healthy"]
                           and row["abstraction"]["ok"] and recovered)
    return row


def _legacy_problem_id(spec: dict[str, Any]) -> str:
    problem_id = str(spec.get("problem_id"))
    if spec.get("is_multifault") and "--" in problem_id:
        return problem_id.rsplit("--", 1)[0]
    return problem_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario_ids", required=True, help="One problem id per line (legacy multifault ids accepted)")
    ap.add_argument("--output_dir", required=True, help="New directory; raw/, processed_states/, processed_phases/ are created inside")
    ap.add_argument("--aiopslab_root", default=str(REPO_ROOT / "AIOpsLab"))
    ap.add_argument("--state_abstraction_root", default=str(REPO_ROOT / "state_abstraction_full"))
    ap.add_argument("--workload_rate", type=int, default=10)
    ap.add_argument("--workload_duration_seconds", type=int, default=150,
                    help="Must match the training/calibration launch and cover two Prometheus scrapes plus 5s")
    ap.add_argument("--manifestation_settle_seconds", type=float, default=45.0)
    ap.add_argument("--recovery_timeout_seconds", type=float, default=300.0)
    ap.add_argument("--no_recovered_phase", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")
    twins = twin_namespaces_present()
    if twins:
        raise SystemExit(f"refusing to record while Twin namespaces exist (they clone the source): {twins}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)

    from dataset_generation.injector_fixes import INJECTION_JOURNAL, apply_injector_fixes
    from dataset_generation.regenerate import _import_generator
    from dataset_generation.warm_cluster import namespace_is_clean, wait_until_clean
    from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig

    aiopslab_root = Path(args.aiopslab_root).expanduser().resolve()
    generator = _import_generator(aiopslab_root, output / "generator_output")
    patched = apply_injector_fixes()
    wanted = {line.strip() for line in Path(args.scenario_ids).read_text().splitlines() if line.strip() and not line.startswith("#")}
    specs = [s for s in generator.generate_specs()
             if str(s.get("problem_id")) in wanted or _legacy_problem_id(s) in wanted]
    specs = generator.unique_scenarios([s if s.get("scenario_spec_sha256") else generator.attach_scenario_identity(s) for s in specs])
    missing = sorted(wanted - {str(s.get("problem_id")) for s in specs} - {_legacy_problem_id(s) for s in specs})
    if args.limit:
        specs = specs[: args.limit]
    scrape_interval = discover_prometheus_scrape_interval()
    require_phase_window_covers_scrapes(args.workload_duration_seconds, scrape_interval)
    cfg = RecorderConfig(aiopslab_root=aiopslab_root, output_dir=output,
                         state_abstraction_root=Path(args.state_abstraction_root).expanduser().resolve(),
                         workload_rate=args.workload_rate, workload_duration_seconds=args.workload_duration_seconds,
                         manifestation_settle_seconds=args.manifestation_settle_seconds,
                         recovery_timeout_seconds=args.recovery_timeout_seconds, capture_recovered=not args.no_recovered_phase,
                         app_roots=APP_ROOTS, source_namespaces=SOURCE_NAMESPACES)
    manifest = {"format": CAPTURE_FORMAT, "measurement_contract": MEASUREMENT_CONTRACT, "created_unix": time.time(),
                "arguments": vars(args), "scrape_interval_seconds": scrape_interval, "patched_injectors": patched,
                "requested": sorted(wanted), "missing_from_generator": missing,
                "generator_commit": subprocess.run(["git", "-C", str(aiopslab_root), "rev-parse", "HEAD"], text=True,
                                                   stdout=subprocess.PIPE, check=False).stdout.strip(),
                "repo_commit": subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True,
                                              stdout=subprocess.PIPE, check=False).stdout.strip()}
    (output / "recorder_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log = (output / "log.jsonl").open("a")
    (output / "accepted").mkdir(exist_ok=True)
    print(json.dumps({"event": "start", "specs": len(specs), "missing": missing, "scrape_interval": scrape_interval}), flush=True)
    sessions: dict[str, tuple[SourceSession, Any]] = {}
    for spec in specs:
        app = str(spec.get("app") or spec.get("app_name") or "")
        row: dict[str, Any] = {"problem_id": spec.get("problem_id"), "app": app, "started_unix": time.time()}
        try:
            namespace = cfg.source_namespaces[app]
            if app not in sessions:
                session = load_source_session(namespace)
                verifier = SparseLiveTwinVerifier(SparseLiveVerifierConfig(
                    source_namespace=namespace,
                    application_source_root=str(aiopslab_root / "aiopslab-applications" / cfg.app_roots[app]),
                    state_abstraction_root=str(cfg.state_abstraction_root), require_reward_calibration=False,
                    workload_rate=cfg.workload_rate, workload_duration_seconds=cfg.workload_duration_seconds))
                verifier.runtime_profile = verifier._profile({"services": session.controller_names})
                verifier.selected_paths = []
                sessions[app] = (session, verifier)
            session, verifier = sessions[app]
            row.update(record_scenario(spec, cfg=cfg, generator=generator, problem_factory=generator.make_problem_factory(spec),
                                       session=session, verifier=verifier, journal=INJECTION_JOURNAL,
                                       scrape_interval=scrape_interval, is_clean=namespace_is_clean, wait_clean=wait_until_clean))
            if row.get("accepted"):
                (output / "accepted" / f"{spec['problem_id']}.json").write_text(json.dumps(row, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001 - journal and continue unless the source is unhealthy
            row.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:], "accepted": False})
            fatal = "did not return to a clean state" in str(exc) or "not clean before recording" in str(exc)
            row["fatal"] = fatal
        row["elapsed_seconds"] = round(time.time() - row["started_unix"], 1)
        log.write(json.dumps(row, default=str) + "\n"); log.flush()
        print(json.dumps({k: row.get(k) for k in ("problem_id", "accepted", "injection_verified", "recovered_clean", "error", "elapsed_seconds")}), flush=True)
        if row.get("fatal"):
            print(json.dumps({"event": "abort", "reason": row["error"]}), flush=True)
            return 3
    print(json.dumps({"event": "done", "accepted": len(list((output / "accepted").glob("*.json"))), "specs": len(specs)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
