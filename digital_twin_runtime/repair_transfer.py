"""Export independently verified repairs, bind real targets, and verify application.

Training only exports portable plans. This separate CLI performs real mutations
only with --execute and an explicit kube context/namespace. No LLM selects the
cluster or binds Twin-owned Chaos resources to real objects.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from training_pipeline.command_safety import check_command_safety
from training_pipeline.kubectl_command_shape import positional_args, resource_target
from .targeted_telemetry import MEASUREMENT_CONTRACT
from .telemetry_comparator import compare_symptoms_scoped, score_resolution

FORMAT = "independently_verified_repair_v1"
_KIND = {"deploy": "deployment", "deployments": "deployment", "deployment": "deployment",
         "sts": "statefulset", "statefulsets": "statefulset", "statefulset": "statefulset",
         "svc": "service", "services": "service", "service": "service",
         "podchaos": "podchaos", "networkchaos": "networkchaos"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _run(args: list[str], *, payload=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kubectl", *args], input=json.dumps(payload) if payload is not None else None,
                          text=True, capture_output=True, timeout=180, check=False)


def _json(args: list[str], *, payload=None) -> dict[str, Any]:
    proc = _run(args, payload=payload)
    if proc.returncode:
        raise RuntimeError("Kubernetes operation failed: " + proc.stderr[-1500:])
    return json.loads(proc.stdout)


def _without_namespace(parts: list[str], expected: str) -> list[str]:
    out = []
    namespaces = []
    i = 0
    while i < len(parts):
        value = parts[i]
        if value in {"-n", "--namespace"}:
            if i + 1 >= len(parts):
                raise ValueError("namespace flag has no value")
            namespaces.append(parts[i + 1]); i += 2; continue
        if value.startswith("--namespace=") or value.startswith("-n="):
            namespaces.append(value.split("=", 1)[1]); i += 1; continue
        out.append(value); i += 1
    if namespaces != [expected]:
        raise ValueError("repair needs one exact verifier-owned namespace")
    return out


def export_verified_repair(verifier, commands: list[str], result: dict[str, Any]) -> dict[str, Any]:
    gate = verifier.last_rca_result or {}
    if not (gate.get("rca_twin_verified") and gate.get("live_reward_calibrated")
            and result.get("resolved") and result.get("sla_condition_satisfied")
            and result.get("repair_validation") == "independent_frozen_fault_state_v1"
            and commands == result.get("verified_commands")
            and result.get("measurement_contract") == MEASUREMENT_CONTRACT):
        raise ValueError("repair has no calibrated, independent live verification evidence")
    if not check_command_safety(commands).get("safe"):
        raise ValueError("unsafe repair cannot be exported")
    namespace = verifier.session.namespace
    operations = []
    for command in commands:
        args = _without_namespace(shlex.split(command), namespace)
        if not args or args[0] != "kubectl":
            raise ValueError("only kubectl repair operations are portable")
        positional = positional_args(args, 1)
        verb = positional[0] if positional else ""
        if verb not in {"patch", "scale", "delete"}:
            continue  # The machine-owned postcheck replaces get/rollout probes.
        target = resource_target(positional)
        if not target or target[0].lower() not in _KIND or not target[1]:
            raise ValueError("repair operation lacks an exact supported target")
        kind, name = _KIND[target[0].lower()], target[1]
        if verb == "delete" and kind not in {"podchaos", "networkchaos"}:
            raise ValueError("only explicitly bound Chaos deletions are portable")
        operations.append({"verb": verb, "kind": kind, "name": name, "argv": args[1:]})
    if not operations:
        raise ValueError("verified repair contains no portable mutations")
    plan = {"format": FORMAT, "measurement_contract": MEASUREMENT_CONTRACT,
            "environment_sha256": verifier.environment_sha256,
            "operations": operations,
            "services": list(verifier._incident_state["services"]),
            "selected_services": list(verifier.selected_services),
            "affected_services": verifier._incident_spec.resource_summary["incident_affected_services"],
            "selected_paths": verifier.selected_paths,
            "expected_incident_state": verifier._incident_state,
            "reference_configuration": verifier._incident_reference,
            "workload_rate": verifier.config.workload_rate,
            "workload_duration_seconds": verifier.config.workload_duration_seconds,
            "decision_threshold": gate["decision_threshold"],
            "evidence": {"rca_verified": True, "calibrated": True, "recovery_verified": True,
                         "independent_fault_state": True,
                         "calibration_sha256": gate.get("reward_calibration", {}).get("manifest_sha256"),
                         "rca_result_sha256": digest(gate), "action_result_sha256": digest(result)}}
    return {**plan, "plan_sha256": digest(plan)}


def validate_plan(plan: dict[str, Any]) -> None:
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("format") != FORMAT or plan.get("plan_sha256") != digest(body):
        raise ValueError("repair plan integrity/format mismatch")
    if plan.get("measurement_contract") != MEASUREMENT_CONTRACT:
        raise ValueError("repair plan uses an obsolete verification contract")
    evidence = plan.get("evidence", {})
    if not all(evidence.get(k) is True for k in ("rca_verified", "calibrated", "recovery_verified", "independent_fault_state")):
        raise ValueError("repair plan lacks independent qualification")
    if not evidence.get("calibration_sha256"):
        raise ValueError("repair plan is not bound to calibration evidence")
    if not plan.get("operations") or not 0 <= float(plan["decision_threshold"]) <= 1:
        raise ValueError("repair plan lacks bounded operations or a valid threshold")
    for op in plan["operations"]:
        argv = op.get("argv", [])
        if not argv or any(x in {"-n", "--namespace"} or x.startswith(("--namespace=", "-n=")) for x in argv):
            raise ValueError("portable operations cannot select a namespace")
        positional = positional_args(["kubectl", *argv], 1)
        target = resource_target(positional)
        if (not target or op.get("verb") not in {"patch", "scale", "delete"}
                or positional[0] != op["verb"] or _KIND.get(target[0]) != op.get("kind")
                or target[1] != op.get("name")):
            raise ValueError("operation metadata does not match its executable target")
        command = shlex.join(["kubectl", *argv, "-n", "aiops-twin-plan-check"])
        if not check_command_safety([command]).get("safe"):
            raise ValueError("portable operation violates the command boundary")
        if op["verb"] == "delete" and op["kind"] not in {"podchaos", "networkchaos"}:
            raise ValueError("only explicitly bound Chaos deletion is portable")


@contextlib.contextmanager
def bound_context(context: str):
    # Isolate the CLI's subprocesses without changing the user's kubectl context.
    config = _json(["config", "view", "--raw", "--flatten", "--minify", "--context", context, "-o", "json"])
    config["current-context"] = context
    with tempfile.TemporaryDirectory(prefix="aiops-repair-context-") as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(config)); path.chmod(0o600)
        previous = os.environ.get("KUBECONFIG")
        os.environ["KUBECONFIG"] = str(path)
        try:
            yield digest([{k: v for k, v in row.get("cluster", {}).items()
                           if k in {"server", "certificate-authority-data"}} for row in config.get("clusters", [])])
        finally:
            if previous is None:
                os.environ.pop("KUBECONFIG", None)
            else:
                os.environ["KUBECONFIG"] = previous


def bind_plan(plan: dict[str, Any], namespace: str, cluster_sha: str, bindings: dict[str, Any]) -> dict[str, Any]:
    validate_plan(plan)
    if (not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", namespace)
            or namespace in {"kube-system", "kube-public", "kube-node-lease"} or namespace.startswith("aiops-twin-")):
        raise ValueError("an explicit application namespace is required")
    namespace_uid = _json(["get", "namespace", namespace, "-o", "json"])["metadata"]["uid"]
    snapshots = {}
    operations = []
    for operation in plan["operations"]:
        op = deepcopy(operation)
        key = op["kind"] + "/" + op["name"]
        binding = bindings.get(key)
        if op["verb"] == "delete":
            if not binding or binding.get("allow_delete") is not True or not binding.get("uid"):
                raise ValueError("Chaos repair requires an explicit real object name, UID, and allow_delete binding")
            op["name"] = str(binding["name"])
        elif binding:
            raise ValueError("controller/Service repair preserves logical names; only Chaos resources are rebound")
        obj = _json(["get", op["kind"], op["name"], "-n", namespace, "-o", "json"])
        if binding and obj["metadata"]["uid"] != binding["uid"]:
            raise ValueError("real Chaos resource UID differs from its binding")
        if op["verb"] != "delete" and op["name"] not in plan["selected_services"]:
            raise ValueError("repair target is outside the qualified incident scope")
        target = op["kind"] + "/" + op["name"]
        snapshots[target] = obj
        operations.append(op)
    return {"namespace": namespace, "namespace_uid": namespace_uid, "cluster_sha256": cluster_sha,
            "plan_sha256": plan["plan_sha256"], "operations": operations, "snapshots": snapshots}


def _delete_exact(kind, name, namespace, obj):
    # The ordinary kubectl delete CLI has no UID/resourceVersion precondition.
    # Use the Kubernetes API's atomic DeleteOptions for exact Chaos ownership.
    from kubernetes import client, config
    config.load_kube_config(config_file=os.environ["KUBECONFIG"])
    options = client.V1DeleteOptions(preconditions=client.V1Preconditions(
        uid=obj["metadata"]["uid"], resource_version=obj["metadata"]["resourceVersion"]))
    client.CustomObjectsApi().delete_namespaced_custom_object(
        "chaos-mesh.org", "v1alpha1", namespace, kind, name, body=options)
    proc = _run(["wait", "--for=delete", f"{kind}/{name}", "-n", namespace, "--timeout=120s"])
    if proc.returncode:
        raise RuntimeError("Chaos deletion/finalizer did not finish")


def apply_bound_plan(plan: dict[str, Any], bound: dict[str, Any], *, verify) -> dict[str, Any]:
    """Machine verification is mandatory; return failure and rollback on any error."""
    validate_plan(plan)
    namespace = bound["namespace"]
    if bound["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("bound repair belongs to a different plan")
    if _json(["get", "namespace", namespace, "-o", "json"])["metadata"]["uid"] != bound["namespace_uid"]:
        raise ValueError("target namespace was recreated")
    expected = deepcopy(bound["snapshots"])
    for target, original in expected.items():
        current = _json(["get", target, "-n", namespace, "-o", "json"])
        if current["metadata"]["uid"] != original["metadata"]["uid"] or digest(current.get("spec")) != digest(original.get("spec")):
            raise ValueError("real repair precondition changed: " + target)
    before = verify("real_before")
    comparison = compare_symptoms_scoped(plan["expected_incident_state"], before["state"],
                                         plan["selected_services"], plan["selected_services"])
    if not comparison.get("positive_incident_evidence") or comparison["reproduction_score"] < plan["decision_threshold"]:
        raise ValueError("real incident no longer matches the independently verified repair")
    changes = []
    try:
        for operation in bound["operations"]:
            kind, name, verb = operation["kind"], operation["name"], operation["verb"]
            target = kind + "/" + name
            current = _json(["get", target, "-n", namespace, "-o", "json"])
            if (current["metadata"]["uid"] != expected[target]["metadata"]["uid"]
                    or digest(current.get("spec")) != digest(expected[target].get("spec"))):
                raise ValueError("repair target changed concurrently: " + target)
            if verb == "delete":
                # Record intent first: an API timeout can follow successful deletion.
                changes.append((target, None))
                _delete_exact(kind, name, namespace, current)
                continue
            # Dry-run uses the exact qualified operation semantics, then applies
            # its resulting spec with atomic UID/resourceVersion preconditions.
            desired = _json([*operation["argv"], "-n", namespace, "--dry-run=server", "-o", "json"])
            patch = [{"op": "test", "path": "/metadata/uid", "value": current["metadata"]["uid"]},
                     {"op": "test", "path": "/metadata/resourceVersion", "value": current["metadata"]["resourceVersion"]},
                     {"op": "replace", "path": "/spec", "value": desired["spec"]}]
            applied = _json(["patch", target, "-n", namespace, "--type=json", "-p", json.dumps(patch), "-o", "json"])
            expected[target] = applied
            changes.append((target, applied))
        after = verify("real_after")
        resolution = score_resolution(before["state"], after["state"])
        sla = after["state"].get("sla", {})
        passed = bool(after["workload"].completed and not after["workload"].failed
                      and after["workload"].application_failures == 0
                      and resolution["resolved"] and not sla.get("violated")
                      and sla.get("global_sla", {}).get("healthy") and after.get("ready"))
        if not passed:
            raise RuntimeError("real post-action workload, symptom or SLA verification failed")
        return {"success": True, "plan_sha256": plan["plan_sha256"], "namespace": namespace,
                "cluster_sha256": bound["cluster_sha256"], "resolution": resolution,
                "after_sla": sla, "applied_targets": [t for t, _ in changes]}
    except Exception as exc:
        rollback = []
        seen = set()
        for target, applied in reversed(changes):
            if target in seen:
                continue
            seen.add(target)
            original = bound["snapshots"][target]
            try:
                if applied is None:
                    remaining = _run(["get", target, "-n", namespace, "--ignore-not-found", "-o", "json"])
                    if remaining.returncode:
                        raise RuntimeError("cannot establish whether the Chaos resource was deleted")
                    if remaining.stdout.strip():
                        current = json.loads(remaining.stdout)
                        if (current["metadata"]["uid"] == original["metadata"]["uid"]
                                and not current["metadata"].get("deletionTimestamp")):
                            rollback.append({"target": target, "restored": True, "unchanged": True})
                            continue
                        raise RuntimeError("Chaos object still deleting or replaced; automatic recreation refused")
                    restored = {k: original[k] for k in ("apiVersion", "kind", "spec")}
                    restored["metadata"] = {k: original["metadata"][k] for k in ("name", "namespace", "labels", "annotations") if k in original["metadata"]}
                    _json(["create", "-f", "-", "-o", "json"], payload=restored)
                else:
                    current = _json(["get", target, "-n", namespace, "-o", "json"])
                    if current["metadata"]["uid"] != applied["metadata"]["uid"] or digest(current["spec"]) != digest(applied["spec"]):
                        raise ValueError("concurrent change; rollback refused")
                    patch = [{"op": "test", "path": "/metadata/resourceVersion", "value": current["metadata"]["resourceVersion"]},
                             {"op": "replace", "path": "/spec", "value": original["spec"]}]
                    _json(["patch", target, "-n", namespace, "--type=json", "-p", json.dumps(patch), "-o", "json"])
                rollback.append({"target": target, "restored": True})
            except Exception as rollback_error:
                rollback.append({"target": target, "restored": False, "error": str(rollback_error)})
        return {"success": False, "error": str(exc), "rollback": rollback, "plan_sha256": plan["plan_sha256"]}


def observation_verifier(plan, namespace, args, root):
    from .sparse_live_manifest import SparseManifestBundle
    from .sparse_live_session import SparseLiveTwinSession
    from .sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
    from .twin_spec_builder import TwinSpec
    observer = SparseLiveTwinVerifier(SparseLiveVerifierConfig(
        source_namespace=namespace, application_source_root=args.application_source_root,
        state_abstraction_root=str(Path(args.state_abstraction_root).resolve()),
        workload_rate=plan["workload_rate"], workload_duration_seconds=plan["workload_duration_seconds"]))
    observer.runtime_profile = observer._profile({"services": plan["services"]})
    if observer.runtime_profile.source_namespace != namespace:
        raise ValueError("explicit real namespace does not contain the qualified application")
    observer.work_root = root
    observer.environment_sha256 = plan.get("environment_sha256")
    observer._incident_state = plan["expected_incident_state"]
    observer._incident_reference = plan["reference_configuration"]
    observer.selected_services, observer.selected_paths = plan["selected_services"], plan["selected_paths"]
    observer._incident_spec = TwinSpec("real", namespace, "real_observation", plan["selected_services"], [], [],
        resource_summary={"incident_affected_services": plan["affected_services"]})
    objects = _json(["get", "deployments,statefulsets,services", "-n", namespace, "-o", "json"])["items"]
    refs = [{"kind": o["kind"], "name": o["metadata"]["name"], "namespace": namespace}
            for o in objects if o["metadata"]["name"] in plan["selected_services"]]
    class ExistingApplicationObservation:
        # Reuse read-only readiness checks without granting Twin lifecycle
        # methods or weakening the isolated session's namespace checks.
        _baseline_snapshot = SparseLiveTwinSession._baseline_snapshot
        wait_for_clean_baseline = SparseLiveTwinSession.wait_for_clean_baseline

        def __init__(self):
            self.namespace = namespace
            self.bundle = SparseManifestBundle(namespace, namespace, objects=objects, object_refs=refs)
            self.applied = True

    session = ExistingApplicationObservation()
    observer.session = session
    def verify(phase):
        ready = session.wait_for_clean_baseline().ready if phase == "real_after" else False
        capture = observer._capture_phase(phase, require_trace_coverage=phase == "real_after")
        capture["ready"] = ready
        return capture
    return verify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--bindings", default=None, help="Explicit real Chaos resource names/UIDs JSON")
    ap.add_argument("--application_source_root", required=True)
    ap.add_argument("--state_abstraction_root", default="state_abstraction_full")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--execute", action="store_true", help="Apply to the explicitly selected real incident")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text()); validate_plan(plan)
    bindings = json.loads(Path(args.bindings).read_text()) if args.bindings else {}
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=False)
    with bound_context(args.context) as cluster_sha:
        bound = bind_plan(plan, args.namespace, cluster_sha, bindings)
        (root / "bound_plan.json").write_text(json.dumps(bound, indent=2))
        result = {"status": "prepared", "plan_sha256": plan["plan_sha256"], "namespace": args.namespace}
        if args.execute:
            verify = observation_verifier(plan, args.namespace, args, root)
            result = apply_bound_plan(plan, bound, verify=verify)
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("status", "success", "plan_sha256", "namespace") if k in result}))
        if args.execute and not result.get("success"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
