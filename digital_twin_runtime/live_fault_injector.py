from __future__ import annotations

import json
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from training_pipeline.schemas import FaultLabel, normalize_fault_mechanism

from .sparse_live_session import SparseLiveTwinSession
from .live_capabilities import live_injector_implemented


def _run(args: list[str], payload: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        input=json.dumps(payload) if payload is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str], operation: str) -> str:
    if proc.returncode != 0:
        raise RuntimeError(f"{operation} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _json(args: list[str]) -> dict[str, Any]:
    return json.loads(_ok(_run([*args, "-o", "json"]), "kubectl read") or "{}")


@dataclass
class FaultManifestation:
    service: str
    mechanism: str
    manifested: bool
    elapsed_seconds: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    condition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveFaultHandle:
    def __init__(
        self,
        session: SparseLiveTwinSession,
        fault: FaultLabel,
        original_deployment: dict[str, Any],
        restore_mode: str = "replace_deployment",
        original_service: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.fault = fault
        self.original_deployment = original_deployment
        self.restore_mode = restore_mode
        self.original_service = original_service
        self.restored = False

    def wait_for_manifestation(
        self, *, timeout_seconds: float = 60.0, poll_seconds: float = 0.5
    ) -> FaultManifestation:
        started = time.monotonic()
        evidence: list[dict[str, Any]] = []
        while time.monotonic() - started < timeout_seconds:
            mechanism = normalize_fault_mechanism(self.fault.fault_mechanism)
            if mechanism == "target_port_misconfig":
                service = _json([
                    "get", "service", self.fault.service,
                    "-n", self.session.namespace,
                ])
                ports = (service.get("spec", {}) or {}).get("ports", []) or []
                target_ports = [row.get("targetPort") for row in ports]
                endpoint = _json([
                    "get", "endpoints", self.fault.service,
                    "-n", self.session.namespace,
                ])
                address_count = sum(
                    len((subset or {}).get("addresses", []) or [])
                    for subset in endpoint.get("subsets", []) or []
                )
                evidence = [{
                    "service": self.fault.service,
                    "target_ports": target_ports,
                    "ready_endpoint_addresses": address_count,
                }]
                if 9999 in target_ports and address_count > 0:
                    return FaultManifestation(
                        service=self.fault.service,
                        mechanism=self.fault.fault_mechanism,
                        manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        evidence=evidence,
                        condition="Service targetPort=9999 while endpoint addresses remain present",
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            if mechanism == "scale_replicas_zero":
                deployment = _json([
                    "get", "deployment", self.fault.service,
                    "-n", self.session.namespace,
                ])
                desired = int((deployment.get("spec", {}) or {}).get("replicas", -1) or 0)
                ready = int((deployment.get("status", {}) or {}).get("readyReplicas", 0) or 0)
                endpoint = _json([
                    "get", "endpoints", self.fault.service,
                    "-n", self.session.namespace,
                ])
                address_count = sum(
                    len((subset or {}).get("addresses", []) or [])
                    for subset in endpoint.get("subsets", []) or []
                )
                evidence = [{
                    "deployment": self.fault.service,
                    "desired_replicas": desired,
                    "ready_replicas": ready,
                    "ready_endpoint_addresses": address_count,
                }]
                if desired == 0 and ready == 0 and address_count == 0:
                    return FaultManifestation(
                        service=self.fault.service,
                        mechanism=self.fault.fault_mechanism,
                        manifested=True,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        evidence=evidence,
                        condition="desired replicas=0, ready replicas=0, and zero ready endpoints",
                    )
                time.sleep(max(0.1, poll_seconds))
                continue
            pods = _json([
                "get", "pods", "-n", self.session.namespace,
                "-l", f"service={self.fault.service}",
            ]).get("items", []) or []
            evidence = []
            for pod in pods:
                row = {
                    "pod": str((pod.get("metadata", {}) or {}).get("name") or ""),
                    "phase": str((pod.get("status", {}) or {}).get("phase") or ""),
                    "conditions": [],
                }
                for condition in (pod.get("status", {}) or {}).get("conditions", []) or []:
                    item = {
                        "type": condition.get("type"),
                        "status": condition.get("status"),
                        "reason": condition.get("reason"),
                        "message": condition.get("message"),
                    }
                    row["conditions"].append(item)
                    if (
                        item["type"] == "PodScheduled"
                        and str(item["status"]).lower() == "false"
                        and item["reason"] == "Unschedulable"
                    ):
                        endpoint = _json([
                            "get", "endpoints", self.fault.service,
                            "-n", self.session.namespace,
                        ])
                        address_count = sum(
                            len((subset or {}).get("addresses", []) or [])
                            for subset in endpoint.get("subsets", []) or []
                        )
                        row["ready_endpoint_addresses"] = address_count
                        if address_count != 0:
                            continue
                        evidence.append(row)
                        return FaultManifestation(
                            service=self.fault.service,
                            mechanism=self.fault.fault_mechanism,
                            manifested=True,
                            elapsed_seconds=round(time.monotonic() - started, 3),
                            evidence=evidence,
                            condition="PodScheduled=False/Unschedulable and zero ready endpoints",
                        )
                evidence.append(row)
            time.sleep(max(0.1, poll_seconds))
        return FaultManifestation(
            service=self.fault.service,
            mechanism=self.fault.fault_mechanism,
            manifested=False,
            elapsed_seconds=round(time.monotonic() - started, 3),
            evidence=evidence,
            condition="PodScheduled=False/Unschedulable and zero ready endpoints",
        )

    def restore(self) -> None:
        if self.restore_mode == "apply_service":
            if self.original_service is None:
                raise RuntimeError("original Service snapshot is unavailable")
            _ok(_run(["apply", "-f", "-"], self.original_service), "restore original Service targetPort")
            self.restored = True
            return
        if self.restore_mode == "scale_replicas":
            replicas = int((self.original_deployment.get("spec", {}) or {}).get("replicas", 1) or 1)
            _ok(_run([
                "scale", f"deployment/{self.fault.service}",
                "-n", self.session.namespace, f"--replicas={replicas}",
            ]), "restore original Deployment replica count")
            self.restored = True
            return
        _ok(_run([
            "delete", "deployment", self.fault.service,
            "-n", self.session.namespace, "--wait=true", "--timeout=60s",
        ]), "remove faulted scheduling Deployment")
        _ok(_run(["apply", "-f", "-"], self.original_deployment), "restore scheduling Deployment")
        self.restored = True


def inject_predicted_fault(
    session: SparseLiveTwinSession,
    fault: FaultLabel,
) -> LiveFaultHandle:
    """Inject exactly the agent-predicted mechanism; never infer from oracle state."""
    if not session.applied:
        raise RuntimeError("Twin baseline manifests are not applied")
    if not fault.is_injectible():
        raise ValueError("RCA fault is generic, unknown, or mechanism/type-inconsistent")
    if not live_injector_implemented(fault):
        raise NotImplementedError(
            "predicted service/mechanism pair has not passed the complete live workload audit"
        )
    mechanism = normalize_fault_mechanism(fault.fault_mechanism)

    selected_deployments = {
        row["name"] for row in session.bundle.object_refs
        if row.get("kind") == "Deployment"
    }
    selected_services = {
        row["name"] for row in session.bundle.object_refs
        if row.get("kind") == "Service"
    }
    if fault.service not in selected_deployments:
        raise ValueError("predicted service is not a selected Twin Deployment")
    original = next(
        deepcopy(obj) for obj in session.bundle.objects
        if obj.get("kind") == "Deployment"
        and (obj.get("metadata", {}) or {}).get("name") == fault.service
    )
    if mechanism == "target_port_misconfig":
        if fault.service not in selected_services:
            raise ValueError("predicted service is not a selected Twin Service")
        original_service = next(
            deepcopy(obj) for obj in session.bundle.objects
            if obj.get("kind") == "Service"
            and (obj.get("metadata", {}) or {}).get("name") == fault.service
        )
        faulted_service = deepcopy(original_service)
        changed = False
        for port in faulted_service["spec"].get("ports", []) or []:
            if port.get("targetPort") == 9090:
                port["targetPort"] = 9999
                changed = True
        if not changed:
            raise ValueError("selected Service has no numeric targetPort 9090")
        _ok(_run(["apply", "-f", "-"], faulted_service), "inject predicted targetPort misconfiguration")
        return LiveFaultHandle(
            session, fault, original, restore_mode="apply_service",
            original_service=original_service,
        )
    if mechanism == "scale_replicas_zero":
        _ok(_run([
            "scale", f"deployment/{fault.service}",
            "-n", session.namespace, "--replicas=0",
        ]), "inject predicted zero-replica fault")
        return LiveFaultHandle(
            session, fault, original, restore_mode="scale_replicas"
        )
    if mechanism != "assign_to_non_existent_node":
        raise NotImplementedError(f"live injector is not implemented for {mechanism!r}")
    faulted = deepcopy(original)
    faulted["spec"]["template"]["spec"]["nodeSelector"] = {
        "kubernetes.io/hostname": "aiops-twin-non-existent-node"
    }
    # Match AIOpsLab's actual mechanism: delete/reapply instead of a rolling
    # patch. This ensures the old healthy ReplicaSet cannot continue serving.
    _ok(_run([
        "delete", "deployment", fault.service,
        "-n", session.namespace, "--wait=true", "--timeout=60s",
    ]), "remove clean Deployment before predicted scheduling fault")
    _ok(_run(["apply", "-f", "-"], faulted), "inject predicted scheduling fault")
    return LiveFaultHandle(session, fault, original)
