from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .sparse_live_manifest import SparseManifestBundle


_TWIN_LABEL = "aiopslab.ibm/managed-by"
_TWIN_VALUE = "sparse-live-twin"
_NAMESPACE_PREFIX = "aiops-twin-"


def _kubectl(
    args: list[str], *, input_obj: dict[str, Any] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        input=json.dumps(input_obj) if input_obj is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _require_success(proc: subprocess.CompletedProcess[str], operation: str) -> str:
    if proc.returncode != 0:
        raise RuntimeError(f"{operation} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _get_json(args: list[str]) -> dict[str, Any]:
    proc = _kubectl([*args, "-o", "json"])
    return json.loads(_require_success(proc, f"kubectl {' '.join(args)}") or "{}")


@dataclass
class SparseTwinBaselineResult:
    namespace: str
    ready: bool
    elapsed_seconds: float
    controllers: list[dict[str, Any]] = field(default_factory=list)
    unexpected_pods: list[str] = field(default_factory=list)
    failed_pods: list[dict[str, Any]] = field(default_factory=list)
    condition: str = "all_selected_controllers_available"
    stable_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SparseLiveTwinSession:
    """Lifecycle owner for one isolated, rendered sparse Kubernetes Twin.

    The session accepts already sanitized manifests. It never reads hidden
    scenario state and it will only delete a namespace carrying its ownership
    label and reserved prefix.
    """

    def __init__(self, bundle: SparseManifestBundle) -> None:
        if not bundle.read_only:
            raise ValueError("manifest bundle provenance must be read-only")
        if not bundle.target_namespace.startswith(_NAMESPACE_PREFIX):
            raise ValueError(
                f"Twin namespace must start with {_NAMESPACE_PREFIX!r}"
            )
        if bundle.source_namespace == bundle.target_namespace:
            raise ValueError("source and Twin namespaces must differ")
        self.bundle = bundle
        self.namespace = bundle.target_namespace
        self.created = False
        self.applied = False

    def create_namespace(self) -> None:
        probe = _kubectl(["get", "namespace", self.namespace, "-o", "name"])
        if probe.returncode == 0:
            raise RuntimeError(f"refusing to reuse existing namespace {self.namespace!r}")
        namespace = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.namespace,
                "labels": {
                    _TWIN_LABEL: _TWIN_VALUE,
                    "aiopslab.ibm/twin": "sparse-live",
                },
            },
        }
        _require_success(
            _kubectl(["apply", "-f", "-"], input_obj=namespace),
            "create isolated Twin namespace",
        )
        self.created = True

    def apply_manifests(self) -> None:
        if not self.created:
            raise RuntimeError("Twin namespace has not been created")
        manifest_list = {
            "apiVersion": "v1",
            "kind": "List",
            "items": self.bundle.objects,
        }
        _require_success(
            _kubectl(["apply", "-f", "-"], input_obj=manifest_list),
            "apply sparse Twin manifests",
        )
        self.applied = True

    def _baseline_snapshot(self) -> SparseTwinBaselineResult:
        controllers: list[dict[str, Any]] = []
        expected_pod_labels: set[str] = set()
        ready = True
        for kind in ("deployments", "statefulsets"):
            data = _get_json(["get", kind, "-n", self.namespace])
            for obj in data.get("items", []) or []:
                meta = obj.get("metadata", {}) or {}
                spec = obj.get("spec", {}) or {}
                status = obj.get("status", {}) or {}
                desired = int(spec.get("replicas", 1) or 0)
                available = int(status.get("availableReplicas", 0) or 0)
                observed = int(status.get("observedGeneration", 0) or 0)
                generation = int(meta.get("generation", 0) or 0)
                row_ready = desired > 0 and available >= desired and observed >= generation
                ready = ready and row_ready
                selectors = (spec.get("selector", {}) or {}).get("matchLabels", {}) or {}
                expected_pod_labels.update(
                    str(v) for k, v in selectors.items() if k in {"app", "service"}
                )
                controllers.append({
                    "kind": str(obj.get("kind") or kind),
                    "name": str(meta.get("name") or ""),
                    "desired": desired,
                    "available": available,
                    "observed_generation": observed,
                    "generation": generation,
                    "ready": row_ready,
                })

        pods = _get_json(["get", "pods", "-n", self.namespace]).get("items", []) or []
        unexpected: list[str] = []
        failed: list[dict[str, Any]] = []
        for pod in pods:
            meta = pod.get("metadata", {}) or {}
            labels = meta.get("labels", {}) or {}
            name = str(meta.get("name") or "")
            identity = str(labels.get("service") or labels.get("app") or "")
            if identity and identity not in expected_pod_labels:
                unexpected.append(name)
            status = pod.get("status", {}) or {}
            phase = str(status.get("phase") or "")
            bad_waiting = []
            for container in status.get("containerStatuses", []) or []:
                waiting = ((container.get("state") or {}).get("waiting") or {})
                reason = str(waiting.get("reason") or "")
                if reason in {"CrashLoopBackOff", "CreateContainerError", "ErrImagePull", "ImagePullBackOff"}:
                    bad_waiting.append(reason)
            if phase == "Failed" or bad_waiting:
                failed.append({"name": name, "phase": phase, "reasons": bad_waiting})
        ready = ready and bool(controllers) and not unexpected and not failed
        return SparseTwinBaselineResult(
            namespace=self.namespace,
            ready=ready,
            elapsed_seconds=0.0,
            controllers=controllers,
            unexpected_pods=sorted(unexpected),
            failed_pods=failed,
        )

    def wait_for_clean_baseline(
        self, *, timeout_seconds: float = 180.0, poll_seconds: float = 1.0,
        stability_seconds: float = 15.0,
    ) -> SparseTwinBaselineResult:
        if not self.applied:
            raise RuntimeError("sparse manifests have not been applied")
        started = time.monotonic()
        healthy_since: float | None = None
        last: SparseTwinBaselineResult | None = None
        while time.monotonic() - started < timeout_seconds:
            last = self._baseline_snapshot()
            last.elapsed_seconds = round(time.monotonic() - started, 3)
            if last.ready:
                healthy_since = healthy_since or time.monotonic()
                last.stable_seconds = round(time.monotonic() - healthy_since, 3)
                if last.stable_seconds >= max(0.0, stability_seconds):
                    return last
            else:
                healthy_since = None
            time.sleep(max(0.1, poll_seconds))
        if last is None:
            last = self._baseline_snapshot()
        last.elapsed_seconds = round(time.monotonic() - started, 3)
        return last

    def destroy(self, *, timeout_seconds: float = 300.0) -> None:
        obj = _get_json(["get", "namespace", self.namespace])
        labels = (obj.get("metadata", {}) or {}).get("labels", {}) or {}
        if labels.get(_TWIN_LABEL) != _TWIN_VALUE:
            raise RuntimeError("refusing to delete namespace without Twin ownership label")
        _require_success(
            _kubectl([
                "delete", "namespace", self.namespace,
                "--wait=true", f"--timeout={int(timeout_seconds)}s",
            ]),
            "destroy isolated Twin namespace",
        )
        self.created = False
        self.applied = False
