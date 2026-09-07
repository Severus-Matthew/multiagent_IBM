from __future__ import annotations

"""Warm-application regeneration: reuse a healthy deployment between scenarios.

``gen_and_telmetry.run_one`` inherits AIOpsLab's full lifecycle: install OpenEBS,
deploy Prometheus, uninstall and reinstall the whole application, wait for every
pod, inject, collect, recover, delete the namespace, tear down Prometheus and
OpenEBS. Measured on this host that is roughly 23 minutes per capture, almost
all of it spent rebuilding infrastructure the next scenario immediately rebuilds
again.

This module patches those lifecycle hooks so that a namespace which is already
healthy is reused: infrastructure setup and teardown become no-ops, the
application is deployed only when the namespace is not ready, and cleanup keeps
the application only when recovery provably restored it. Every reuse decision is
gated by :func:`namespace_is_clean`, which fails closed: any unavailable
controller, any surviving Chaos Mesh object, any Twin/generator overlay, or any
pod outside Running/Succeeded forces the original full teardown so the next
scenario starts from a fresh install. Cross-scenario contamination therefore
costs one slow scenario rather than silently poisoning the next capture.
"""

import json
import subprocess
import time
from typing import Any, Callable

WARM_JOURNAL: list[dict[str, Any]] = []

_CHAOS_KINDS = ("podchaos", "networkchaos", "stresschaos", "iochaos", "timechaos", "httpchaos")


def _kubectl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kubectl", *args], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def _items(kind: str, namespace: str) -> list[dict[str, Any]]:
    proc = _kubectl(["get", kind, "-n", namespace, "-o", "json"])
    if proc.returncode != 0:
        return []
    try:
        return list(json.loads(proc.stdout or "{}").get("items", []) or [])
    except json.JSONDecodeError:
        return []


def namespace_is_clean(namespace: str) -> tuple[bool, dict[str, Any]]:
    """Whether a namespace holds a fully available, un-faulted application."""
    report: dict[str, Any] = {"namespace": namespace}
    proc = _kubectl(["get", "namespace", namespace])
    if proc.returncode != 0:
        report["reason"] = "namespace_absent"
        return False, report
    deployments = _items("deployments", namespace)
    if not deployments:
        report["reason"] = "no_deployments"
        return False, report
    unavailable = []
    for row in deployments:
        spec_replicas = int((row.get("spec", {}) or {}).get("replicas", 0) or 0)
        available = int((row.get("status", {}) or {}).get("availableReplicas", 0) or 0)
        generation = (row.get("metadata", {}) or {}).get("generation")
        observed = (row.get("status", {}) or {}).get("observedGeneration")
        if spec_replicas <= 0 or available < spec_replicas or generation != observed:
            unavailable.append(str((row.get("metadata", {}) or {}).get("name")))
    if unavailable:
        report["reason"], report["unavailable_deployments"] = "deployments_unavailable", unavailable
        return False, report
    bad_pods = [
        str((p.get("metadata", {}) or {}).get("name"))
        for p in _items("pods", namespace)
        if str((p.get("status", {}) or {}).get("phase")) not in {"Running", "Succeeded"}
        or (p.get("metadata", {}) or {}).get("deletionTimestamp")
    ]
    if bad_pods:
        report["reason"], report["pods"] = "pods_not_running", bad_pods
        return False, report
    chaos = [
        f"{kind}/{(c.get('metadata', {}) or {}).get('name')}"
        for kind in _CHAOS_KINDS for c in _items(kind, namespace)
    ]
    if chaos:
        report["reason"], report["chaos_objects"] = "chaos_objects_present", chaos
        return False, report
    overlays = [
        str((c.get("metadata", {}) or {}).get("name"))
        for c in _items("configmaps", namespace)
        if str((c.get("metadata", {}) or {}).get("name") or "").startswith("twin-config-overlay-")
    ]
    if overlays:
        report["reason"], report["overlays"] = "fault_overlays_present", overlays
        return False, report
    report["reason"] = "clean"
    return True, report


def wait_until_clean(namespace: str, timeout: float) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        ok, last = namespace_is_clean(namespace)
        if ok:
            return True, last
        time.sleep(5)
    return False, last


def _journal(event: str, **details: Any) -> None:
    WARM_JOURNAL.append({"event": event, "unix": time.time(), **details})


def install_warm_application_mode(*, recovery_timeout: float = 300.0) -> dict[str, Any]:
    """Patch AIOpsLab lifecycle hooks for warm reuse. Idempotent."""
    from aiopslab.service.apps import hotelres, socialnet
    from aiopslab.service.kubectl import KubeCtl
    from aiopslab.service.telemetry.prometheus import Prometheus

    patched: dict[str, list[str]] = {}

    if not getattr(KubeCtl, "_warm_patched", False):
        original_exec: Callable[..., Any] = KubeCtl.exec_command
        original_wait: Callable[..., Any] = KubeCtl.wait_for_ready

        def exec_command(self: Any, command: str, input_data: Any = None) -> str:
            # OpenEBS is installed once per cluster; re-applying and deleting the
            # operator per scenario is what the 1200 s readiness wait paid for.
            if "openebs" in command and ("apply" in command or "delete" in command or "patch" in command):
                _journal("skip_openebs_command", command=command)
                return ""
            return original_exec(self, command, input_data)

        def wait_for_ready(self: Any, namespace: str, sleep: int = 2, max_wait: Any = None) -> Any:
            if namespace == "openebs":
                _journal("skip_openebs_wait")
                return None
            return original_wait(self, namespace, sleep, max_wait)

        original_wait_deletion: Callable[..., Any] = KubeCtl.wait_for_namespace_deletion

        def wait_for_namespace_deletion(self: Any, namespace: str, sleep: int = 2, max_wait: int = 300) -> Any:
            # The operator was never deleted, so waiting for its namespace to
            # vanish would only burn the full timeout and fail the capture.
            if namespace == "openebs":
                _journal("skip_openebs_namespace_deletion_wait")
                return None
            return original_wait_deletion(self, namespace, sleep, max_wait)

        KubeCtl.exec_command = exec_command  # type: ignore[assignment]
        KubeCtl.wait_for_ready = wait_for_ready  # type: ignore[assignment]
        KubeCtl.wait_for_namespace_deletion = wait_for_namespace_deletion  # type: ignore[assignment]
        KubeCtl._warm_patched = True  # type: ignore[attr-defined]
        patched["KubeCtl"] = ["exec_command", "wait_for_ready", "wait_for_namespace_deletion"]

    if not getattr(Prometheus, "_warm_patched", False):
        def teardown(self: Any) -> None:
            _journal("skip_prometheus_teardown")
        Prometheus.teardown = teardown  # type: ignore[assignment]
        Prometheus._warm_patched = True  # type: ignore[attr-defined]
        patched["Prometheus"] = ["teardown"]

    for module, class_name in ((hotelres, "HotelReservation"), (socialnet, "SocialNetwork")):
        cls = getattr(module, class_name)
        if getattr(cls, "_warm_patched", False):
            continue
        original_delete = cls.delete
        original_deploy = cls.deploy
        original_cleanup = cls.cleanup

        def delete(self: Any, _orig: Callable[..., Any] = original_delete) -> None:
            clean, report = namespace_is_clean(self.namespace)
            if clean:
                _journal("reuse_deployment_skip_delete", **report)
                self._warm_reused = True
                return
            _journal("full_delete", **report)
            self._warm_reused = False
            _orig(self)

        def deploy(self: Any, _orig: Callable[..., Any] = original_deploy) -> None:
            if getattr(self, "_warm_reused", False):
                clean, report = namespace_is_clean(self.namespace)
                if clean:
                    _journal("reuse_deployment_skip_deploy", **report)
                    return
                _journal("deployment_lost_between_delete_and_deploy", **report)
            _orig(self)

        def cleanup(self: Any, _orig: Callable[..., Any] = original_cleanup) -> None:
            # Called after recover_fault. Keep the application only when it has
            # provably returned to a clean state; otherwise fall back to the
            # original full teardown so the next scenario reinstalls.
            clean, report = wait_until_clean(self.namespace, recovery_timeout)
            if clean:
                _journal("recovery_verified_keep_deployment", **report)
                return
            _journal("recovery_not_clean_full_cleanup", **report)
            _orig(self)

        cls.delete = delete
        cls.deploy = deploy
        cls.cleanup = cleanup
        cls._warm_patched = True
        patched[class_name] = ["delete", "deploy", "cleanup"]
    return patched


def install_jaeger_port_isolation(local_port: int) -> dict[str, Any]:
    """Give this process its own local Jaeger port-forward port.

    Upstream ``TraceAPI`` always forwards ``localhost:16686`` and, when that port
    is already taken, gives up and queries whatever is listening there. Two
    regeneration shards on the same host would therefore read each other's
    (wrong-namespace) traces. Forwarding to a per-shard port removes the race
    without changing what is collected. Idempotent.
    """
    import subprocess as _subprocess
    import threading as _threading

    from aiopslab.observer.trace_api import TraceAPI

    if getattr(TraceAPI, "_isolated_port", None) == int(local_port):
        return {"TraceAPI": ["start_port_forward", "__init__"], "port": int(local_port)}
    original_init = TraceAPI.__init__

    def __init__(self: Any, namespace: str) -> None:  # noqa: N807
        self.port_forward_process = None
        self.namespace = namespace
        self.stop_event = _threading.Event()
        self.output_threads = []
        node_port = self.get_nodeport("jaeger", namespace) if namespace != "astronomy-shop" else None
        if node_port:
            self.base_url = f"http://localhost:{node_port}"
            return
        self.base_url = f"http://localhost:{int(local_port)}"
        if namespace == "astronomy-shop":
            self.base_url += "/jaeger/ui"
        self.start_port_forward()

    def start_port_forward(self: Any) -> None:
        port = int(local_port)
        for attempt in range(3):
            if self.is_port_in_use(port):
                _journal("jaeger_local_port_in_use", port=port, attempt=attempt + 1)
                time.sleep(3)
                continue
            if self.namespace == "astronomy-shop":
                pod_name = self.get_jaeger_pod_name()
                command = f"kubectl port-forward pod/{pod_name} {port}:16686 -n {self.namespace}"
            else:
                command = f"kubectl port-forward svc/jaeger {port}:16686 -n {self.namespace}"
            self.port_forward_process = _subprocess.Popen(
                command, shell=True, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE, text=True,
            )
            for stream in (self.port_forward_process.stdout, self.port_forward_process.stderr):
                _threading.Thread(target=self.print_output, args=(stream,), daemon=True).start()
            time.sleep(3)
            if self.port_forward_process.poll() is None:
                _journal("jaeger_port_forward_established", port=port, namespace=self.namespace)
                return
        raise RuntimeError(f"could not establish Jaeger port-forward on localhost:{port}")

    TraceAPI.__init__ = __init__  # type: ignore[assignment]
    TraceAPI.start_port_forward = start_port_forward  # type: ignore[assignment]
    TraceAPI._isolated_port = int(local_port)  # type: ignore[attr-defined]
    TraceAPI._original_init = original_init  # type: ignore[attr-defined]
    return {"TraceAPI": ["start_port_forward", "__init__"], "port": int(local_port)}
