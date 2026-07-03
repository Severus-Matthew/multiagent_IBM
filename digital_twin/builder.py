"""
digital_twin/builder.py

Builds a K8s digital twin for a given scenario:
  1. Instantiates the AIOpsLab problem class from the scenario's class_name
  2. Deploys the full application stack via existing Helm charts
  3. Trims bystander pods identified by digital_twin_filter
  4. Provides inject_fault / recover_fault / reset_fault lifecycle methods

Usage:
    import json, sys
    sys.path.insert(0, "/path/to/AIOpsLab")

    from digital_twin.builder import DigitalTwin

    state = json.load(open("processed_state/state_abstraction.json"))
    compressed = json.load(open("processed_state/state_abstraction_compressed.json"))

    twin = DigitalTwin(state, compressed)
    twin_info = twin.build()           # deploy + trim
    twin.inject_fault()                # replicate scenario fault

    snap = twin.pod_health_snapshot()  # baseline after fault injection

    # ... run solution iteration ...

    twin.reset_fault()                 # recover + re-inject for next iteration
    twin.teardown()                    # uninstall when done
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
for _p in [str(_ROOT), str(_ROOT / "AIOpsLab")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from state_abstraction_full.digital_twin_filter import get_digital_twin_services


class DigitalTwin:
    """
    Manages the full lifecycle of one K8s digital twin deployment.

    The twin wraps an AIOpsLab problem class (e.g. MongoDBAuthMissingAnalysis)
    and provides a clean interface for the training loop:

        build()            → deploy full stack, trim bystanders
        inject_fault()     → replicate scenario fault
        reset_fault()      → recover + re-inject (multi-iteration)
        teardown()         → recover + helm uninstall
        pod_health_snapshot() → {pod: is_ready} dict for reward
    """

    def __init__(self, state: dict, compressed: dict | None = None):
        self.state = state
        self.compressed = compressed

        fc = state.get("fault_context", {})
        self.problem_id: str = fc.get("problem_id", "unknown")
        self.class_name: str = fc.get("class_name", "")
        self.faulty_service: str = fc.get("faulty_service", "")
        self.fault_family: str = fc.get("fault_family", "")

        self.namespace: str = ""
        self.problem = None
        self.twin_filter: dict = {}
        self._fault_injected: bool = False
        self._built: bool = False

    # ── Public lifecycle ────────────────────────────────────────────────────

    def build(self) -> dict:
        """
        Deploy full application stack and trim bystander pods.

        Returns:
            twin_filter dict — keys: namespace, relevant_services,
            bystander_services, relevant_pods, bystander_pods, summary
        """
        from aiopslab.service.helm import Helm
        from aiopslab.service.kubectl import KubeCtl

        self.twin_filter = get_digital_twin_services(self.state, self.compressed)
        self.problem = self._instantiate_problem()
        self.namespace = self.problem.app.namespace

        release = self.problem.app.helm_configs.get("release_name", "")
        if Helm.exists_release(release, self.namespace):
            print(f"[DigitalTwin] '{release}' already deployed — waiting for ready")
            KubeCtl().wait_for_ready(self.namespace)
        else:
            print(f"[DigitalTwin] Deploying {self.problem_id} → ns={self.namespace}")
            self.problem.app.deploy()

        self._trim_bystanders()
        self._built = True
        print(
            f"[DigitalTwin] Built. "
            f"relevant={len(self.twin_filter.get('relevant_services', []))} "
            f"bystanders_removed={len(self.twin_filter.get('bystander_pods', []))}"
        )
        return self.twin_filter

    def inject_fault(self) -> None:
        """Inject the scenario fault using the problem's inject_fault method."""
        if not self._built:
            raise RuntimeError("Call build() before inject_fault()")
        print(
            f"[DigitalTwin] Injecting fault: "
            f"family={self.fault_family} service={self.faulty_service}"
        )
        self.problem.inject_fault()
        self._fault_injected = True

    def recover_fault(self) -> None:
        """Recover / undo the injected fault."""
        if self.problem and self._fault_injected:
            print("[DigitalTwin] Recovering fault")
            self.problem.recover_fault()
            self._fault_injected = False

    def reset_fault(self, stabilize_seconds: int = 15) -> None:
        """
        Recover the active fault, wait for pods to stabilize, then re-inject.
        Called between training iterations so each attempt starts from a
        freshly-faulted but otherwise healthy twin.
        """
        self.recover_fault()
        time.sleep(stabilize_seconds)
        self.inject_fault()

    def teardown(self) -> None:
        """Recover fault and uninstall the Helm release (releases cluster resources)."""
        self.recover_fault()
        print(f"[DigitalTwin] Tearing down namespace={self.namespace}")
        self.problem.app.delete()

    def pod_health_snapshot(self) -> dict[str, bool]:
        """
        Return {pod_name: is_ready} for every pod in the twin's namespace.

        Only pods in relevant_pod_names() carry reward signal; bystanders
        are already deleted and won't appear here.
        """
        from aiopslab.service.kubectl import KubeCtl
        kubectl = KubeCtl()
        pods = kubectl.list_pods(self.namespace)
        result: dict[str, bool] = {}
        for pod in pods.items:
            cs_list = pod.status.container_statuses or []
            ready = bool(cs_list) and all(cs.ready for cs in cs_list)
            result[pod.metadata.name] = ready
        return result

    def relevant_pod_names(self) -> list[str]:
        """Pod names that belong to the relevant (fault-affected) service set."""
        return self.twin_filter.get("relevant_pods", [])

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _instantiate_problem(self):
        """
        Dynamically import and instantiate the AIOpsLab problem class.

        All problem classes are re-exported by registry.py via star-imports,
        so any class_name in state_abstraction fault_context will resolve.

        For generated scenarios the faulty_service may differ from the
        hardcoded default inside the class — we patch it after construction.
        """
        if not self.class_name:
            raise ValueError(
                f"fault_context for '{self.problem_id}' has no class_name. "
                "Re-run state abstraction on this scenario."
            )

        registry_mod = importlib.import_module(
            "aiopslab.orchestrator.problems.registry"
        )
        cls = getattr(registry_mod, self.class_name, None)
        if cls is None:
            raise ValueError(
                f"Problem class '{self.class_name}' not found in "
                "aiopslab/orchestrator/problems/registry.py. "
                "Add the missing import there."
            )

        # Some classes accept faulty_service as a constructor arg; others don't.
        problem = self._safe_construct(cls)

        # Patch faulty_service for generated scenarios that differ from the default
        default_svc = getattr(problem, "faulty_service", None)
        if self.faulty_service and self.faulty_service != default_svc:
            print(
                f"[DigitalTwin] Patching faulty_service: "
                f"'{default_svc}' → '{self.faulty_service}'"
            )
            problem.faulty_service = self.faulty_service

        return problem

    def _safe_construct(self, cls):
        """
        Try constructing with faulty_service kwarg first; fall back to no args.
        This handles both classes that accept faulty_service (e.g. K8sTargetPort*)
        and those that hardcode it (e.g. MongoDBAuthMissing*).
        """
        if self.faulty_service:
            try:
                return cls(faulty_service=self.faulty_service)
            except TypeError:
                pass
        return cls()

    def _trim_bystanders(self) -> None:
        """
        Delete bystander pods to reduce resource consumption.

        Uses kubectl.exec_command so the AIOPSLAB_CLUSTER env var is
        respected (correct kind context for parallel workers).
        """
        from aiopslab.service.kubectl import KubeCtl
        kubectl = KubeCtl()
        bystander_pods = self.twin_filter.get("bystander_pods", [])

        if not bystander_pods:
            print("[DigitalTwin] No bystander pods to trim")
            return

        print(f"[DigitalTwin] Trimming {len(bystander_pods)} bystander pods")
        for pod in bystander_pods:
            cmd = (
                f"kubectl delete pod {pod} "
                f"-n {self.namespace} --ignore-not-found"
            )
            out = kubectl.exec_command(cmd)
            print(f"  {pod}: {out.strip()[:80]}")
