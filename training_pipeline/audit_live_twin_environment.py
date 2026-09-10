from __future__ import annotations

"""Read-only preflight for the live Kubernetes digital-twin environment.

This audit intentionally performs no deployment, fault injection, remediation, or
namespace mutation.  It answers the questions needed before wiring the final live
Twin verifier:

* is kubectl configured and is the cluster reachable?;
* which AIOpsLab benchmark namespaces/apps are currently present?;
* is Chaos Mesh installed and which CRDs are available?;
* does the current identity have the minimal read/write verbs the Twin will need?;
* is the local AIOpsLab checkout importable and does it expose the expected fault
  injector modules?

The result is diagnostic only.  A PASS means the environment is *capable* of the
next live reinjection experiment; it does not validate Twin reproduction quality.
"""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "command_not_found"}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": "timeout",
        }


def _kubectl_json(args: list[str], *, timeout: int = 30) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    result = _run(["kubectl", *args, "-o", "json"], timeout=timeout)
    if not result["ok"]:
        return None, result
    try:
        return json.loads(result["stdout"]), result
    except Exception as exc:
        result = dict(result)
        result["ok"] = False
        result["stderr"] = f"invalid_json:{type(exc).__name__}:{exc}"
        return None, result


def _auth_can_i(verb: str, resource: str, namespace: str | None = None) -> bool | None:
    cmd = ["kubectl", "auth", "can-i", verb, resource]
    if namespace:
        cmd += ["-n", namespace]
    result = _run(cmd)
    if not result["ok"] and result["stdout"].strip().lower() not in {"yes", "no"}:
        return None
    text = result["stdout"].strip().lower()
    if text == "yes":
        return True
    if text == "no":
        return False
    return None


def _module_probe(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "importable": True,
            "file": str(getattr(module, "__file__", "") or ""),
        }
    except Exception as exc:
        return {
            "importable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only live Kubernetes Twin environment audit")
    ap.add_argument("--aiopslab_root", default="~/multiagent_IBM/AIOpsLab")
    ap.add_argument("--preferred_namespace", default="test-social-network")
    args = ap.parse_args()

    aiopslab_root = Path(args.aiopslab_root).expanduser().resolve()
    preferred_namespace = str(args.preferred_namespace)

    # Probe the checkout directly even when AIOpsLab is not installed as an
    # editable package in the active venv.  This does not modify the checkout.
    if aiopslab_root.exists() and str(aiopslab_root) not in sys.path:
        sys.path.insert(0, str(aiopslab_root))

    binaries = {
        name: shutil.which(name)
        for name in ("kubectl", "helm")
    }

    current_context = _run(["kubectl", "config", "current-context"])
    cluster_info = _run(["kubectl", "cluster-info"], timeout=20)
    nodes_obj, nodes_raw = _kubectl_json(["get", "nodes"], timeout=30)
    ns_obj, ns_raw = _kubectl_json(["get", "namespaces"], timeout=30)
    crd_obj, _crd_raw = _kubectl_json(["get", "crds"], timeout=30)

    node_names: list[str] = []
    if isinstance(nodes_obj, dict):
        for item in nodes_obj.get("items", []) or []:
            name = ((item.get("metadata") or {}).get("name"))
            if name:
                node_names.append(str(name))

    namespaces: list[str] = []
    if isinstance(ns_obj, dict):
        for item in ns_obj.get("items", []) or []:
            name = ((item.get("metadata") or {}).get("name"))
            if name:
                namespaces.append(str(name))

    crds: list[str] = []
    if isinstance(crd_obj, dict):
        for item in crd_obj.get("items", []) or []:
            name = ((item.get("metadata") or {}).get("name"))
            if name:
                crds.append(str(name))

    chaos_crds = sorted(
        name for name in crds
        if name.endswith(".chaos-mesh.org")
    )

    benchmark_namespaces = sorted(
        ns for ns in namespaces
        if ns.startswith("test-") or ns == "chaos-mesh"
    )

    namespace_probe: dict[str, Any] = {}
    if preferred_namespace in namespaces:
        pods_obj, pods_raw = _kubectl_json(["get", "pods", "-n", preferred_namespace], timeout=30)
        deployments_obj, deployments_raw = _kubectl_json(
            ["get", "deployments", "-n", preferred_namespace], timeout=30
        )
        services_obj, services_raw = _kubectl_json(
            ["get", "services", "-n", preferred_namespace], timeout=30
        )
        namespace_probe = {
            "present": True,
            "pods": len((pods_obj or {}).get("items", []) or []) if isinstance(pods_obj, dict) else None,
            "deployments": len((deployments_obj or {}).get("items", []) or []) if isinstance(deployments_obj, dict) else None,
            "services": len((services_obj or {}).get("items", []) or []) if isinstance(services_obj, dict) else None,
            "pods_query_ok": pods_raw["ok"],
            "deployments_query_ok": deployments_raw["ok"],
            "services_query_ok": services_raw["ok"],
        }
    else:
        namespace_probe = {"present": False}

    # Read permissions are required for state collection. Mutation permissions are
    # reported only; this audit never exercises them.
    auth = {
        "get_pods": _auth_can_i("get", "pods", preferred_namespace),
        "get_logs": _auth_can_i("get", "pods/log", preferred_namespace),
        "get_deployments": _auth_can_i("get", "deployments.apps", preferred_namespace),
        "patch_deployments": _auth_can_i("patch", "deployments.apps", preferred_namespace),
        "delete_pods": _auth_can_i("delete", "pods", preferred_namespace),
        "create_podchaos": _auth_can_i("create", "podchaos.chaos-mesh.org", preferred_namespace),
        "delete_podchaos": _auth_can_i("delete", "podchaos.chaos-mesh.org", preferred_namespace),
        "create_networkchaos": _auth_can_i("create", "networkchaos.chaos-mesh.org", preferred_namespace),
        "delete_networkchaos": _auth_can_i("delete", "networkchaos.chaos-mesh.org", preferred_namespace),
        "create_stresschaos": _auth_can_i("create", "stresschaos.chaos-mesh.org", preferred_namespace),
        "delete_stresschaos": _auth_can_i("delete", "stresschaos.chaos-mesh.org", preferred_namespace),
    }

    local = {
        "aiopslab_root": str(aiopslab_root),
        "exists": aiopslab_root.exists(),
        "is_git_checkout": (aiopslab_root / ".git").exists(),
        "processed_states_exists": (aiopslab_root / "processed_states").exists(),
        "dynamic_results_exists": (aiopslab_root / "dynamic_generated_scenario_results_all_new").exists(),
        "modules": {
            name: _module_probe(name)
            for name in (
                "aiopslab",
                "aiopslab.generators.fault.base",
                "aiopslab.generators.fault.inject_app",
                "aiopslab.generators.fault.inject_operator",
                "aiopslab.generators.fault.inject_symp",
            )
        },
    }

    critical = {
        "kubectl_present": bool(binaries["kubectl"]),
        "cluster_reachable": bool(cluster_info["ok"] and nodes_raw["ok"]),
        "nodes_visible": bool(node_names),
        "namespaces_visible": bool(ns_raw["ok"]),
        "aiopslab_checkout_present": bool(local["exists"]),
        "aiopslab_importable": bool(local["modules"]["aiopslab"]["importable"]),
    }
    environment_ready = all(critical.values())

    report = {
        "status": "PASS_ENVIRONMENT_READY" if environment_ready else "BLOCKED_ENVIRONMENT",
        "read_only": True,
        "mutated_cluster": False,
        "binaries": binaries,
        "kubernetes": {
            "current_context": current_context["stdout"] if current_context["ok"] else None,
            "cluster_reachable": bool(cluster_info["ok"]),
            "node_count": len(node_names),
            "nodes": node_names,
            "benchmark_namespaces": benchmark_namespaces,
            "preferred_namespace": preferred_namespace,
            "preferred_namespace_probe": namespace_probe,
        },
        "chaos_mesh": {
            "installed_by_crd_evidence": bool(chaos_crds),
            "crd_count": len(chaos_crds),
            "crds": chaos_crds,
        },
        "authorization": auth,
        "local_aiopslab": local,
        "critical_checks": critical,
        "next_gate": (
            "live_oracle_reinjection_preflight"
            if environment_ready
            else "fix_environment_before_live_reinjection"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
