from __future__ import annotations

import json
import hashlib
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .sparse_live_session import SparseLiveTwinSession


def _run(args: list[str], payload: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args], input=json.dumps(payload) if payload is not None else None,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=180,
    )


def _must(proc: subprocess.CompletedProcess[str], operation: str) -> str:
    if proc.returncode != 0:
        raise RuntimeError(f"{operation} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


@dataclass
class WorkloadResult:
    name: str
    endpoint: str
    completed: bool
    failed: bool
    elapsed_seconds: float
    requests_per_second: float | None
    total_requests: int | None
    non_success_responses: int
    application_failures: int
    probe_http_status: int | None
    probe_body: str
    required_service: str | None
    required_ready_endpoints: int | None
    socket_errors: dict[str, int]
    output: str
    execution_started: bool = False
    requested_rate: int | None = None
    requested_duration_seconds: int | None = None
    payload_sha256: str | None = None
    scope_policy: str = "minimal_root_reaching_request_path"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_wrk_output(text: str) -> tuple[float | None, int | None, int, int, dict[str, int]]:
    rate = None
    match = re.search(r"Requests/sec:\s*([0-9.]+)", text)
    if match:
        rate = float(match.group(1))
    total = None
    match = re.search(r"\b(\d+) requests in\b", text)
    if match:
        total = int(match.group(1))
    non_success = 0
    match = re.search(r"Non-2xx or 3xx responses:\s*(\d+)", text)
    if match:
        non_success = int(match.group(1))
    application_failures = 0
    match = re.search(r"Twin application failures:\s*(\d+)", text)
    if match:
        application_failures = int(match.group(1))
    errors: dict[str, int] = {}
    match = re.search(
        r"Socket errors:\s*connect\s+(\d+),\s*read\s+(\d+),\s*write\s+(\d+),\s*timeout\s+(\d+)",
        text,
    )
    if match:
        errors = dict(zip(("connect", "read", "write", "timeout"), map(int, match.groups())))
    return rate, total, non_success, application_failures, errors


def run_targeted_wrk(
    session: SparseLiveTwinSession,
    *,
    payload_script: str | Path,
    endpoint: str,
    required_service: str | None = None,
    frontend_service: str = "nginx-thrift",
    frontend_container: str | None = "nginx-thrift",
    frontend_port: int = 8080,
    rate: int = 10,
    duration_seconds: int = 10,
    timeout_seconds: float = 60.0,
) -> WorkloadResult:
    """Run the established AIOpsLab wrk2 client inside the Twin namespace."""
    if not session.applied:
        raise RuntimeError("Twin manifests are not applied")
    script_path = Path(payload_script)
    script = script_path.read_text() + """

-- Sparse-Twin verifier instrumentation. The SocialNetwork Lua handlers often
-- return HTTP 200 with an application failure string, which plain wrk counters
-- would incorrectly classify as success.
twin_application_failures = 0
response = function(status, headers, body)
  local text = string.lower(body or "")
  if status >= 400 or string.find(text, "failure", 1, true) or string.find(text, "error", 1, true) then
    twin_application_failures = twin_application_failures + 1
  end
end
done = function(summary, latency, requests)
  io.write("Twin application failures: " .. tostring(twin_application_failures) .. "\\n")
end
"""
    # Unique create-only resources are also safe in an existing application
    # namespace. Never apply over another verifier's workload or payload.
    suffix = uuid.uuid4().hex[:20]
    cm_name = "twin-wrk2-payload-" + suffix
    job_name = "twin-wrk2-job-" + suffix
    configmap = {
        "apiVersion": "v1", "kind": "ConfigMap",
        "metadata": {"name": cm_name, "namespace": session.namespace},
        "data": {script_path.name: script},
    }
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job_name, "namespace": session.namespace},
        "spec": {
            "backoffLimit": 0,
            # A deliberately wrong server binary may accept TCP connections but
            # never speak HTTP. Bound both individual requests and the Job so a
            # valid fault manifestation cannot stall a rollout worker forever.
            "activeDeadlineSeconds": max(15, int(duration_seconds) + 15),
            "template": {
                "metadata": {"labels": {"aiopslab.ibm/workload": "targeted-wrk2"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "wrk2", "image": "deathstarbench/wrk2-client:latest",
                        "imagePullPolicy": "IfNotPresent",
                        "args": [
                            "wrk", "-D", "exp", "-t", "2", "-c", "2",
                            "-d", f"{int(duration_seconds)}s", "-L", "-s",
                            f"/scripts/{script_path.name}", endpoint, "-R", str(int(rate)),
                            "--latency", "--timeout", "2s",
                        ],
                        "volumeMounts": [{
                            "name": "scripts", "mountPath": f"/scripts/{script_path.name}",
                            "subPath": script_path.name, "readOnly": True,
                        }],
                    }],
                    "volumes": [{"name": "scripts", "configMap": {"name": cm_name}}],
                },
            },
        },
    }
    _must(_run(["create", "-f", "-"], configmap), "create targeted workload ConfigMap")
    try:
        _must(_run(["create", "-f", "-"], job), "create targeted workload Job")
    except Exception:
        _run(["delete", "configmap", cm_name, "-n", session.namespace])
        raise
    started = time.monotonic()
    completed = failed = False
    output = ""
    probe_http_status = None
    probe_body = ""
    required_ready_endpoints = None
    execution_started = False
    try:
        while time.monotonic() - started < timeout_seconds:
            status_obj = json.loads(_must(
                _run(["get", "job", job_name, "-n", session.namespace, "-o", "json"]),
                "read targeted workload status",
            ))
            status = status_obj.get("status", {}) or {}
            if int(status.get("succeeded", 0) or 0) > 0:
                completed = True
                break
            if int(status.get("failed", 0) or 0) > 0:
                failed = True
                break
            time.sleep(0.5)
        output_proc = _run(["logs", "job/" + job_name, "-n", session.namespace])
        output = _must(output_proc, "read targeted workload output")
        workload_pods = json.loads(_must(
            _run(["get", "pods", "-n", session.namespace, "-l", f"job-name={job_name}", "-o", "json"]),
            "read targeted workload pod status",
        )).get("items", []) or []
        execution_started = any(
            bool(((status or {}).get("state") or {}).get("running")
                 or ((status or {}).get("state") or {}).get("terminated")
                 or (status or {}).get("lastState"))
            for pod in workload_pods
            for status in ((pod.get("status", {}) or {}).get("containerStatuses", []) or [])
        )
        pods_obj = json.loads(_must(
            _run(["get", "pods", "-n", session.namespace, "-l", f"service={frontend_service}", "-o", "json"]),
            "find frontend pod for application probe",
        ))
        running = [
            pod for pod in pods_obj.get("items", []) or []
            if (pod.get("status", {}) or {}).get("phase") == "Running"
        ]
        if running:
            pod_name = str((running[0].get("metadata", {}) or {}).get("name") or "")
            parsed = urlsplit(endpoint)
            path = parsed.path or "/"
            query = parsed.query
            if not query and path.endswith("/user-timeline/read"):
                query = "user_id=1&start=0&stop=10"
            if query:
                path += "?" + query
            exec_args = ["exec", "-n", session.namespace, pod_name]
            if frontend_container:
                exec_args.extend(["-c", frontend_container])
            exec_args.extend([
                "--", "curl", "-sS", "--max-time", "5", "-w", "\n__HTTP_STATUS__:%{http_code}",
                f"http://127.0.0.1:{int(frontend_port)}" + path,
            ])
            probe = _run(exec_args)
            probe_text = probe.stdout + probe.stderr
            marker = re.search(r"\n__HTTP_STATUS__:(\d+)\s*$", probe_text)
            if marker:
                probe_http_status = int(marker.group(1))
                probe_body = probe_text[: marker.start()]
            else:
                probe_body = probe_text
        if required_service:
            endpoint_obj = json.loads(_must(
                _run(["get", "endpoints", required_service, "-n", session.namespace, "-o", "json"]),
                "read required workload-path endpoints",
            ))
            required_ready_endpoints = sum(
                len((subset or {}).get("addresses", []) or [])
                for subset in endpoint_obj.get("subsets", []) or []
            )
    finally:
        _run(["delete", "job", job_name, "-n", session.namespace, "--wait=true", "--timeout=60s"])
        _run(["delete", "configmap", cm_name, "-n", session.namespace])
    requests_per_second, total_requests, non_success, application_failures, socket_errors = _parse_wrk_output(output)
    probe_lower = probe_body.lower()
    if (
        (probe_http_status is not None and probe_http_status >= 400)
        or "failure" in probe_lower or "error" in probe_lower
    ):
        application_failures = max(1, application_failures)
    if required_service and required_ready_endpoints == 0 and total_requests:
        # Some SocialNetwork handlers mask an unavailable Thrift dependency as
        # HTTP 200 with `{}`. A request path with zero endpoints cannot be
        # credited as successful merely because the frontend masked the error.
        application_failures = max(application_failures, total_requests)
    return WorkloadResult(
        name=job_name,
        endpoint=endpoint,
        completed=completed,
        failed=failed or not completed,
        elapsed_seconds=round(time.monotonic() - started, 3),
        requests_per_second=requests_per_second,
        total_requests=total_requests,
        non_success_responses=non_success,
        application_failures=application_failures,
        probe_http_status=probe_http_status,
        probe_body=probe_body[:4000],
        required_service=required_service,
        required_ready_endpoints=required_ready_endpoints,
        socket_errors=socket_errors,
        output=output,
        execution_started=execution_started,
        requested_rate=int(rate), requested_duration_seconds=int(duration_seconds),
        payload_sha256=hashlib.sha256(script.encode()).hexdigest(),
    )
