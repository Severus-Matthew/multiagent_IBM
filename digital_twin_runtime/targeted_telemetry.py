from __future__ import annotations

import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .sparse_live_session import SparseLiveTwinSession
from .targeted_workload import WorkloadResult


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )


def _read(args: list[str]) -> str:
    proc = _run(args)
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


@dataclass
class TelemetryCollectionResult:
    run_dir: str
    namespace: str
    selected_services: list[str]
    files_written: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    collection_mode: str = "targeted_sparse_live_concurrent_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def _jaeger_rows(namespace: str, services: list[str], lookback: str = "5m") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for service in services:
        path = (
            f"/api/v1/namespaces/{namespace}/services/http:jaeger:16686/proxy/api/traces"
            f"?service={quote(service)}&lookback={quote(lookback)}&limit=200"
        )
        proc = _run(["get", "--raw", path])
        if proc.returncode != 0:
            continue
        payload = json.loads(proc.stdout or "{}")
        for trace in payload.get("data", []) or []:
            trace_id = str(trace.get("traceID") or "")
            processes = trace.get("processes", {}) or {}
            for span in trace.get("spans", []) or []:
                process = processes.get(span.get("processID"), {}) or {}
                parent = ""
                for ref in span.get("references", []) or []:
                    if ref.get("refType") == "CHILD_OF":
                        parent = str(ref.get("spanID") or "")
                        break
                tags = {str(x.get("key")): x.get("value") for x in span.get("tags", []) or []}
                status = tags.get("http.status_code") or tags.get("status.code") or ""
                has_error = bool(tags.get("error")) or str(status).startswith(("4", "5"))
                rows.append({
                    "trace_id": trace_id,
                    "span_id": span.get("spanID", ""),
                    "parent_span": parent,
                    "service_name": process.get("serviceName", service),
                    "operation_name": span.get("operationName", ""),
                    "duration": span.get("duration", 0),
                    "response": status,
                    "has_error": str(has_error).lower(),
                })
    unique = {(r["trace_id"], r["span_id"]): r for r in rows}
    return list(unique.values())


def _prometheus_rows(namespace: str) -> list[dict[str, Any]]:
    queries = {
        "container_cpu_usage_seconds_total": (
            f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container!=""}}[1m]))'
        ),
        "container_memory_working_set_bytes": (
            f'sum by (pod) (container_memory_working_set_bytes{{namespace="{namespace}",container!=""}})'
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric_name, query in queries.items():
        path = (
            "/api/v1/namespaces/observe/services/http:prometheus-server:80/proxy"
            f"/api/v1/query?query={quote(query, safe='')}"
        )
        proc = _run(["get", "--raw", path])
        if proc.returncode != 0:
            continue
        payload = json.loads(proc.stdout or "{}")
        for result in (((payload.get("data") or {}).get("result")) or []):
            metric = result.get("metric", {}) or {}
            value = result.get("value", [None, 0]) or [None, 0]
            rows.append({
                "timestamp": value[0], "cmdb_id": metric.get("pod", "unknown"),
                "kpi_name": metric_name, "value": value[1],
            })
    return rows


def collect_targeted_telemetry(
    session: SparseLiveTwinSession,
    run_dir: str | Path,
    *,
    workload: WorkloadResult | None = None,
) -> TelemetryCollectionResult:
    """Collect selected K8s state, logs, metrics and traces concurrently."""
    started = time.monotonic()
    root = Path(run_dir)
    direct = root / "direct_k8s_outputs"
    selected = sorted({
        str(row.get("name")) for row in session.bundle.object_refs
        if row.get("kind") in {"Deployment", "StatefulSet"}
    })
    result = TelemetryCollectionResult(str(root), session.namespace, selected)
    root.mkdir(parents=True, exist_ok=True)

    tasks: dict[str, Any] = {
        "pods.json": lambda: _read(["get", "pods", "-n", session.namespace, "-o", "json"]),
        "deployments.json": lambda: _read(["get", "deployments", "-n", session.namespace, "-o", "json"]),
        "services.json": lambda: _read(["get", "services", "-n", session.namespace, "-o", "json"]),
        "replicasets.json": lambda: _read(["get", "replicasets", "-n", session.namespace, "-o", "json"]),
        "endpoints.json": lambda: _read(["get", "endpoints", "-n", session.namespace, "-o", "json"]),
        "events.json": lambda: _read(["get", "events", "-n", session.namespace, "-o", "json"]),
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_names = {pool.submit(fn): name for name, fn in tasks.items()}
        future_names[pool.submit(_prometheus_rows, session.namespace)] = "metrics.csv"
        future_names[pool.submit(_jaeger_rows, session.namespace, selected)] = "traces.csv"
        for future in as_completed(future_names):
            name = future_names[future]
            try:
                value = future.result()
                if name.endswith(".json"):
                    result.files_written.append(_write(direct / name, value))
                else:
                    out = root / "builtin_api_outputs" / ("metrics" if name == "metrics.csv" else "traces") / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    rows = value
                    fields = list(rows[0]) if rows else (
                        ["timestamp", "cmdb_id", "kpi_name", "value"] if name == "metrics.csv"
                        else ["trace_id", "span_id", "parent_span", "service_name", "operation_name", "duration", "response", "has_error"]
                    )
                    with out.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                    result.files_written.append(str(out))
            except Exception as exc:
                result.errors.append({"channel": name, "error": f"{type(exc).__name__}: {exc}"})

    pods = json.loads((direct / "pods.json").read_text()).get("items", []) or []
    for pod in pods:
        pod_name = str((pod.get("metadata", {}) or {}).get("name") or "")
        labels = (pod.get("metadata", {}) or {}).get("labels", {}) or {}
        service = str(labels.get("service") or labels.get("app") or pod_name)
        if service not in selected:
            continue
        proc = _run(["logs", pod_name, "-n", session.namespace, "--all-containers=true", "--tail=2000"])
        text = proc.stdout + proc.stderr
        # The shared parser already removes ReplicaSet/pod suffixes. Prefixing
        # the service a second time creates a false ``service__service`` node.
        result.files_written.append(_write(direct / "pod_logs" / f"{pod_name}.log", text))

    if workload:
        result.files_written.append(_write(root / "workload_result.json", json.dumps(workload.to_dict(), indent=2)))
        result.files_written.append(_write(root / "builtin_api_outputs" / "shell" / "targeted_workload.txt", workload.output))
    result.files_written.append(_write(root / "collection_metadata.json", json.dumps({
        "mode": result.collection_mode,
        "namespace": session.namespace,
        "selected_services": selected,
        "oracle_fields_used": False,
        "collected_at_unix": time.time(),
    }, indent=2)))
    result.elapsed_seconds = round(time.monotonic() - started, 3)
    return result
