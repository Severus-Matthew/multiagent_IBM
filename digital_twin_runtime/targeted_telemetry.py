from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .sparse_live_session import SparseLiveTwinSession
from .targeted_workload import WorkloadResult

MEASUREMENT_CONTRACT = "phase_windows_query_status_raw_spans_incident_scope_v1"


@dataclass(frozen=True)
class ObservationWindow:
    start_unix: float
    end_unix: float
    phase: str

    def __post_init__(self) -> None:
        if (not math.isfinite(self.start_unix) or not math.isfinite(self.end_unix)
                or self.end_unix <= self.start_unix or not self.phase):
            raise ValueError("a named, finite, positive observation window is required")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "duration_seconds": self.end_unix - self.start_unix}


class TelemetryCollectionError(RuntimeError):
    def __init__(self, result: "TelemetryCollectionResult") -> None:
        super().__init__("telemetry collection failed: " + json.dumps(result.errors))
        self.result = result


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kubectl", *args], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=90)


def _read(args: list[str]) -> str:
    proc = _run(args)
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl query failed: {proc.stderr.strip()}")
    return proc.stdout


@dataclass
class TelemetryCollectionResult:
    run_dir: str
    namespace: str
    selected_services: list[str]
    files_written: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    collection_mode: str = MEASUREMENT_CONTRACT
    window: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_complete(self) -> "TelemetryCollectionResult":
        if self.errors or not all(self.channels.get(k, {}).get("query_succeeded")
                                  for k in ("system", "traces", "metrics", "logs")):
            raise TelemetryCollectionError(self)
        return self


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def _jaeger_rows(namespace: str, services: list[str], *, window: ObservationWindow,
                 limit: int = 10000) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for service in services:
        path = (f"/api/v1/namespaces/{namespace}/services/http:jaeger:16686/proxy/api/traces"
                f"?service={quote(service)}&start={int(window.start_unix * 1e6)}"
                f"&end={int(window.end_unix * 1e6)}&limit={limit}")
        payload = json.loads(_read(["get", "--raw", path]))
        if payload.get("errors") or not isinstance(payload.get("data"), list):
            raise RuntimeError(f"invalid Jaeger response for {service}")
        if len(payload["data"]) >= limit:
            raise RuntimeError("Jaeger result limit reached; split the observation window")
        for trace in payload["data"]:
            trace_id = str(trace.get("traceID") or "")
            processes = trace.get("processes", {}) or {}
            for span in trace.get("spans", []) or []:
                if "startTime" not in span:
                    raise RuntimeError("Jaeger span is missing its observation timestamp")
                start = float(span["startTime"]) / 1e6
                duration = float(span["duration"])
                if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                    raise RuntimeError("invalid Jaeger timestamp or duration")
                if not window.start_unix <= start < window.end_unix:
                    continue
                process = processes.get(span.get("processID"), {}) or {}
                parent = next((str(r.get("spanID") or "") for r in span.get("references", [])
                               if r.get("refType") == "CHILD_OF"), "")
                tags = {str(x.get("key")): x.get("value") for x in span.get("tags", []) or []}
                status = tags.get("http.status_code") or tags.get("status.code") or ""
                error = str(tags.get("error", "false")).lower() in {"true", "1"}
                row = {"trace_id": trace_id, "span_id": str(span.get("spanID") or ""),
                       "parent_span": parent, "service_name": process.get("serviceName", service),
                       "operation_name": span.get("operationName", ""), "duration": span.get("duration", 0),
                       "response": status, "has_error": str(error or str(status).startswith(("4", "5"))).lower(),
                       "start_time_unix": start}
                key = (trace_id, row["span_id"])
                if not all(key):
                    raise RuntimeError("Jaeger span is missing its identity")
                if key in unique and unique[key] != row:
                    raise RuntimeError("conflicting observations of one Jaeger span")
                unique[key] = row
    return list(unique.values())


def _prometheus_rows(namespace: str, *, window: ObservationWindow) -> list[dict[str, Any]]:
    # Range functions use only actual scrapes inside this phase. The evaluation
    # timestamp is pinned to phase end, not to the later collection time.
    duration_ms = max(1, int((window.end_unix - window.start_unix) * 1000))
    selector = f'namespace="{namespace}",container!="",container!="POD"'
    queries = {
        "container_cpu_usage_cores": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{{selector}}}[{duration_ms}ms]))',
        "container_memory_working_set_bytes": f'sum by (pod) (avg_over_time(container_memory_working_set_bytes{{{selector}}}[{duration_ms}ms]))',
    }
    rows: list[dict[str, Any]] = []
    for metric_name, query in queries.items():
        path = ("/api/v1/namespaces/observe/services/http:prometheus-server:80/proxy"
                f"/api/v1/query?query={quote(query, safe='')}&time={window.end_unix}")
        payload = json.loads(_read(["get", "--raw", path]))
        if payload.get("status") != "success" or payload.get("warnings"):
            raise RuntimeError(f"incomplete Prometheus response for {metric_name}")
        data = payload.get("data", {})
        if data.get("resultType") != "vector" or not isinstance(data.get("result"), list):
            raise RuntimeError("invalid Prometheus vector response")
        for result in data["result"]:
            value = result.get("value", [])
            if len(value) != 2 or not math.isfinite(float(value[1])):
                raise RuntimeError("non-finite or missing metric sample")
            rows.append({"timestamp": value[0], "cmdb_id": result.get("metric", {}).get("pod", "unknown"),
                         "kpi_name": metric_name, "value": float(value[1])})
    return rows


def _phase_logs(text: str, window: ObservationWindow) -> str:
    lines = []
    for line in text.splitlines():
        stamp, sep, body = line.partition(" ")
        if not sep:
            raise RuntimeError("pod log line lacks a timestamp")
        # datetime handles the microsecond precision required for phase bounds.
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        if window.start_unix <= when < window.end_unix:
            lines.append(body)
    return "\n".join(lines) + ("\n" if lines else "")


def _pod_inventory(pods, selected):
    return {p["metadata"]["name"]: {
                "uid": p["metadata"].get("uid"),
                "containers": sorted(c.get("containerID", "") for c in p.get("status", {}).get("containerStatuses", []))}
            for p in pods if p.get("status", {}).get("phase") == "Running"
            and any(p.get("metadata", {}).get("name", "").startswith(s + "-")
                    or p.get("metadata", {}).get("labels", {}).get("service") == s
                    or p.get("metadata", {}).get("labels", {}).get("app") == s for s in selected)}


def application_services(session):
    """Separate instrumented application controllers from telemetry backends."""
    objects = session.bundle.objects
    selectors = [o.get("spec", {}).get("selector", {}) for o in objects if o.get("kind") == "Service"
                 and any(p.get("port") in {4317, 4318, 14268, 16686} for p in o.get("spec", {}).get("ports", []))]
    observers = set()
    for obj in objects:
        if obj.get("kind") not in {"Deployment", "StatefulSet"}:
            continue
        labels = obj.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
        if any(selector and all(labels.get(k) == v for k, v in selector.items()) for selector in selectors):
            observers.add(obj["metadata"]["name"])
    return sorted({r["name"] for r in session.bundle.object_refs
                   if r.get("kind") in {"Deployment", "StatefulSet"}} - observers)


def capture_pod_inventory(session):
    selected = application_services(session)
    pods = json.loads(_read(["get", "pods", "-n", session.namespace, "-o", "json"]))["items"]
    return _pod_inventory(pods, selected)


def collect_targeted_telemetry(session: SparseLiveTwinSession, run_dir: str | Path, *,
                               window: ObservationWindow,
                               workload: WorkloadResult | None = None,
                               initial_pod_inventory: dict[str, Any] | None = None) -> TelemetryCollectionResult:
    started = time.monotonic()
    root = Path(run_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("telemetry capture directory must be new; refusing stale phase artifacts")
    direct = root / "direct_k8s_outputs"
    selected = application_services(session)
    result = TelemetryCollectionResult(str(root), session.namespace, selected, window=window.to_dict())
    root.mkdir(parents=True, exist_ok=True)
    tasks = {f"{kind}.json": (lambda kind=kind: _read(["get", kind, "-n", session.namespace, "-o", "json"]))
             for kind in ("pods", "deployments", "statefulsets", "services", "replicasets", "endpoints", "events")}
    metrics: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        futures[pool.submit(_prometheus_rows, session.namespace, window=window)] = "metrics.csv"
        futures[pool.submit(_jaeger_rows, session.namespace, selected, window=window)] = "traces.csv"
        for future in as_completed(futures):
            name = futures[future]
            channel = "system" if name.endswith(".json") else name.split(".")[0]
            try:
                value = future.result()
                if name.endswith(".json"):
                    json.loads(value)
                    result.files_written.append(_write(direct / name, value))
                else:
                    rows = value
                    if name == "metrics.csv":
                        metrics = rows
                    out = root / "builtin_api_outputs" / channel / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fields = list(rows[0]) if rows else (
                        ["timestamp", "cmdb_id", "kpi_name", "value"] if channel == "metrics" else
                        ["trace_id", "span_id", "parent_span", "service_name", "operation_name", "duration", "response", "has_error", "start_time_unix"])
                    with out.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader(); writer.writerows(rows)
                    result.files_written.append(str(out))
                    result.channels[channel] = {"query_succeeded": True, "row_count": len(rows)}
            except Exception as exc:
                result.errors.append({"channel": channel, "query": name, "error": f"{type(exc).__name__}: {exc}"})
    result.channels["system"] = {"query_succeeded": not any(e["channel"] == "system" for e in result.errors)}
    pods_path = direct / "pods.json"
    pods = json.loads(pods_path.read_text()).get("items", []) if pods_path.exists() else []
    log_count = 0
    running_pods: set[str] = set()
    for pod in pods:
        meta = pod.get("metadata", {}) or {}
        name = str(meta.get("name") or "")
        labels = meta.get("labels", {}) or {}
        if not any(labels.get(k) in selected for k in ("service", "app", "app.kubernetes.io/name")):
            # Workload/observer pods are not application service telemetry.
            if not any(name == s or name.startswith(s + "-") for s in selected):
                continue
        statuses = (pod.get("status") or {}).get("containerStatuses", [])
        if not any((c.get("state") or {}).get("running") for c in statuses):
            continue  # no running container is an observed structural result
        running_pods.add(name)
        try:
            since = datetime.fromtimestamp(window.start_unix, timezone.utc).isoformat().replace("+00:00", "Z")
            logs = _read(["logs", name, "-n", session.namespace, "--all-containers=true", "--timestamps=true", f"--since-time={since}"])
            result.files_written.append(_write(direct / "pod_logs" / f"{name}.log", _phase_logs(logs, window)))
            log_count += 1
        except Exception as exc:
            result.errors.append({"channel": "logs", "error": f"{name}: {exc}"})
    result.channels["logs"] = {"query_succeeded": not any(e["channel"] == "logs" for e in result.errors), "pod_count": log_count}
    for metric in ("container_cpu_usage_cores", "container_memory_working_set_bytes"):
        observed = {row["cmdb_id"] for row in metrics if row["kpi_name"] == metric}
        missing = sorted(running_pods - observed)
        if missing:
            result.errors.append({"channel": "metrics", "error": f"{metric} missing running pods: {missing}"})
            result.channels["metrics"] = {"query_succeeded": False}
    cpu = sum(r["value"] for r in metrics if r["kpi_name"] == "container_cpu_usage_cores" and r["cmdb_id"] in running_pods)
    memory = sum(r["value"] for r in metrics if r["kpi_name"] == "container_memory_working_set_bytes" and r["cmdb_id"] in running_pods)
    final_inventory = _pod_inventory(pods, selected)
    metric_pods = {r["cmdb_id"] for r in metrics if any(r["cmdb_id"].startswith(s + "-") for s in selected)}
    stable_population = initial_pod_inventory is not None and initial_pod_inventory == final_inventory and metric_pods == running_pods
    result.resources = {"application_running_pods": len(running_pods), "application_cpu_cores_mean": cpu,
                        "application_cpu_core_seconds": cpu * (window.end_unix - window.start_unix),
                        "application_memory_bytes_mean": memory, "measurement_window_seconds": window.end_unix - window.start_unix,
                        "includes_observer_overhead": False,
                        "estimator": "Prometheus reset-aware rate and per-container window mean",
                        "stable_pod_population": stable_population,
                        "valid": not result.errors and stable_population,
                        "invalid_reason": None if stable_population else "pod_population_unverified_or_changed"}
    if workload:
        result.files_written.append(_write(root / "workload_result.json", json.dumps(workload.to_dict(), indent=2)))
        result.files_written.append(_write(root / "builtin_api_outputs" / "shell" / "targeted_workload.txt", workload.output))
    result.elapsed_seconds = round(time.monotonic() - started, 3)
    _write(root / "collection_metadata.json", json.dumps(result.to_dict(), indent=2))
    return result.require_complete()
