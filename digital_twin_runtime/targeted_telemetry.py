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
PROMETHEUS_PROXY = "/api/v1/namespaces/observe/services/http:prometheus-server:80/proxy"
# ``rate()``/``avg_over_time()`` evaluate only samples inside the phase window.
# Prometheus needs at least two samples of a series for a rate, so a phase must
# cover two scrape intervals plus scheduling jitter; otherwise every pod would
# be reported as unobserved. The lookback is never widened beyond the phase.
MIN_SCRAPES_PER_PHASE = 2
PHASE_WINDOW_MARGIN_SECONDS = 5.0
_PROMETHEUS_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0, "y": 31536000.0}


def parse_prometheus_duration(text: str) -> float:
    """Seconds for a Prometheus duration such as ``15s``, ``1m``, ``1m30s`` or ``500ms``."""
    import re
    value = str(text or "").strip()
    if not re.fullmatch(r"(\d+(ms|[smhdwy]))+", value):
        raise ValueError(f"invalid Prometheus duration: {text!r}")
    return sum(float(n) * _PROMETHEUS_DURATION_UNITS[u] for n, u in re.findall(r"(\d+)(ms|[smhdwy])", value))


def scrape_interval_from_config_yaml(yaml_text: str) -> float:
    """Global ``scrape_interval`` from Prometheus' rendered configuration (default 1m)."""
    lines = str(yaml_text or "").splitlines()
    in_global = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            in_global = line.strip() == "global:"
            continue
        if in_global:
            key, sep, value = line.strip().partition(":")
            if sep and key.strip() == "scrape_interval":
                return parse_prometheus_duration(value.strip().strip("'\""))
    return 60.0


def discover_prometheus_scrape_interval() -> float:
    """Read the effective global scrape interval from the Prometheus API."""
    payload = json.loads(_read(["get", "--raw", PROMETHEUS_PROXY + "/api/v1/status/config"]))
    if payload.get("status") != "success" or not isinstance((payload.get("data") or {}).get("yaml"), str):
        raise RuntimeError("Prometheus configuration is not readable; cannot verify phase sample coverage")
    return scrape_interval_from_config_yaml(payload["data"]["yaml"])


def minimum_phase_window_seconds(scrape_interval_seconds: float) -> float:
    return MIN_SCRAPES_PER_PHASE * float(scrape_interval_seconds) + PHASE_WINDOW_MARGIN_SECONDS


def require_phase_window_covers_scrapes(window_seconds: float, scrape_interval_seconds: float) -> None:
    minimum = minimum_phase_window_seconds(scrape_interval_seconds)
    if not math.isfinite(float(window_seconds)) or float(window_seconds) < minimum:
        raise ValueError(
            f"phase window of {float(window_seconds):.1f}s cannot contain {MIN_SCRAPES_PER_PHASE} Prometheus "
            f"scrapes at a {float(scrape_interval_seconds):.0f}s scrape interval; use a workload phase of at least "
            f"{minimum:.0f}s (--twin_workload_duration_seconds) or a shorter Prometheus scrape_interval"
        )


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


def _prometheus_query(query: str, *, time_unix: float) -> list[dict[str, Any]]:
    path = f"{PROMETHEUS_PROXY}/api/v1/query?query={quote(query, safe='')}&time={time_unix}"
    payload = json.loads(_read(["get", "--raw", path]))
    if payload.get("status") != "success" or payload.get("warnings"):
        raise RuntimeError("incomplete Prometheus response")
    data = payload.get("data", {})
    if data.get("resultType") != "vector" or not isinstance(data.get("result"), list):
        raise RuntimeError("invalid Prometheus vector response")
    return data["result"]


def _prometheus_sample_coverage(namespace: str, *, window: ObservationWindow) -> dict[str, int]:
    """Samples of the CPU counter observed per pod strictly inside the phase window."""
    duration_ms = max(1, int((window.end_unix - window.start_unix) * 1000))
    selector = f'namespace="{namespace}",container!="",container!="POD"'
    query = f'max by (pod) (count_over_time(container_cpu_usage_seconds_total{{{selector}}}[{duration_ms}ms]))'
    counts: dict[str, int] = {}
    for result in _prometheus_query(query, time_unix=window.end_unix):
        value = result.get("value", [])
        if len(value) != 2 or not math.isfinite(float(value[1])):
            raise RuntimeError("non-finite or missing sample count")
        counts[str(result.get("metric", {}).get("pod", "unknown"))] = int(float(value[1]))
    return counts


def _prometheus_rows(namespace: str, *, window: ObservationWindow,
                     scrape_interval_seconds: float | None = None) -> list[dict[str, Any]]:
    # Range functions use only actual scrapes inside this phase. The evaluation
    # timestamp is pinned to phase end, not to the later collection time.
    if scrape_interval_seconds is None:
        scrape_interval_seconds = discover_prometheus_scrape_interval()
    require_phase_window_covers_scrapes(window.end_unix - window.start_unix, scrape_interval_seconds)
    duration_ms = max(1, int((window.end_unix - window.start_unix) * 1000))
    selector = f'namespace="{namespace}",container!="",container!="POD"'
    queries = {
        "container_cpu_usage_cores": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{{selector}}}[{duration_ms}ms]))',
        "container_memory_working_set_bytes": f'sum by (pod) (avg_over_time(container_memory_working_set_bytes{{{selector}}}[{duration_ms}ms]))',
    }
    rows: list[dict[str, Any]] = []
    for metric_name, query in queries.items():
        try:
            results = _prometheus_query(query, time_unix=window.end_unix)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} for {metric_name}") from exc
        for result in results:
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
                               initial_pod_inventory: dict[str, Any] | None = None,
                               scrape_interval_seconds: float | None = None) -> TelemetryCollectionResult:
    started = time.monotonic()
    root = Path(run_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("telemetry capture directory must be new; refusing stale phase artifacts")
    direct = root / "direct_k8s_outputs"
    selected = application_services(session)
    result = TelemetryCollectionResult(str(root), session.namespace, selected, window=window.to_dict())
    root.mkdir(parents=True, exist_ok=True)
    if scrape_interval_seconds is None:
        try:
            scrape_interval_seconds = discover_prometheus_scrape_interval()
        except Exception as exc:
            result.errors.append({"channel": "metrics", "query": "scrape_interval",
                                  "error": f"{type(exc).__name__}: {exc}"})
    result.window.update({"scrape_interval_seconds": scrape_interval_seconds,
                          "minimum_window_seconds": (minimum_phase_window_seconds(scrape_interval_seconds)
                                                     if scrape_interval_seconds is not None else None)})
    tasks = {f"{kind}.json": (lambda kind=kind: _read(["get", kind, "-n", session.namespace, "-o", "json"]))
             for kind in ("pods", "deployments", "statefulsets", "services", "replicasets", "endpoints", "events")}
    metrics: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        futures[pool.submit(_prometheus_rows, session.namespace, window=window,
                            scrape_interval_seconds=scrape_interval_seconds)] = "metrics.csv"
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
    # Resource accounting is measured strictly inside the phase. A running pod
    # without enough scrapes in the window (it started mid-phase, or the phase
    # is too short for the scrape cadence) invalidates the CPU/memory
    # measurement; it is not a failed observation channel for the reward.
    coverage: dict[str, Any] = {"query_succeeded": False, "min_samples_required": MIN_SCRAPES_PER_PHASE,
                                "samples_by_pod": {}, "insufficient_pods": [], "missing_pods": []}
    if not any(e["channel"] == "metrics" for e in result.errors):
        try:
            counts = _prometheus_sample_coverage(session.namespace, window=window)
            coverage.update({"query_succeeded": True,
                             "samples_by_pod": {pod: counts.get(pod, 0) for pod in sorted(running_pods)},
                             "insufficient_pods": sorted(pod for pod in running_pods
                                                         if counts.get(pod, 0) < MIN_SCRAPES_PER_PHASE)})
        except Exception as exc:
            result.errors.append({"channel": "metrics", "query": "sample_coverage",
                                  "error": f"{type(exc).__name__}: {exc}"})
    for metric in ("container_cpu_usage_cores", "container_memory_working_set_bytes"):
        observed = {row["cmdb_id"] for row in metrics if row["kpi_name"] == metric}
        coverage["missing_pods"].extend(f"{metric}:{pod}" for pod in sorted(running_pods - observed))
    metrics_failed = any(e["channel"] == "metrics" for e in result.errors)
    result.channels["metrics"] = {**(result.channels.get("metrics") or {}), "query_succeeded": not metrics_failed}
    cpu = sum(r["value"] for r in metrics if r["kpi_name"] == "container_cpu_usage_cores" and r["cmdb_id"] in running_pods)
    memory = sum(r["value"] for r in metrics if r["kpi_name"] == "container_memory_working_set_bytes" and r["cmdb_id"] in running_pods)
    final_inventory = _pod_inventory(pods, selected)
    metric_pods = {r["cmdb_id"] for r in metrics if any(r["cmdb_id"].startswith(s + "-") for s in selected)}
    stable_population = initial_pod_inventory is not None and initial_pod_inventory == final_inventory and metric_pods == running_pods
    invalid_reasons = []
    if result.errors:
        invalid_reasons.append("collection_errors")
    if not stable_population:
        invalid_reasons.append("pod_population_unverified_or_changed")
    if coverage["insufficient_pods"] or not coverage["query_succeeded"]:
        invalid_reasons.append("insufficient_metric_samples_in_phase")
    if coverage["missing_pods"]:
        invalid_reasons.append("running_pods_without_metric_series")
    result.resources = {"application_running_pods": len(running_pods), "application_cpu_cores_mean": cpu,
                        "application_cpu_core_seconds": cpu * (window.end_unix - window.start_unix),
                        "application_memory_bytes_mean": memory, "measurement_window_seconds": window.end_unix - window.start_unix,
                        "includes_observer_overhead": False,
                        "estimator": "Prometheus reset-aware rate and per-container window mean",
                        "scrape_interval_seconds": scrape_interval_seconds,
                        "metric_sample_coverage": coverage,
                        "stable_pod_population": stable_population,
                        "valid": not invalid_reasons,
                        "invalid_reason": ";".join(invalid_reasons) if invalid_reasons else None}
    if workload:
        result.files_written.append(_write(root / "workload_result.json", json.dumps(workload.to_dict(), indent=2)))
        result.files_written.append(_write(root / "builtin_api_outputs" / "shell" / "targeted_workload.txt", workload.output))
    result.elapsed_seconds = round(time.monotonic() - started, 3)
    _write(root / "collection_metadata.json", json.dumps(result.to_dict(), indent=2))
    return result.require_complete()
