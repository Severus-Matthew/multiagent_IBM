import json
from collections import defaultdict
from pathlib import Path

from config import SERVICES
from utils import safe_float


def _service_from_metric_labels(metric_labels):
    pod = metric_labels.get("pod", "")
    container = metric_labels.get("container", "")
    workload = metric_labels.get("workload", "")

    for svc in SERVICES:
        if svc in pod:
            return svc
        if svc in container:
            return svc
        if svc in workload:
            return svc
    return None


def _extract_last_value(entry):
    if "value" in entry and isinstance(entry["value"], list) and len(entry["value"]) >= 2:
        return safe_float(entry["value"][1], 0.0)

    if "values" in entry and isinstance(entry["values"], list) and entry["values"]:
        last = entry["values"][-1]
        if isinstance(last, list) and len(last) >= 2:
            return safe_float(last[1], 0.0)

    return 0.0


def _parse_prom_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []

    return data.get("data", {}).get("result", [])


def parse_metrics_snapshot(files_by_type):
    """
    files_by_type keys:
      cpu, memory, network_rx, network_tx, restarts
    """
    service_metrics = defaultdict(lambda: {
        "cpu": 0.0,
        "memory": 0.0,
        "latency": 0.0,      # filled later from traces
        "network_rx": 0.0,
        "network_tx": 0.0,
        "restarts": 0.0,
    })

    name_map = {
        "cpu": "cpu",
        "memory": "memory",
        "network_rx": "network_rx",
        "network_tx": "network_tx",
        "restarts": "restarts",
    }

    for metric_type, feature_name in name_map.items():
        path = files_by_type.get(metric_type)
        if not path or not path.exists():
            continue

        results = _parse_prom_json(path)
        for entry in results:
            labels = entry.get("metric", {})
            svc = _service_from_metric_labels(labels)
            if not svc:
                continue
            val = _extract_last_value(entry)
            service_metrics[svc][feature_name] += val

    for svc in SERVICES:
        _ = service_metrics[svc]

    return dict(service_metrics)