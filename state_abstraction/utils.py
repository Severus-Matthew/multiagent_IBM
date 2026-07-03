import csv
import json
import re
from pathlib import Path
from datetime import datetime
from statistics import mean as _mean

TS_RE = re.compile(r"(\d{8}_\d{6}|\d{10}(?:\.\d+)?)")
REPLICA_SUFFIX_RE = re.compile(r"-[a-f0-9]{6,12}-[a-z0-9]{4,6}$")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_text(path, default=""):
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return default


def read_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def write_jsonl(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=False) + "\n")


def append_jsonl(row, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=False) + "\n")


def read_jsonl(path):
    rows = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    return rows


def safe_float(x, default=0.0):
    try:
        if x in [None, "", "NaN", "nan"]:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def percentile(values, p):
    vals = sorted([float(v) for v in values if v is not None])
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def mean(values):
    vals = [float(v) for v in values if v is not None]
    return float(_mean(vals)) if vals else 0.0


def summarize(values):
    vals = [safe_float(v, None) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "last": 0.0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": mean(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "last": vals[-1],
    }


def extract_timestamp_from_name(name):
    m = TS_RE.search(name)
    return m.group(1) if m else None


def normalize_service_name(name):
    if not name:
        return "unknown"
    name = str(name).strip()
    for suffix in [".log", ".stderr", ".txt", ".json", ".csv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if "/" in name:
        name = name.split("/")[-1]
    if name.startswith("pod/"):
        parts = name.split("/")
        if len(parts) >= 2:
            name = parts[1]
    name = REPLICA_SUFFIX_RE.sub("", name)
    return name


def service_from_filename(path):
    return normalize_service_name(Path(path).name)


def service_from_labels(labels, known_services=None):
    labels = labels or {}
    candidates = [
        labels.get("service"), labels.get("app"), labels.get("app.kubernetes.io/name"),
        labels.get("workload"), labels.get("container"), labels.get("pod"), labels.get("pod_name"),
        labels.get("service_name"), labels.get("destination_service"), labels.get("source_service"),
    ]
    text = " ".join(str(x) for x in candidates if x)
    if known_services:
        for svc in sorted(known_services, key=len, reverse=True):
            if svc and svc in text:
                return svc
    for x in candidates:
        if x:
            return normalize_service_name(x)
    return None


def read_csv_rows(path):
    try:
        with open(path, newline="", errors="ignore") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def find_files(root, patterns):
    root = Path(root)
    out = []
    for pat in patterns:
        out.extend(root.rglob(pat))
    return sorted(set(out))
