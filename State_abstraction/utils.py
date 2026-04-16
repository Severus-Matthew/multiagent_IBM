import json
import re
from pathlib import Path
from datetime import datetime


TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def safe_read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def extract_timestamp_from_name(name: str):
    m = TIMESTAMP_RE.search(name)
    if not m:
        return None
    return m.group(1)


def parse_ts(ts: str):
    return datetime.strptime(ts, "%Y%m%d_%H%M%S")


def sort_timestamps(ts_list):
    return sorted(ts_list, key=parse_ts)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def percentile(values, p):
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)