import json
from pathlib import Path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)