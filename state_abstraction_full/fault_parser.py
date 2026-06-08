import re
from utils import read_json, read_text


def parse_fault_context(run_dir):
    from pathlib import Path
    run_dir = Path(run_dir)
    spec = read_json(run_dir / "spec.json", {}) or {}
    desc = read_text(run_dir / "problem_desc.txt", "")
    out = {
        "problem_id": spec.get("problem_id") or run_dir.name,
        "fault_family": spec.get("fault_family"),
        "task": spec.get("task"),
        "faulty_service": spec.get("faulty_service") or spec.get("fault_service") or spec.get("service"),
        "class_name": spec.get("class_name"),
        "target_namespace": spec.get("namespace") or spec.get("target_namespace"),
        "raw_spec": spec,
        "problem_description": desc[:5000],
    }
    text = " ".join(str(x) for x in [out["problem_id"], out["fault_family"], out["class_name"], desc]).lower()
    if not out["fault_family"]:
        if "mongodb" in text and "auth" in text:
            out["fault_family"] = "auth_miss_mongodb"
        elif "latency" in text:
            out["fault_family"] = "latency_degradation"
        elif "cpu" in text:
            out["fault_family"] = "resource_cpu"
        elif "memory" in text:
            out["fault_family"] = "resource_memory"
    if not out["faulty_service"]:
        m = re.search(r"(?:service|target)[_ -]?([a-z0-9-]+service)", text)
        if m:
            out["faulty_service"] = m.group(1)
    return out
