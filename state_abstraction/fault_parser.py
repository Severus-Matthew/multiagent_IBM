import re
from utils import read_json, read_text


def _variant_name(variant):
    if isinstance(variant, dict):
        return variant.get("variant", "default")
    if variant:
        return str(variant)
    return "default"


def _variant_params(variant):
    if not isinstance(variant, dict):
        return {}
    return {k: v for k, v in variant.items() if k != "variant"}


def _target_service(spec):
    return (
        spec.get("faulty_service")
        or spec.get("fault_service")
        or spec.get("service")
        or spec.get("app_name")
    )


def _fault_instance(spec, index=0):
    variant = spec.get("variant") or {"variant": "default"}
    return {
        "index": index,
        "problem_id": spec.get("problem_id"),
        "fault_family": spec.get("fault_family"),
        "task": spec.get("task"),
        "faulty_service": _target_service(spec),
        "app": spec.get("app"),
        "mode": spec.get("mode"),
        "deployment": spec.get("deployment"),
        "class_name": spec.get("class_name"),
        "folder_name": spec.get("folder_name"),
        "py_file_name": spec.get("py_file_name"),
        "variant": variant,
        "variant_name": _variant_name(variant),
        "variant_params": _variant_params(variant),
    }


def _fault_instances(spec):
    if spec.get("is_multifault") or spec.get("mode") == "multifault":
        return [
            _fault_instance(sub_spec, i)
            for i, sub_spec in enumerate(spec.get("subproblems", []) or [])
        ]
    return [_fault_instance(spec, 0)]


def parse_fault_context(run_dir):
    from pathlib import Path
    run_dir = Path(run_dir)
    spec = read_json(run_dir / "spec.json", {}) or {}
    desc = read_text(run_dir / "problem_desc.txt", "")
    instances = _fault_instances(spec)
    expected_faulty_services = sorted(
        {
            inst.get("faulty_service")
            for inst in instances
            if inst.get("faulty_service")
        }
    )
    primary_fault = instances[0] if instances else {}
    out = {
        "problem_id": spec.get("problem_id") or run_dir.name,
        "fault_family": spec.get("fault_family"),
        "task": spec.get("task"),
        "faulty_service": primary_fault.get("faulty_service"),
        "expected_faulty_services": expected_faulty_services,
        "is_multifault": bool(spec.get("is_multifault") or spec.get("mode") == "multifault"),
        "fault_instances": instances,
        "primary_fault": primary_fault,
        "variant": spec.get("variant") or primary_fault.get("variant") or {"variant": "default"},
        "variant_name": _variant_name(spec.get("variant") or primary_fault.get("variant")),
        "variant_params": _variant_params(spec.get("variant") or primary_fault.get("variant")),
        "mode": spec.get("mode"),
        "deployment": spec.get("deployment"),
        "app": spec.get("app"),
        "app_name": spec.get("app_name"),
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
