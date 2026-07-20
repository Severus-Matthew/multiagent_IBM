from __future__ import annotations

from typing import Any

DANGEROUS_PATTERNS = [
    "delete namespace", "delete ns", "delete clusterrole", "delete crd",
    "delete pvc --all", "rm -rf", "kubectl delete all --all",
    "kubectl delete pods --all --all-namespaces",
]


def check_command_safety(commands: list[str]) -> dict[str, Any]:
    unsafe = []
    for cmd in commands:
        low = str(cmd).lower()
        for pat in DANGEROUS_PATTERNS:
            if pat in low:
                unsafe.append({"command": cmd, "pattern": pat})
        if not (low.startswith("kubectl ") or low.startswith("helm ") or low.startswith("mongosh ")):
            unsafe.append({"command": cmd, "pattern": "unsupported_prefix"})
    return {"safe": not unsafe, "unsafe": unsafe, "num_commands": len(commands)}
