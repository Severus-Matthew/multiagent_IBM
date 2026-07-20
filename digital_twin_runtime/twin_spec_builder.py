from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any
from training_pipeline.schemas import FaultLabel

@dataclass
class TwinSpec:
    scenario_id: str
    namespace: str | None
    mode: str
    services_to_keep: list[str]
    services_to_prune: list[str]
    target_faults: list[dict[str, Any]]
    reason: dict[str, list[str]] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]: return asdict(self)


def _neighbors(state: dict[str, Any], service: str) -> set[str]:
    keep = {service}
    for edge in (state.get("graph", {}) or {}).get("edges", []) or []:
        src, dst = edge.get("src"), edge.get("dst")
        if src == service and dst: keep.add(dst)
        if dst == service and src: keep.add(src)
    return keep


def build_predicted_twin_spec(compressed_state: dict[str, Any], predicted_faults: list[FaultLabel]) -> TwinSpec:
    """RCA-predicted twin spec: uses only redacted state + RCA prediction."""
    all_services = set(compressed_state.get("services", []) or [])
    keep: set[str] = set(); reason: dict[str, list[str]] = {}
    for fault in predicted_faults:
        keep.add(fault.service); reason.setdefault(fault.service, []).append("rca_predicted_root_cause")
        for n in _neighbors(compressed_state, fault.service):
            keep.add(n); reason.setdefault(n, []).append(f"neighbor_of_{fault.service}")
    for svc, h in (compressed_state.get("service_health", {}) or {}).items():
        if isinstance(h, dict) and h.get("status", "healthy") != "healthy":
            keep.add(svc); reason.setdefault(svc, []).append("redacted_service_health_degraded")
    for svc, info in (compressed_state.get("system", {}) or {}).items():
        health = info.get("health", {}) if isinstance(info, dict) else {}
        if health.get("infra_issue_flag") or health.get("pods_unready", 0) > 0:
            keep.add(svc); reason.setdefault(svc, []).append("redacted_system_infra_signal")
    if not keep:
        keep = set(all_services)
        for svc in keep: reason.setdefault(svc, []).append("fallback_keep_all_no_signal")
    return TwinSpec(compressed_state.get("scenario_id", "unknown"), compressed_state.get("namespace"),
                    "rca_predicted_redacted", sorted(keep & all_services), sorted(all_services - keep),
                    [x.to_dict() for x in predicted_faults], reason)


def build_oracle_twin_spec(full_state: dict[str, Any], gt_faults: list[FaultLabel]) -> TwinSpec:
    """Oracle spec for offline evaluation only; agents must never see it."""
    all_services = set(full_state.get("services", []) or [])
    keep = {f.service for f in gt_faults if f.service}; reason = {s: ["oracle_ground_truth_fault"] for s in keep}
    for svc in list(keep):
        for n in _neighbors(full_state, svc):
            keep.add(n); reason.setdefault(n, []).append(f"oracle_neighbor_of_{svc}")
    ns = (full_state.get("fault_context", {}) or {}).get("target_namespace")
    return TwinSpec(full_state.get("scenario_id", "unknown"), ns, "oracle_offline",
                    sorted(keep & all_services), sorted(all_services - keep),
                    [x.to_dict() for x in gt_faults], reason)
