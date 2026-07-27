from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data_loader import iter_scenarios
from .ground_truth import ground_truth_summary, labels_from_full_state
from .llm_rca_solver import compact_state_for_llm
from .rca_reward import normalize_service_name
from .schemas import FaultLabel, normalize_fault_type
from .split_utils import read_scenario_ids


def _label_key(label: FaultLabel) -> str:
    return f"{normalize_service_name(label.service)}::{normalize_fault_type(label.fault_type or label.fault_family)}"


def _candidate_key(row: dict[str, Any]) -> str:
    return f"{normalize_service_name(row.get('service'))}::{normalize_fault_type(row.get('fault_type'))}"


def _candidate_service(row: dict[str, Any]) -> str:
    return normalize_service_name(row.get("service"))


def _candidate_fault(row: dict[str, Any]) -> str:
    return normalize_fault_type(row.get("fault_type"))


def _rank_first(values: list[str], target: str) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value == target:
            return idx
    return None


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank == 1:
        return "top1"
    if rank <= 3:
        return "top3"
    if rank <= 5:
        return "top5"
    if rank <= 10:
        return "top10"
    if rank <= 20:
        return "top20"
    return "rank_gt20"


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _candidate_excerpt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "service": row.get("service"),
        "fault_type": normalize_fault_type(row.get("fault_type")),
        "score": row.get("score"),
        "reasons": _safe_list(row.get("reasons"))[:8],
    }


def audit_one(rec: Any, top_k: int = 20) -> dict[str, Any]:
    labels = labels_from_full_state(rec.full_state)
    compact = compact_state_for_llm(rec.compressed_state)
    evidence = compact.get("high_signal_evidence", {}) if isinstance(compact, dict) else {}
    valid_services = [normalize_service_name(x) for x in _safe_list(compact.get("valid_services", []))]
    valid_service_set = set(valid_services)

    candidates_raw = _safe_list(evidence.get("candidate_root_causes", [])) if isinstance(evidence, dict) else []
    candidates = [row for row in candidates_raw if isinstance(row, dict)]
    candidate_keys = [_candidate_key(row) for row in candidates]
    candidate_services = [_candidate_service(row) for row in candidates]
    candidate_faults = [_candidate_fault(row) for row in candidates]

    gt_rows = []
    exact_ranks: list[int] = []
    service_ranks: list[int] = []
    fault_ranks: list[int] = []

    for label in labels:
        gt_service = normalize_service_name(label.service)
        gt_fault = normalize_fault_type(label.fault_type or label.fault_family)
        gt_key = f"{gt_service}::{gt_fault}"
        exact_rank = _rank_first(candidate_keys, gt_key)
        service_rank = _rank_first(candidate_services, gt_service)
        fault_rank = _rank_first(candidate_faults, gt_fault)
        if exact_rank is not None:
            exact_ranks.append(exact_rank)
        if service_rank is not None:
            service_ranks.append(service_rank)
        if fault_rank is not None:
            fault_ranks.append(fault_rank)

        matching_service_candidates = [
            _candidate_excerpt(row)
            for row in candidates
            if _candidate_service(row) == gt_service
        ][:10]
        matching_fault_candidates = [
            _candidate_excerpt(row)
            for row in candidates
            if _candidate_fault(row) == gt_fault
        ][:10]

        gt_rows.append({
            "service": label.service,
            "service_norm": gt_service,
            "fault_type": gt_fault,
            "fault_family": label.fault_family,
            "key": gt_key,
            "service_in_valid_services": gt_service in valid_service_set,
            "service_in_candidates": service_rank is not None,
            "exact_key_in_candidates": exact_rank is not None,
            "exact_rank": exact_rank,
            "service_rank": service_rank,
            "fault_type_first_rank": fault_rank,
            "rank_bucket": _rank_bucket(exact_rank),
            "matching_service_candidates": matching_service_candidates,
            "matching_fault_candidates": matching_fault_candidates,
        })

    all_service_in_valid = all(row["service_in_valid_services"] for row in gt_rows) if gt_rows else False
    all_service_in_candidates = all(row["service_in_candidates"] for row in gt_rows) if gt_rows else False
    all_exact_in_candidates = all(row["exact_key_in_candidates"] for row in gt_rows) if gt_rows else False
    exact_top1 = bool(gt_rows) and all((row["exact_rank"] == 1) for row in gt_rows)
    exact_top3 = bool(gt_rows) and all((row["exact_rank"] is not None and row["exact_rank"] <= 3) for row in gt_rows)
    exact_top5 = bool(gt_rows) and all((row["exact_rank"] is not None and row["exact_rank"] <= 5) for row in gt_rows)
    exact_top10 = bool(gt_rows) and all((row["exact_rank"] is not None and row["exact_rank"] <= 10) for row in gt_rows)

    if all_exact_in_candidates:
        recoverability = "exact_candidate_available"
    elif all_service_in_candidates:
        recoverability = "service_available_fault_type_missing_or_low_rank"
    elif all_service_in_valid:
        recoverability = "service_valid_but_not_ranked"
    else:
        recoverability = "gt_service_missing_from_valid_services"

    return {
        "scenario_id": rec.scenario_id,
        "num_gt_labels": len(labels),
        "ground_truth": [row for row in gt_rows],
        "ground_truth_summary": ground_truth_summary(rec.full_state),
        "recoverability": recoverability,
        "all_gt_services_in_valid_services": all_service_in_valid,
        "all_gt_services_in_candidates": all_service_in_candidates,
        "all_exact_gt_keys_in_candidates": all_exact_in_candidates,
        "all_exact_gt_keys_top1": exact_top1,
        "all_exact_gt_keys_top3": exact_top3,
        "all_exact_gt_keys_top5": exact_top5,
        "all_exact_gt_keys_top10": exact_top10,
        "min_exact_rank": min(exact_ranks) if exact_ranks else None,
        "min_service_rank": min(service_ranks) if service_ranks else None,
        "min_fault_type_rank": min(fault_ranks) if fault_ranks else None,
        "candidate_count": len(candidates),
        "valid_service_count": len(valid_services),
        "valid_services_sample": valid_services[:50],
        "top_candidates": [_candidate_excerpt(row) for row in candidates[:top_k]],
        "service_mention_counts": _safe_list(evidence.get("service_mention_counts", []))[:top_k] if isinstance(evidence, dict) else [],
        "signal_by_service": _safe_list(evidence.get("signal_by_service", []))[:top_k] if isinstance(evidence, dict) else [],
        "direct_health_services": _safe_list(evidence.get("direct_health_services", []))[:top_k] if isinstance(evidence, dict) else [],
        "global_signal_flags": evidence.get("global_signal_flags") if isinstance(evidence, dict) else None,
        "evidence_extractor_version": evidence.get("extractor_version") if isinstance(evidence, dict) else None,
        "builder_source": evidence.get("builder_source") if isinstance(evidence, dict) else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    total_labels = sum(int(row.get("num_gt_labels", 0)) for row in rows)
    label_counts = Counter()
    scenario_counts = Counter()
    rank_buckets = Counter()
    recoverability = Counter(row.get("recoverability", "unknown") for row in rows)

    for row in rows:
        scenario_counts["all_services_in_valid"] += int(bool(row.get("all_gt_services_in_valid_services")))
        scenario_counts["all_services_in_candidates"] += int(bool(row.get("all_gt_services_in_candidates")))
        scenario_counts["all_exact_in_candidates"] += int(bool(row.get("all_exact_gt_keys_in_candidates")))
        scenario_counts["all_exact_top1"] += int(bool(row.get("all_exact_gt_keys_top1")))
        scenario_counts["all_exact_top3"] += int(bool(row.get("all_exact_gt_keys_top3")))
        scenario_counts["all_exact_top5"] += int(bool(row.get("all_exact_gt_keys_top5")))
        scenario_counts["all_exact_top10"] += int(bool(row.get("all_exact_gt_keys_top10")))
        for gt in row.get("ground_truth", []) or []:
            label_counts["service_in_valid"] += int(bool(gt.get("service_in_valid_services")))
            label_counts["service_in_candidates"] += int(bool(gt.get("service_in_candidates")))
            label_counts["exact_in_candidates"] += int(bool(gt.get("exact_key_in_candidates")))
            rank_buckets[str(gt.get("rank_bucket", "unknown"))] += 1

    def rate(count: int, denom: int) -> float:
        return round(count / max(denom, 1), 6)

    missing_examples = []
    low_rank_examples = []
    for row in rows:
        missing = [gt for gt in row.get("ground_truth", []) or [] if not gt.get("exact_key_in_candidates")]
        low_rank = [gt for gt in row.get("ground_truth", []) or [] if gt.get("exact_rank") is not None and gt.get("exact_rank") > 10]
        if missing and len(missing_examples) < 20:
            missing_examples.append({
                "scenario_id": row.get("scenario_id"),
                "recoverability": row.get("recoverability"),
                "missing_gt": [{"key": gt.get("key"), "service_in_valid": gt.get("service_in_valid_services"), "service_rank": gt.get("service_rank")} for gt in missing],
                "top_candidates": row.get("top_candidates", [])[:8],
            })
        if low_rank and len(low_rank_examples) < 20:
            low_rank_examples.append({
                "scenario_id": row.get("scenario_id"),
                "low_rank_gt": [{"key": gt.get("key"), "exact_rank": gt.get("exact_rank")} for gt in low_rank],
                "top_candidates": row.get("top_candidates", [])[:8],
            })

    return {
        "total_scenarios": total,
        "total_gt_labels": total_labels,
        "scenario_counts": dict(scenario_counts),
        "scenario_rates": {k: rate(v, total) for k, v in scenario_counts.items()},
        "label_counts": dict(label_counts),
        "label_rates": {k: rate(v, total_labels) for k, v in label_counts.items()},
        "recoverability_counts": dict(recoverability),
        "recoverability_rates": {k: rate(v, total) for k, v in recoverability.items()},
        "exact_rank_bucket_counts": dict(rank_buckets),
        "exact_rank_bucket_rates": {k: rate(v, total_labels) for k, v in rank_buckets.items()},
        "missing_exact_examples": missing_examples,
        "low_rank_exact_examples": low_rank_examples,
        "interpretation": {
            "service_in_valid": "The oracle root service exists in the redacted agent-visible service universe.",
            "service_in_candidates": "The evidence extractor can rank the oracle root service somewhere as a possible root.",
            "exact_in_candidates": "The evidence extractor ranks the exact oracle service::fault_type pair somewhere.",
            "topk_rates": "These are recoverability upper bounds for any solver that only consumes the current evidence candidates.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit whether redacted RCA evidence contains/ranks the oracle root cause. Offline evaluation only.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--scenario_ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    allowed = read_scenario_ids(args.scenario_ids)
    rows: list[dict[str, Any]] = []
    skipped_filter = skipped_unlabeled = 0

    for rec in iter_scenarios(args.processed_states):
        if allowed is not None and rec.scenario_id not in allowed:
            skipped_filter += 1
            continue
        if not labels_from_full_state(rec.full_state):
            skipped_unlabeled += 1
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break
        rows.append(audit_one(rec, top_k=args.top_k))

    out = {
        "audit_name": "rca_evidence_recoverability_audit",
        "oracle_usage": "offline_eval_only_full_state_fault_context; not agent-visible",
        "processed_states": str(Path(args.processed_states).expanduser()),
        "scenario_ids_file": args.scenario_ids,
        "limit": args.limit,
        "skipped_filter": skipped_filter,
        "skipped_unlabeled": skipped_unlabeled,
        "summary": summarize(rows),
        "rows": rows,
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "total_scenarios": len(rows), "total_gt_labels": out["summary"]["total_gt_labels"]}, indent=2))


if __name__ == "__main__":
    main()
