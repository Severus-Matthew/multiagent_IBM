from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

from .data_loader import iter_scenarios
from .ground_truth import labels_from_full_state
from .split_utils import write_scenario_ids


def _label_signature(labels: list[Any]) -> tuple[tuple[str, str, str, str], ...]:
    """Keep task variants of the same incident out of opposite split sides."""
    return tuple(sorted(
        (
            str(label.service),
            str(label.fault_type),
            str(label.fault_mechanism),
            "all_parameter_variants",
        )
        for label in labels
    ))


def _strata(labels: list[Any]) -> tuple[str, ...]:
    cardinality = "multi_fault" if len(labels) > 1 else "single_fault"
    mechanisms = sorted({str(label.fault_mechanism) for label in labels})
    return tuple([cardinality] + [f"mechanism:{item}" for item in mechanisms])


def _choose_grouped_holdout(groups: list[dict[str, Any]], size: int, seed: int) -> set[int]:
    """Find an exact-size, approximately stratified group-preserving holdout."""
    if size < 0 or size > sum(len(group["ids"]) for group in groups):
        raise ValueError("holdout size is outside the labeled dataset")
    if size == 0:
        return set()

    population = Counter()
    for group in groups:
        for stratum in group["strata"]:
            population[stratum] += len(group["ids"])
    total = sum(len(group["ids"]) for group in groups)
    target = {key: value * size / total for key, value in population.items()}

    best: tuple[float, set[int]] | None = None
    # Randomized exact subset-sum searches provide varied candidates; scoring
    # then chooses the candidate closest to the dataset's mechanism/cardinality mix.
    for attempt in range(2048):
        order = list(range(len(groups)))
        random.Random(seed + attempt).shuffle(order)
        reachable: dict[int, tuple[int, ...]] = {0: ()}
        for index in order:
            width = len(groups[index]["ids"])
            for count, selected in sorted(list(reachable.items()), reverse=True):
                new_count = count + width
                if new_count <= size and new_count not in reachable:
                    reachable[new_count] = selected + (index,)
        if size not in reachable:
            continue
        selected = set(reachable[size])
        observed = Counter()
        for index in selected:
            for stratum in groups[index]["strata"]:
                observed[stratum] += len(groups[index]["ids"])
        score = sum(
            ((observed[key] - wanted) / max(1.0, wanted)) ** 2
            for key, wanted in target.items()
        )
        if best is None or score < best[0]:
            best = (score, selected)
    if best is None:
        raise ValueError(f"cannot create an exact {size}-record holdout without splitting incident groups")
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Create labeled/unlabeled scenario split files from processed_states.")
    ap.add_argument("--processed_states", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_size", type=int, default=None)
    ap.add_argument("--test_size", type=int, default=None)
    # Optional third split. Tuning iteration caps, reward thresholds or checkpoint
    # selection against the test set is leakage; a validation carve-out taken from
    # the training side keeps the test set untouched.
    ap.add_argument("--val_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    labeled: list[str] = []
    unlabeled: list[str] = []
    grouped: dict[tuple[tuple[str, str, str, str], ...], dict[str, Any]] = {}
    missing_examples: list[dict] = []

    for rec in iter_scenarios(args.processed_states):
        labels = labels_from_full_state(rec.full_state)
        if labels:
            labeled.append(rec.scenario_id)
            signature = _label_signature(labels)
            row = grouped.setdefault(signature, {"ids": [], "strata": _strata(labels)})
            row["ids"].append(rec.scenario_id)
        else:
            unlabeled.append(rec.scenario_id)
            if len(missing_examples) < 20:
                fc = rec.full_state.get("fault_context", {}) or {}
                missing_examples.append({
                    "scenario_id": rec.scenario_id,
                    "fault_family": fc.get("fault_family"),
                    "task": fc.get("task"),
                    "faulty_service": fc.get("faulty_service"),
                    "expected_faulty_services": fc.get("expected_faulty_services"),
                })

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_scenario_ids(out / f"labeled_{len(labeled)}.txt", labeled)
    write_scenario_ids(out / f"unlabeled_{len(unlabeled)}.txt", unlabeled)

    train_file = None
    test_file = None
    val_file = None
    if args.train_size is not None or args.test_size is not None:
        test_size = args.test_size
        train_size = args.train_size
        val_size = int(args.val_size or 0)
        if test_size is None:
            test_size = len(labeled) - int(train_size) - val_size
        if train_size is None:
            train_size = len(labeled) - int(test_size) - val_size
        if train_size + test_size + val_size != len(labeled):
            raise ValueError(
                f"train_size + test_size + val_size must equal {len(labeled)} labeled records"
            )
        groups = list(grouped.values())
        selected = _choose_grouped_holdout(groups, test_size, args.seed)
        test_ids = sorted(item for index in selected for item in groups[index]["ids"])
        test_set = set(test_ids)

        val_ids: list[str] = []
        if val_size:
            # Draw validation from the groups the test split did not take, so an
            # incident group can never appear in two splits.
            remaining = [row for index, row in enumerate(groups) if index not in selected]
            val_selected = _choose_grouped_holdout(remaining, val_size, args.seed + 1)
            val_ids = sorted(item for index in val_selected for item in remaining[index]["ids"])
        val_set = set(val_ids)
        if test_set & val_set:
            raise AssertionError("test and validation splits overlap")

        train_ids = sorted(
            item for item in labeled if item not in test_set and item not in val_set
        )
        train_file = out / f"train_{len(train_ids)}.txt"
        test_file = out / f"test_{len(test_ids)}.txt"
        write_scenario_ids(train_file, train_ids)
        write_scenario_ids(test_file, test_ids)
        if val_ids:
            val_file = out / f"val_{len(val_ids)}.txt"
            write_scenario_ids(val_file, val_ids)

    summary = {
        "processed_states": str(Path(args.processed_states).expanduser()),
        "total": len(labeled) + len(unlabeled),
        "labeled": len(labeled),
        "unlabeled": len(unlabeled),
        "labeled_file": str(out / f"labeled_{len(labeled)}.txt"),
        "unlabeled_file": str(out / f"unlabeled_{len(unlabeled)}.txt"),
        "grouping_policy": "exact_ground_truth_fault_signature",
        "num_incident_groups": len(grouped),
        "train_file": str(train_file) if train_file else None,
        "test_file": str(test_file) if test_file else None,
        "val_file": str(val_file) if val_file else None,
        "seed": args.seed,
        "missing_examples": missing_examples,
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
