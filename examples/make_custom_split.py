#!/usr/bin/env python3
"""Create a deterministic train/test holdout across large-instance replicas.

This measures held-out-instance generalization, not strict scale extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "splits/time_limits.csv")
    parser.add_argument("--output", type=Path, default=Path("my_large_replica_split.csv"))
    parser.add_argument("--train-count", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--incomplete-policy",
        choices=("proportional", "skip", "error"),
        default="proportional",
        help="How to handle tasks with fewer large replicas than train-count + test-count.",
    )
    return parser.parse_args()


def rank(seed: int, row: dict) -> str:
    material = f"{seed}:{row['case_id']}:{row['instance']}".encode()
    return hashlib.sha256(material).hexdigest()


def main() -> None:
    args = parse_args()
    if args.train_count < 1 or args.test_count < 1:
        raise SystemExit("train-count and test-count must both be positive")

    with args.input.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        if row["scale"] == "large":
            by_task[row["case_id"]].append(row)

    requested = args.train_count + args.test_count
    output_rows = []
    exceptions = []
    skipped = []
    for case_id in sorted(by_task):
        rows = sorted(by_task[case_id], key=lambda row: rank(args.seed, row))
        if len(rows) < requested:
            if args.incomplete_policy == "error":
                raise SystemExit(
                    f"{case_id} has {len(rows)} large replicas; {requested} are required"
                )
            if args.incomplete_policy == "skip":
                skipped.append(case_id)
                continue
            train_count = max(
                1,
                min(
                    len(rows) - 1,
                    math.floor(len(rows) * args.train_count / requested + 0.5),
                ),
            )
            test_count = len(rows) - train_count
            exceptions.append(
                {
                    "case_id": case_id,
                    "available": len(rows),
                    "train": train_count,
                    "test": test_count,
                }
            )
        else:
            train_count = args.train_count
            test_count = args.test_count
            rows = rows[:requested]

        for phase, selected in (
            ("train", rows[:train_count]),
            ("test", rows[train_count : train_count + test_count]),
        ):
            for row in selected:
                output_rows.append(
                    {
                        "phase": phase,
                        "case_id": row["case_id"],
                        "instance": row["instance"],
                        "solution": row["solution"],
                        "scale": row["scale"],
                        "gurobi_runtime_seconds": row["gurobi_runtime_seconds"],
                        "gurobi_runtime_status": row["gurobi_runtime_status"],
                        "gurobi_runtime_source": row["gurobi_runtime_source"],
                        "time_limit_seconds": row["time_limit_seconds"],
                        "checker_accepted": row["checker_accepted"],
                        "selection_seed": args.seed,
                    }
                )

    fields = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    phase_counts = {
        phase: sum(row["phase"] == phase for row in output_rows)
        for phase in ("train", "test")
    }
    print(f"Wrote {args.output}")
    print(f"Tasks represented: {len({row['case_id'] for row in output_rows})}")
    print(f"Rows: {phase_counts['train']} train / {phase_counts['test']} test")
    if exceptions:
        print("Proportional exceptions:", exceptions)
    if skipped:
        print("Skipped incomplete tasks:", skipped)
    print("Interpretation: large-replica holdout, not strict scale OOD")


if __name__ == "__main__":
    main()
