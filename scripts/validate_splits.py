#!/usr/bin/env python3
"""Validate coverage, leakage boundaries, holdout semantics, and generated hashes."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_IDS = ("scale_ood", "task_ood_full", "task_ood_low_resource", "joint_ood")
ALLOWED_LIMITS = {60, 120, 300, 600, 900}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict) -> tuple[str, str]:
    return row["case_id"], row["instance"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    provenance = load_json(ROOT / "data/source_provenance.json")
    replica_audit = provenance["large_replica_audit"]
    assert replica_audit["complete_five_large_task_count"] == 179
    assert replica_audit["byte_size_monotone_task_count"] == 6
    assert [row["case_id"] for row in replica_audit["incomplete_large_replica_tasks"]] == [
        "segundo2019"
    ]

    partition = load_json(ROOT / "splits/task_partition.json")
    train_tasks = set(partition["train_tasks"])
    holdout_tasks = set(partition["holdout_tasks"])
    test_tasks = set(partition["test_tasks"])
    assert len(train_tasks) == 130
    assert len(holdout_tasks) == 20
    assert len(test_tasks) == 30
    assert train_tasks.isdisjoint(test_tasks)
    assert train_tasks.isdisjoint(holdout_tasks)
    assert holdout_tasks.isdisjoint(test_tasks)
    assert {"earl2005", "ostrowski2012"} <= train_tasks

    catalog = list(csv.DictReader((ROOT / "data/task_catalog.csv").open(newline="")))
    assert len(catalog) == 180
    assert {row["case_id"] for row in catalog} == train_tasks | holdout_tasks | test_tasks
    for field in ("problem_class", "formulation_type", "application_field"):
        all_counts = Counter(row[field] for row in catalog)
        train_values = {row[field] for row in catalog if row["partition"] == "train"}
        holdout_values = {row[field] for row in catalog if row["partition"] == "holdout"}
        test_values = {row[field] for row in catalog if row["partition"] == "test"}
        assert holdout_values | test_values <= train_values, f"Eval-only value in {field}"
        assert all(
            row["partition"] == "train"
            for row in catalog
            if all_counts[row[field]] == 1
        )

    time_rows = list(csv.DictReader((ROOT / "splits/time_limits.csv").open(newline="")))
    all_keys = {key(row) for row in time_rows}
    assert len(time_rows) == len(all_keys) == 1095
    assert all(int(row["time_limit_seconds"]) in ALLOWED_LIMITS for row in time_rows)
    assert max(int(row["time_limit_seconds"]) for row in time_rows) == 900
    assert all(row["checker_accepted"] == "True" for row in time_rows)

    loaded = {split_id: load_json(ROOT / f"splits/{split_id}.json") for split_id in SPLIT_IDS}
    for split_id, payload in loaded.items():
        records = payload["records"]
        keys = [key(row) for row in records]
        assert len(keys) == len(set(keys)), f"Duplicate record in {split_id}"
        assert set(keys) <= all_keys
        assert all(row["phase"] in {"train", "holdout", "test"} for row in records)
        assert all(row["time_limit_seconds"] <= 900 for row in records)
        assert all(
            row["eval_time_limit_seconds"] == 900
            for row in records
            if row["phase"] in {"holdout", "test"}
        )
        assert all(
            row["eval_time_limit_seconds"] is None
            for row in records
            if row["phase"] == "train"
        )
        assert payload["evaluation_analysis"]["official_uniform_eval_cap_seconds"] == 900
        assert payload["evaluation_analysis"]["p90_gurobi_runtime_seconds"] > 300

    scale = loaded["scale_ood"]["records"]
    assert {key(row) for row in scale} == all_keys
    assert all(row["scale"] == "small" for row in scale if row["phase"] == "train")
    assert all(row["scale"] == "large" for row in scale if row["phase"] == "holdout")
    assert all(row["scale"] == "large" for row in scale if row["phase"] == "test")
    all_tasks = train_tasks | holdout_tasks | test_tasks
    assert {row["case_id"] for row in scale if row["phase"] == "train"} == all_tasks
    assert {row["case_id"] for row in scale if row["phase"] == "holdout"} == all_tasks
    assert {row["case_id"] for row in scale if row["phase"] == "test"} == all_tasks
    assert len([row for row in scale if row["phase"] == "holdout"]) == 180

    full = loaded["task_ood_full"]["records"]
    assert {key(row) for row in full} == all_keys
    assert {row["case_id"] for row in full if row["phase"] == "train"} == train_tasks
    assert {row["case_id"] for row in full if row["phase"] == "holdout"} == holdout_tasks
    assert {row["case_id"] for row in full if row["phase"] == "test"} == test_tasks

    low = loaded["task_ood_low_resource"]["records"]
    assert len(low) == 360
    low_by_task = defaultdict(list)
    for row in low:
        low_by_task[row["case_id"]].append(row)
    assert set(low_by_task) == train_tasks | holdout_tasks | test_tasks
    assert all(len(rows) == 2 for rows in low_by_task.values())
    assert all(Counter(row["scale"] for row in rows) == {"small": 1, "large": 1} for rows in low_by_task.values())
    assert {row["case_id"] for row in low if row["phase"] == "train"} == train_tasks
    assert {row["case_id"] for row in low if row["phase"] == "holdout"} == holdout_tasks
    assert {row["case_id"] for row in low if row["phase"] == "test"} == test_tasks

    joint = loaded["joint_ood"]["records"]
    assert all(row["case_id"] in train_tasks and row["scale"] == "small" for row in joint if row["phase"] == "train")
    assert all(row["case_id"] in holdout_tasks and row["scale"] == "large" for row in joint if row["phase"] == "holdout")
    assert all(row["case_id"] in test_tasks and row["scale"] == "large" for row in joint if row["phase"] == "test")
    assert {row["case_id"] for row in joint if row["phase"] == "train"} == train_tasks
    assert {row["case_id"] for row in joint if row["phase"] == "holdout"} == holdout_tasks
    assert {row["case_id"] for row in joint if row["phase"] == "test"} == test_tasks

    evaluation = load_json(ROOT / "splits/evaluation_protocol.json")
    assert evaluation["uniform_eval_limit_seconds_per_instance"] == 900
    assert evaluation["primary_attempts_per_instance"] == 1
    assert {row["split_id"] for row in evaluation["split_analysis"]} == set(SPLIT_IDS)

    expected_hashes = {}
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected_hashes[relative] = digest
    for relative, expected in expected_hashes.items():
        assert sha256(ROOT / relative) == expected, f"Hash mismatch: {relative}"

    print("VALIDATION: PASS")
    print("task partition: 130 train / 20 holdout / 30 test; no alias or type leakage")
    print("instance universe: 1,095 checker-accepted pairs")
    print("official evaluation limit: uniform 900 seconds per holdout/test instance")
    print("planning tiers retained: {60, 120, 300, 600, 900}")
    print("large replicas: 179 complete 5-replica tasks; 3/2 is instance holdout, not scale OOD")
    for split_id in SPLIT_IDS:
        summary = loaded[split_id]["summary"]
        print(
            f"{split_id}: train={summary['train']['instance_count']} instances, "
            f"holdout={summary['holdout']['instance_count']} instances, "
            f"test={summary['test']['instance_count']} instances"
        )


if __name__ == "__main__":
    main()
