#!/usr/bin/env python3
"""Build four reproducible FrontierOR RL train/holdout/test manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_REVISION = "a6fe77d0c79184bbea1e8f72ca6efd1a75eec1cf"
SOURCE_REPO = "LeoJiangOR/FrontierOR-Audited-180"
SEED = 20260831
TEST_TASK_COUNT = 30
HOLDOUT_TASK_COUNT = 20
UNIFIED_EVAL_LIMIT_SECONDS = 900
TIME_LIMIT_TIERS = (
    (5.0, 60),
    (30.0, 120),
    (120.0, 300),
    (300.0, 600),
    (math.inf, 900),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "FrontierOR_Audited_180_HF",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_boundaries(values: list[float], groups: int = 4) -> list[float]:
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, math.ceil(len(ordered) * i / groups) - 1)] for i in range(1, groups)]


def bin_value(value: float, boundaries: list[float]) -> str:
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return f"Q{index + 1}"
    return f"Q{len(boundaries) + 1}"


def instance_scale(instance_name: str) -> str:
    return "small" if instance_name.startswith("tiny_instance") else "large"


def large_replica_audit(dataset_root: Path) -> dict:
    complete = 0
    byte_monotone = 0
    adjacent_ratios: list[float] = []
    incomplete = []
    for case_dir in sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and (path / "instance").is_dir()
    ):
        values = []
        for path in (case_dir / "instance").glob("large_instance*.json"):
            match = re.fullmatch(r"large_instance_(\d+)\.json", path.name)
            if match is None:
                continue
            raw = int(match.group(1))
            slot = raw // 10 if raw in {11, 21, 31, 41, 51} else raw
            values.append((slot, path.stat().st_size, path.name))
        values.sort()
        if len(values) == 5:
            complete += 1
            byte_monotone += all(values[i][1] <= values[i + 1][1] for i in range(4))
            adjacent_ratios.extend(
                values[i + 1][1] / max(1, values[i][1]) for i in range(4)
            )
        else:
            incomplete.append(
                {"case_id": case_dir.name, "large_instances": [value[2] for value in values]}
            )
    return {
        "complete_five_large_task_count": complete,
        "byte_size_monotone_task_count": byte_monotone,
        "byte_size_monotone_fraction": byte_monotone / complete,
        "median_adjacent_byte_size_ratio": statistics.median(adjacent_ratios),
        "incomplete_large_replica_tasks": incomplete,
        "interpretation": (
            "large_1 through large_5 are replicate identifiers, not a reliable increasing-size axis"
        ),
    }


def time_limit(runtime_seconds: float | None, runtime_status: str) -> tuple[int, str]:
    if runtime_status == "timeout_censored" or runtime_seconds is None:
        return 900, "timeout-censored or unavailable -> 900s"
    for upper, limit in TIME_LIMIT_TIERS:
        if runtime_seconds <= upper:
            if math.isinf(upper):
                return limit, "runtime > 300s -> 900s"
            return limit, f"runtime <= {upper:g}s -> {limit}s"
    raise AssertionError("unreachable")


def load_source(dataset_root: Path) -> tuple[list[dict], list[dict], dict]:
    task_meta = json.loads((dataset_root / "paper_meta_info.json").read_text())
    runtime_payload = json.loads(
        (dataset_root / "audit/GUROBI_RUNTIME_METADATA.json").read_text()
    )
    runtime_records = runtime_payload["records"]
    metadata = {row["paper_id"]: dict(row) for row in task_meta}
    if "earl2005" not in metadata or "ostrowski2012" in metadata:
        raise ValueError("Unexpected alias metadata state")
    alias = dict(metadata["earl2005"])
    alias.update(
        {
            "paper_id": "ostrowski2012",
            "paper_title": f"{alias['paper_title']} (legacy alias)",
            "metadata_inherited_from": "earl2005",
        }
    )
    metadata["ostrowski2012"] = alias
    case_ids = sorted({row["case_id"] for row in runtime_records})
    if len(case_ids) != 180 or set(case_ids) != set(metadata):
        raise ValueError("Task metadata does not align to the 180 release case IDs")
    return [metadata[c] for c in case_ids], runtime_records, runtime_payload


def enrich_tasks(task_rows: list[dict], runtime_records: list[dict]) -> list[dict]:
    runtimes: dict[str, list[float]] = defaultdict(list)
    large_runtimes: dict[str, list[float]] = defaultdict(list)
    for row in runtime_records:
        value = row["recommended_runtime_seconds"]
        if value is None:
            value = 900.0
        runtimes[row["case_id"]].append(float(value))
        if instance_scale(row["instance"]) == "large":
            large_runtimes[row["case_id"]].append(float(value))

    median_values = [statistics.median(runtimes[row["paper_id"]]) for row in task_rows]
    size_values = [
        math.log1p(float(row["avg_num_var"]))
        + math.log1p(float(row["avg_num_constr"]))
        for row in task_rows
    ]
    runtime_bounds = quantile_boundaries(median_values)
    size_bounds = quantile_boundaries(size_values)
    enriched = []
    for row in task_rows:
        case_id = row["paper_id"]
        task_median = statistics.median(runtimes[case_id])
        large_median = statistics.median(large_runtimes[case_id])
        size_score = math.log1p(float(row["avg_num_var"])) + math.log1p(
            float(row["avg_num_constr"])
        )
        year = int(row["year"])
        if year < 2010:
            year_bin = "pre-2010"
        elif year < 2015:
            year_bin = "2010-2014"
        elif year < 2020:
            year_bin = "2015-2019"
        else:
            year_bin = "2020+"
        item = dict(row)
        item.update(
            {
                "canonical_group": "earl2005"
                if case_id in {"earl2005", "ostrowski2012"}
                else case_id,
                "task_median_runtime_seconds": task_median,
                "large_median_runtime_seconds": large_median,
                "runtime_quartile": bin_value(task_median, runtime_bounds),
                "size_quartile": bin_value(size_score, size_bounds),
                "year_bin": year_bin,
            }
        )
        enriched.append(item)
    return enriched


def feature_map(tasks: list[dict]) -> tuple[dict[str, set[str]], dict[str, float]]:
    class_counts = Counter(row["problem_class"] for row in tasks)
    weights = {
        "category": 3.0,
        "formulation": 2.5,
        "application": 2.0,
        "direction": 1.0,
        "runtime": 3.0,
        "size": 2.0,
        "year": 1.0,
        "problem": 1.5,
    }
    mapped = {}
    for row in tasks:
        tokens = {
            f"category={row['category']}",
            f"formulation={row['formulation_type']}",
            f"application={row['application_field']}",
            f"direction={row['direction']}",
            f"runtime={row['runtime_quartile']}",
            f"size={row['size_quartile']}",
            f"year={row['year_bin']}",
        }
        if class_counts[row["problem_class"]] >= 4:
            tokens.add(f"problem={row['problem_class']}")
        mapped[row["paper_id"]] = tokens
    return mapped, weights


def choose_task_partition(tasks: list[dict]) -> tuple[set[str], set[str], dict]:
    all_ids = {row["paper_id"] for row in tasks}
    singleton_fields = ("problem_class", "formulation_type", "application_field")
    singleton_values = {
        field: {
            value
            for value, count in Counter(row[field] for row in tasks).items()
            if count == 1
        }
        for field in singleton_fields
    }
    forced_train = {"earl2005", "ostrowski2012"} | {
        row["paper_id"]
        for row in tasks
        if any(row[field] in singleton_values[field] for field in singleton_fields)
    }
    candidates = sorted(all_ids - forced_train)
    features, prefix_weights = feature_map(tasks)
    totals = Counter(token for case_id in all_ids for token in features[case_id])
    ratio = TEST_TASK_COUNT / len(all_ids)
    protected_groups = {
        (field, value): {row["paper_id"] for row in tasks if row[field] == value}
        for field in singleton_fields
        for value in {row[field] for row in tasks}
    }

    def score(selection: set[str]) -> float:
        if any(group <= selection for group in protected_groups.values()):
            return math.inf
        counts = Counter(token for case_id in selection for token in features[case_id])
        total_score = 0.0
        for token, total in totals.items():
            prefix = token.split("=", 1)[0]
            target = total * ratio
            error = counts[token] - target
            total_score += prefix_weights[prefix] * error * error / max(1.0, target)
            if total >= 6 and target >= 1 and counts[token] == 0:
                total_score += 4.0 * prefix_weights[prefix]
        return total_score

    rng = random.Random(SEED)
    best_selection = None
    best_score = math.inf
    for _ in range(24):
        selected = set(rng.sample(candidates, TEST_TASK_COUNT))
        current = score(selected)
        for _round in range(100):
            best_swap = None
            best_swap_score = current
            outside = [case_id for case_id in candidates if case_id not in selected]
            for outgoing in sorted(selected):
                reduced = selected - {outgoing}
                for incoming in outside:
                    proposed = reduced | {incoming}
                    candidate_score = score(proposed)
                    if candidate_score < best_swap_score - 1e-12:
                        best_swap_score = candidate_score
                        best_swap = (outgoing, incoming)
            if best_swap is None:
                break
            selected.remove(best_swap[0])
            selected.add(best_swap[1])
            current = best_swap_score
        if current < best_score:
            best_score = current
            best_selection = set(selected)

    assert best_selection is not None
    test_ids = best_selection
    train_ids = all_ids - test_ids
    if len(train_ids) != 150 or len(test_ids) != 30 or not forced_train <= train_ids:
        raise ValueError("Task partition cardinality or alias constraint failed")

    balance = []
    test_counts = Counter(token for case_id in test_ids for token in features[case_id])
    for token, total in sorted(totals.items()):
        prefix, value = token.split("=", 1)
        balance.append(
            {
                "dimension": prefix,
                "value": value,
                "all_tasks": total,
                "train_tasks": total - test_counts[token],
                "test_tasks": test_counts[token],
                "target_test_tasks": round(total * ratio, 3),
            }
        )
    rationale = {
        "algorithm": "deterministic multi-start local search over stratification error",
        "seed": SEED,
        "objective_score": round(best_score, 12),
        "test_task_count": len(test_ids),
        "train_task_count": len(train_ids),
        "stratification_dimensions": [
            "category",
            "formulation_type",
            "application_field",
            "direction",
            "task median Gurobi runtime quartile",
            "model-size quartile",
            "publication-year bin",
            "problem_class when represented by at least four tasks",
        ],
        "leakage_constraint": (
            "earl2005 and its legacy alias ostrowski2012 are forced into train together; "
            "singleton problem classes, formulations, and application fields stay in train; "
            "every test-side type retains at least one train exemplar"
        ),
        "forced_train_tasks": sorted(forced_train),
        "balance": balance,
    }
    return train_ids, test_ids, rationale


def choose_holdout_partition(
    tasks: list[dict], train_pool: set[str], test_ids: set[str]
) -> tuple[set[str], set[str], dict]:
    """Split the original 150-task train pool into 130 train and 20 holdout tasks."""
    task_by_id = {row["paper_id"]: row for row in tasks}
    features, prefix_weights = feature_map(tasks)
    pool_totals = Counter(token for case_id in train_pool for token in features[case_id])
    ratio = HOLDOUT_TASK_COUNT / len(train_pool)

    # A type that has only one representative in the 150-task pool must remain in
    # train, even when other representatives of that type occur in final test.
    coverage_fields = ("problem_class", "formulation_type", "application_field")
    protected_groups = {
        (field, value): {
            case_id for case_id in train_pool if task_by_id[case_id][field] == value
        }
        for field in coverage_fields
        for value in {task_by_id[case_id][field] for case_id in train_pool}
    }
    forced_train = {"earl2005", "ostrowski2012"} | {
        next(iter(group)) for group in protected_groups.values() if len(group) == 1
    }
    candidates = sorted(train_pool - forced_train)

    def score(selection: set[str]) -> float:
        if any(group <= selection for group in protected_groups.values()):
            return math.inf
        counts = Counter(token for case_id in selection for token in features[case_id])
        total_score = 0.0
        for token, total in pool_totals.items():
            prefix = token.split("=", 1)[0]
            target = total * ratio
            error = counts[token] - target
            total_score += prefix_weights[prefix] * error * error / max(1.0, target)
            if total >= 6 and target >= 1 and counts[token] == 0:
                total_score += 4.0 * prefix_weights[prefix]
        return total_score

    rng = random.Random(SEED + 1)
    best_selection = None
    best_score = math.inf
    for _ in range(24):
        selected = set(rng.sample(candidates, HOLDOUT_TASK_COUNT))
        current = score(selected)
        for _round in range(100):
            best_swap = None
            best_swap_score = current
            outside = [case_id for case_id in candidates if case_id not in selected]
            for outgoing in sorted(selected):
                reduced = selected - {outgoing}
                for incoming in outside:
                    proposed = reduced | {incoming}
                    candidate_score = score(proposed)
                    if candidate_score < best_swap_score - 1e-12:
                        best_swap_score = candidate_score
                        best_swap = (outgoing, incoming)
            if best_swap is None:
                break
            selected.remove(best_swap[0])
            selected.add(best_swap[1])
            current = best_swap_score
        if current < best_score:
            best_score = current
            best_selection = set(selected)

    assert best_selection is not None
    holdout_ids = best_selection
    train_ids = train_pool - holdout_ids
    if (
        len(train_ids) != 130
        or len(holdout_ids) != 20
        or len(test_ids) != 30
        or train_ids & holdout_ids
        or train_ids & test_ids
        or holdout_ids & test_ids
        or not forced_train <= train_ids
    ):
        raise ValueError("Train/holdout/test task partition is invalid")

    holdout_counts = Counter(
        token for case_id in holdout_ids for token in features[case_id]
    )
    balance = []
    for token, pool_total in sorted(pool_totals.items()):
        prefix, value = token.split("=", 1)
        all_total = sum(token in features[case_id] for case_id in task_by_id)
        test_total = sum(token in features[case_id] for case_id in test_ids)
        balance.append(
            {
                "dimension": prefix,
                "value": value,
                "all_tasks": all_total,
                "train_tasks": pool_total - holdout_counts[token],
                "holdout_tasks": holdout_counts[token],
                "test_tasks": test_total,
                "target_holdout_tasks": round(pool_total * ratio, 3),
            }
        )
    return train_ids, holdout_ids, {
        "algorithm": "deterministic multi-start local search over stratification error",
        "seed": SEED + 1,
        "objective_score": round(best_score, 12),
        "train_task_count": len(train_ids),
        "holdout_task_count": len(holdout_ids),
        "test_task_count": len(test_ids),
        "holdout_source": "selected only from the frozen 150-task training pool",
        "role": (
            "holdout is for model selection and tuning; final test remains untouched"
        ),
        "coverage_constraint": (
            "every problem class, formulation type, and application field appearing "
            "in holdout or test retains at least one train exemplar; the alias pair stays in train"
        ),
        "forced_train_tasks": sorted(forced_train),
        "balance": balance,
    }


def prepare_instances(runtime_records: list[dict]) -> list[dict]:
    prepared = []
    for source in runtime_records:
        limit, rule = time_limit(
            source["recommended_runtime_seconds"], source["recommended_runtime_status"]
        )
        prepared.append(
            {
                "case_id": source["case_id"],
                "instance": source["instance"],
                "solution": source["solution"],
                "scale": instance_scale(source["instance"]),
                "gurobi_runtime_seconds": source["recommended_runtime_seconds"],
                "gurobi_runtime_status": source["recommended_runtime_status"],
                "gurobi_runtime_source": source["recommended_runtime_source"],
                "time_limit_seconds": limit,
                "time_limit_rule": rule,
                "checker_accepted": source["final_checker_replay"]["accepted"],
            }
        )
    return sorted(prepared, key=lambda row: (row["case_id"], row["instance"]))


def representative_pair(rows: list[dict]) -> list[dict]:
    small = sorted((row for row in rows if row["scale"] == "small"), key=lambda r: r["instance"])
    large = sorted(
        (row for row in rows if row["scale"] == "large"),
        key=lambda r: (
            r["gurobi_runtime_seconds"] is None,
            r["gurobi_runtime_seconds"] if r["gurobi_runtime_seconds"] is not None else 900,
            r["instance"],
        ),
    )
    if not small or not large:
        raise ValueError(f"Cannot form low-resource pair for {rows[0]['case_id']}")
    return [next((row for row in small if row["instance"] == "tiny_instance.json"), small[0]), large[len(large) // 2]]


def representative_large(rows: list[dict]) -> dict:
    large = sorted(
        (row for row in rows if row["scale"] == "large"),
        key=lambda row: (
            row["gurobi_runtime_seconds"] is None,
            row["gurobi_runtime_seconds"]
            if row["gurobi_runtime_seconds"] is not None
            else UNIFIED_EVAL_LIMIT_SECONDS,
            row["instance"],
        ),
    )
    if not large:
        raise ValueError(f"No large instance for {rows[0]['case_id']}")
    return large[len(large) // 2]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * probability) - 1]


def summarize_records(records: list[dict]) -> dict:
    result = {}
    for phase in ("train", "holdout", "test"):
        selected = [row for row in records if row["phase"] == phase]
        if not selected:
            continue
        limits = [row["time_limit_seconds"] for row in selected]
        runtimes = [
            float(row["gurobi_runtime_seconds"])
            if row["gurobi_runtime_seconds"] is not None
            else float(UNIFIED_EVAL_LIMIT_SECONDS)
            for row in selected
        ]
        result[phase] = {
            "task_count": len({row["case_id"] for row in selected}),
            "instance_count": len(selected),
            "small_instance_count": sum(row["scale"] == "small" for row in selected),
            "large_instance_count": sum(row["scale"] == "large" for row in selected),
            "max_per_instance_time_limit_seconds": max(limits),
            "median_per_instance_time_limit_seconds": statistics.median(limits),
            "p90_per_instance_time_limit_seconds": sorted(limits)[
                math.ceil(0.9 * len(limits)) - 1
            ],
            "serial_upper_bound_hours": round(sum(limits) / 3600, 3),
            "uniform_eval_serial_upper_bound_hours": round(
                len(selected) * UNIFIED_EVAL_LIMIT_SECONDS / 3600, 3
            ),
            "median_gurobi_runtime_seconds": round(statistics.median(runtimes), 6),
            "p90_gurobi_runtime_seconds": round(percentile(runtimes, 0.90), 6),
            "p95_gurobi_runtime_seconds": round(percentile(runtimes, 0.95), 6),
            "runtime_over_300_seconds_count": sum(value > 300 for value in runtimes),
            "time_limit_tier_counts": dict(sorted(Counter(limits).items())),
        }
    return result


def evaluation_analysis(records: list[dict]) -> dict:
    evaluated = [row for row in records if row["phase"] in {"holdout", "test"}]
    runtimes = [
        float(row["gurobi_runtime_seconds"])
        if row["gurobi_runtime_seconds"] is not None
        else float(UNIFIED_EVAL_LIMIT_SECONDS)
        for row in evaluated
    ]
    p90 = percentile(runtimes, 0.90)
    split_cap, _ = time_limit(p90, "observed")
    return {
        "evaluated_instance_count": len(evaluated),
        "median_gurobi_runtime_seconds": round(statistics.median(runtimes), 6),
        "p90_gurobi_runtime_seconds": round(p90, 6),
        "p95_gurobi_runtime_seconds": round(percentile(runtimes, 0.95), 6),
        "runtime_over_300_seconds_count": sum(value > 300 for value in runtimes),
        "runtime_over_300_seconds_fraction": round(
            sum(value > 300 for value in runtimes) / len(runtimes), 6
        ),
        "timeout_censored_count": sum(
            row["gurobi_runtime_status"] == "timeout_censored" for row in evaluated
        ),
        "split_specific_p90_cap_seconds": split_cap,
        "official_uniform_eval_cap_seconds": UNIFIED_EVAL_LIMIT_SECONDS,
        "uniform_serial_upper_bound_hours": round(
            len(evaluated) * UNIFIED_EVAL_LIMIT_SECONDS / 3600, 3
        ),
        "rationale": (
            "The p90 falls in the hardest runtime tier. A single 900-second per-instance "
            "cap is therefore used for comparable evaluation across all protocols."
        ),
    }


def build_splits(
    instances: list[dict], train_tasks: set[str], holdout_tasks: set[str], test_tasks: set[str]
) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in instances:
        by_task[row["case_id"]].append(row)

    scale_holdout_keys = {
        (row["case_id"], row["instance"])
        for case_id in sorted(by_task)
        for row in [representative_large(by_task[case_id])]
    }
    definitions = [
        {
            "id": "scale_ood",
            "title": "Scale-OOD",
            "question": "Tests whether a policy trained on small instances can solve larger instances of the same tasks.",
            "train_rule": "All small instances from all 180 tasks",
            "holdout_rule": "One median-runtime large replica from each task",
            "test_rule": "All remaining large replicas from the same 180 tasks",
            "records": [
                {
                    **row,
                    "phase": "train"
                    if row["scale"] == "small"
                    else "holdout"
                    if (row["case_id"], row["instance"]) in scale_holdout_keys
                    else "test",
                }
                for row in instances
            ],
        },
        {
            "id": "task_ood_full",
            "title": "Task-OOD Full",
            "question": "Tests transfer to unseen tasks after training on every available instance in the training partition.",
            "train_rule": "All available instances from the 130 train tasks",
            "holdout_rule": "All available instances from 20 validation tasks",
            "test_rule": "All available instances from the 30 held-out tasks",
            "records": [
                {
                    **row,
                    "phase": "train"
                    if row["case_id"] in train_tasks
                    else "holdout"
                    if row["case_id"] in holdout_tasks
                    else "test",
                }
                for row in instances
            ],
        },
        {
            "id": "task_ood_low_resource",
            "title": "Task-OOD Low-Resource",
            "question": "Uses two instances per task to test transfer when training data is limited.",
            "train_rule": "Exactly one canonical small and one median-runtime large instance per train task",
            "holdout_rule": "The same two-instance rule on each validation task",
            "test_rule": "The same two-instance selection rule on each held-out task",
            "records": [
                {
                    **row,
                    "phase": "train"
                    if case_id in train_tasks
                    else "holdout"
                    if case_id in holdout_tasks
                    else "test",
                }
                for case_id in sorted(by_task)
                for row in representative_pair(by_task[case_id])
            ],
        },
        {
            "id": "joint_ood",
            "title": "Joint-OOD",
            "question": "Combines both shifts: the test tasks are unseen, and their test instances are large.",
            "train_rule": "All small instances from the 130 train tasks",
            "holdout_rule": "All large instances from the 20 validation tasks",
            "test_rule": "All large instances from the 30 held-out tasks",
            "records": [
                {**row, "phase": "train"}
                for row in instances
                if row["case_id"] in train_tasks and row["scale"] == "small"
            ]
            + [
                {**row, "phase": "holdout"}
                for row in instances
                if row["case_id"] in holdout_tasks and row["scale"] == "large"
            ]
            + [
                {**row, "phase": "test"}
                for row in instances
                if row["case_id"] in test_tasks and row["scale"] == "large"
            ],
        },
    ]
    for definition in definitions:
        definition["records"] = sorted(
            definition["records"], key=lambda row: (row["phase"], row["case_id"], row["instance"])
        )
        definition["summary"] = summarize_records(definition["records"])
        definition["evaluation_analysis"] = evaluation_analysis(definition["records"])
        for row in definition["records"]:
            row["eval_time_limit_seconds"] = (
                UNIFIED_EVAL_LIMIT_SECONDS
                if row["phase"] in {"holdout", "test"}
                else None
            )
        definition["timeout_policy"] = {
            "scope": "per evaluated instance, identical in holdout and final test",
            "official_eval_limit_seconds": UNIFIED_EVAL_LIMIT_SECONDS,
            "official_eval_rule": (
                "one agent attempt per instance; terminate at 900 wall-clock seconds"
            ),
            "runtime_tiers_role": (
                "optional rollout scheduling and compute planning, not the official eval cap"
            ),
            "tiers": [
                {"gurobi_runtime_range": "[0, 5] seconds", "time_limit_seconds": 60},
                {"gurobi_runtime_range": "(5, 30] seconds", "time_limit_seconds": 120},
                {"gurobi_runtime_range": "(30, 120] seconds", "time_limit_seconds": 300},
                {"gurobi_runtime_range": "(120, 300] seconds", "time_limit_seconds": 600},
                {"gurobi_runtime_range": "> 300 seconds or timeout-censored", "time_limit_seconds": 900},
            ],
            "reason": (
                "Every split has a Gurobi-runtime p90 above 300 seconds. The common maximum "
                "avoids giving different methods or protocols different per-instance budgets."
            ),
        }
    return definitions


def task_catalog(
    tasks: list[dict], train_tasks: set[str], holdout_tasks: set[str]
) -> list[dict]:
    fields = [
        "case_id",
        "partition",
        "canonical_group",
        "paper_title",
        "year",
        "direction",
        "problem_class",
        "category",
        "formulation_type",
        "application_field",
        "runtime_quartile",
        "size_quartile",
        "task_median_runtime_seconds",
        "large_median_runtime_seconds",
        "avg_num_var",
        "avg_num_int_var",
        "avg_num_constr",
        "source_link",
    ]
    rows = []
    for task in sorted(tasks, key=lambda row: row["paper_id"]):
        row = {
            "case_id": task["paper_id"],
            "partition": "train"
            if task["paper_id"] in train_tasks
            else "holdout"
            if task["paper_id"] in holdout_tasks
            else "test",
        }
        row.update({field: task.get(field) for field in fields if field not in row})
        rows.append(row)
    return rows


def make_index(
    split_defs: list[dict], catalog: list[dict], rationale: dict, replica_audit: dict
) -> str:
    cards = []
    for split in split_defs:
        train = split["summary"]["train"]
        holdout = split["summary"]["holdout"]
        test = split["summary"]["test"]
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{html.escape(split['id'])}</div>
              <h3>{html.escape(split['title'])}</h3>
              <p>{html.escape(split['question'])}</p>
              <div class="metrics"><span><b>{train['instance_count']}</b> train</span><span class="holdout-metric"><b>{holdout['instance_count']}</b> holdout</span><span><b>{test['instance_count']}</b> test</span></div>
              <p class="rule"><strong>Train:</strong> {html.escape(split['train_rule'])}</p>
              <p class="rule"><strong>Holdout:</strong> {html.escape(split['holdout_rule'])}</p>
              <p class="rule"><strong>Test:</strong> {html.escape(split['test_rule'])}</p>
              <a href="splits/{split['id']}.json">JSON</a> · <a href="splits/{split['id']}.csv">CSV</a>
            </article>"""
        )
    task_json = json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/")
    split_json = json.dumps(
        [{"id": s["id"], "title": s["title"], "summary": s["summary"]} for s in split_defs]
    )
    holdout_preview = [row for row in catalog if row["partition"] == "holdout"]
    test_preview = [row for row in catalog if row["partition"] == "test"]
    monotone_percent = 100 * replica_audit["byte_size_monotone_fraction"]
    balance_rows = "".join(
        f"<tr><td>{html.escape(row['dimension'])}</td><td>{html.escape(row['value'])}</td>"
        f"<td>{row['all_tasks']}</td><td>{row['train_tasks']}</td><td>{row['holdout_tasks']}</td><td>{row['test_tasks']}</td></tr>"
        for row in rationale["balance"]
        if row["dimension"] in {"category", "formulation", "runtime", "direction"}
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Four reproducible RL train/holdout/test protocols for FrontierOR.">
  <title>FrontierOR RL Splits</title>
  <style>
    :root {{ --ink:#17212b; --muted:#5e6b78; --paper:#f6f3ec; --panel:#fffdf8; --line:#d8d1c4; --blue:#205b73; --orange:#d46b3c; --green:#3e765b; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:16px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1180px; margin:auto; padding:32px 24px 80px; }}
    header {{ padding:72px 0 42px; border-bottom:1px solid var(--line); }}
    .kicker,.eyebrow {{ color:var(--orange); font-size:.76rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }}
    h1 {{ font-family:Georgia,serif; font-size:clamp(2.6rem,7vw,5.8rem); line-height:.94; margin:.2em 0; letter-spacing:-.045em; }}
    h2 {{ font:700 clamp(1.8rem,4vw,3rem)/1.1 Georgia,serif; margin:0 0 20px; }} h3 {{ margin:.25rem 0 .65rem; font-size:1.35rem; }}
    .lede {{ color:var(--muted); max-width:760px; font-size:1.15rem; }}
    .actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:24px; }} .button {{ display:inline-block; text-decoration:none; border-radius:9px; padding:10px 14px; background:var(--blue); color:white; font-weight:750; }} .button.secondary {{ background:transparent; color:var(--blue); border:1px solid var(--blue); }}
    .stamp {{ display:inline-flex; gap:10px; flex-wrap:wrap; margin-top:18px; }} .stamp span {{ border:1px solid var(--line); border-radius:99px; padding:6px 12px; background:var(--panel); }}
    section {{ padding:52px 0 12px; }} .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:22px; box-shadow:0 8px 26px #473c2c0b; }}
    .metrics {{ display:flex; gap:8px; margin:18px 0; flex-wrap:wrap; }} .metrics span {{ background:#e7eef0; color:var(--blue); border-radius:8px; padding:8px 12px; }} .metrics .holdout-metric {{ background:#eee8f7; color:#604682; }}
    .rule {{ margin:.45rem 0; color:var(--muted); }} a {{ color:var(--blue); }}
    .matrix {{ display:grid; grid-template-columns:160px repeat(2,1fr); border:1px solid var(--line); border-radius:14px; overflow:hidden; background:var(--panel); }}
    .matrix > div {{ padding:16px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); }} .matrix .head {{ font-weight:800; background:#e8e2d8; }}
    .callout {{ border-left:5px solid var(--orange); padding:16px 20px; background:#fff6ed; border-radius:0 12px 12px 0; }}
    .compare {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .compare article {{ border-top:5px solid var(--green); }} .compare article:last-child {{ border-top-color:var(--orange); }}
    pre {{ overflow:auto; padding:18px; border-radius:12px; background:#17212b; color:#edf5f2; font-size:.86rem; }} code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .tiers {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }} .tier {{ padding:15px; border-radius:12px; background:var(--panel); border:1px solid var(--line); }} .tier b {{ display:block; font-size:1.55rem; color:var(--green); }}
    .controls {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }} input,select {{ padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:white; font:inherit; }} input {{ min-width:280px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:white; max-height:560px; }} table {{ border-collapse:collapse; width:100%; font-size:.88rem; }} th,td {{ padding:9px 11px; border-bottom:1px solid #eee8dd; text-align:left; white-space:nowrap; }} th {{ position:sticky; top:0; background:#ece7dd; z-index:1; }}
    .pill {{ padding:3px 8px; border-radius:99px; font-weight:700; }} .train {{ background:#e3f0e8; color:#245a3f; }} .holdout {{ background:#eee8f7; color:#604682; }} .test {{ background:#fde6db; color:#8b3f20; }}
    footer {{ color:var(--muted); padding-top:48px; }}
    @media(max-width:760px) {{ .grid,.compare {{ grid-template-columns:1fr; }} .tiers {{ grid-template-columns:repeat(2,1fr); }} .matrix {{ grid-template-columns:110px repeat(2,1fr); font-size:.86rem; }} }}
  </style>
</head>
<body><main>
  <header>
    <div class="kicker">FrontierOR · RL evaluation</div>
    <h1>Train, holdout, and test<br>for FrontierOR RL</h1>
    <p class="lede">Four protocols separate model selection from final evaluation while testing transfer to larger instances, unseen tasks, or both.</p>
    <div class="stamp"><span>180 task directories</span><span>1,095 verified instances</span><span>130 / 20 / 30 task partition</span><span>15 min uniform eval cap</span></div>
    <div class="actions"><a class="button" href="https://huggingface.co/datasets/{SOURCE_REPO}">View the data on Hugging Face</a><a class="button secondary" href="https://github.com/Chonghe-Jiang/FrontierOR-RL-Splits">View the split files on GitHub</a></div>
  </header>

  <section>
    <h2>What each split tests</h2>
    <div class="matrix">
      <div class="head"></div><div class="head">Small instance</div><div class="head">Large instance</div>
      <div class="head">Seen task</div><div>Training-distribution baseline</div><div><strong>Scale-OOD</strong><br>same task, larger instance</div>
      <div class="head">Unseen task</div><div><strong>Task-OOD</strong><br>new task, comparable scale mix</div><div><strong>Joint-OOD</strong><br>new task and larger instance</div>
    </div>
  </section>

  <section>
    <h2>Why a 3 / 2 replica split is not Scale-OOD</h2>
    <p>Most tasks have one tiny instance and five large replicas. The suffixes <code>large_1</code> through <code>large_5</code> label separate replicas, not five ordered scale levels. We checked serialized file size as a simple proxy. Only <strong>{replica_audit['byte_size_monotone_task_count']} of {replica_audit['complete_five_large_task_count']}</strong> complete tasks ({monotone_percent:.1f}%) grow monotonically from 1 to 5, and the median ratio between adjacent files is {replica_audit['median_adjacent_byte_size_ratio']:.3f}.</p>
    <div class="compare">
      <article class="card"><div class="eyebrow">Published split</div><h3>Scale-OOD</h3><p>Train on tiny instances and test on large instances of the same tasks. This uses the dataset's explicit small/large boundary.</p></article>
      <article class="card"><div class="eyebrow">Optional split</div><h3>Large-replica holdout 3 / 2</h3><p>Train on three large replicas and test on two others. This tests transfer to held-out replicas or random seeds. It does not test scale extrapolation.</p></article>
    </div>
    <p class="callout"><strong>One exception:</strong> <code>segundo2019</code> has only <code>large_1</code>, <code>large_2</code>, and <code>large_5</code>. An exact 3/2 large-only split therefore covers 179 tasks. To retain all 180, use a documented 2/1 split for this task.</p>
  </section>

  <section>
    <h2>Why there is now a holdout set</h2>
    <p>The holdout partition is the only data used for model selection, prompt revision, reward tuning, and checkpoint choice. Final-test results are consulted only after those choices are frozen. For task-based protocols, 20 tasks are drawn only from the former 150-task training pool, leaving 130 train, 20 holdout, and the original 30 final-test tasks. For Scale-OOD, each task contributes one median-runtime large replica to holdout; the other large replicas remain in final test.</p>
    <p class="callout"><strong>Stable task test:</strong> Task-OOD and Joint-OOD keep the original 30 final-test task IDs unchanged. Scale-OOD uses an instance boundary, so 180 large replicas that previously belonged to test now serve as validation holdout.</p>
  </section>

  <section><h2>Four ready-to-use protocols</h2><div class="grid">{''.join(cards)}</div></section>

  <section>
    <h2>How we chose the task partitions</h2>
    <p>The 30 final-test tasks remain fixed. A second deterministic, stratified search selects 20 holdout tasks from the old 150-task training pool. It balances category, formulation, application field, optimization direction, runtime quartile, model-size quartile, publication era, and sufficiently represented problem classes.</p>
    <div class="callout"><strong>Leakage control:</strong> <code>earl2005</code> and its legacy alias <code>ostrowski2012</code> stay together in train. The release has 180 directories and 179 independent benchmark identities.</div>
    <p><a href="splits/task_partition.json">Full partition rationale and balance table</a><br>Holdout preview: {', '.join(html.escape(row['case_id']) for row in holdout_preview[:8])}…<br>Test preview: {', '.join(html.escape(row['case_id']) for row in test_preview[:8])}…</p>
    <details><summary>Selected balance table</summary><div class="table-wrap"><table><thead><tr><th>Dimension</th><th>Value</th><th>All</th><th>Train</th><th>Holdout</th><th>Test</th></tr></thead><tbody>{balance_rows}</tbody></table></div></details>
  </section>

  <section>
    <h2>One evaluation limit</h2>
    <p>All four protocols use the same wall-clock limit: <strong>900 seconds per holdout or test instance</strong>. We compared the Gurobi runtime distribution in each evaluation set; every protocol has a 90th percentile above 300 seconds and therefore reaches the hardest runtime tier. A common cap makes scores comparable and avoids granting easier-looking splits a smaller budget.</p>
    <div class="table-wrap"><table><thead><tr><th>Protocol</th><th>Holdout</th><th>Final test</th><th>Gurobi p90</th><th>&gt;300s</th><th>Uniform cap</th><th>Serial maximum</th></tr></thead><tbody>{''.join(f"<tr><td>{html.escape(split['title'])}</td><td>{split['summary']['holdout']['instance_count']}</td><td>{split['summary']['test']['instance_count']}</td><td>{split['evaluation_analysis']['p90_gurobi_runtime_seconds']:.1f}s</td><td>{100 * split['evaluation_analysis']['runtime_over_300_seconds_fraction']:.1f}%</td><td>900s / instance</td><td>{split['evaluation_analysis']['uniform_serial_upper_bound_hours']:.1f}h</td></tr>" for split in split_defs)}</tbody></table></div>
    <p class="callout"><strong>Interpretation:</strong> 15 minutes is a per-instance limit, not a promise that a complete evaluation consumes only 15 minutes of compute. The serial maximum is instance count × 15 minutes; parallel workers reduce wall time but not total compute.</p>
    <p>The older 60/120/300/600/900-second tiers remain in <a href="splits/time_limits.csv">the runtime table</a> for optional training-rollout scheduling and capacity planning. They are not the official evaluation limit.</p>
  </section>

  <section>
    <h2>Run evaluation consistently</h2>
    <ol><li>Tune and select checkpoints only on rows marked <code>holdout</code>.</li><li>Freeze the agent, prompt, checker-visibility policy, and decoding settings before using <code>test</code> results.</li><li>Give each evaluated instance one primary attempt and stop it after 900 wall-clock seconds. Any repeated-seed evaluation must use the same repeat count for every compared method.</li><li>Run the task's schema and checker on the submitted solution; retain malformed output, checker failure, and timeout as explicit failures.</li><li>Report feasibility, objective quality, timeout rate, checker-failure rate, and wall-clock compute. Aggregate within each task first, then average across tasks.</li></ol>
    <p>The manifests are public, so this separation is procedural rather than cryptographic. A competition can apply the same IDs to a private copy of the instances or reference solutions for a genuinely hidden final test.</p>
    <p><a href="splits/evaluation_protocol.json">Machine-readable evaluation protocol and budget analysis</a></p>
  </section>

  <section>
    <h2>Inspect the task partition</h2>
    <div class="controls"><input id="search" placeholder="Search task, class, application…"><select id="phase"><option value="">All partitions</option><option>train</option><option>holdout</option><option>test</option></select><select id="klass"><option value="">All problem classes</option></select></div>
    <div class="table-wrap"><table><thead><tr><th>Task</th><th>Split</th><th>Problem class</th><th>Formulation</th><th>Application</th><th>Runtime</th></tr></thead><tbody id="taskRows"></tbody></table></div>
    <p id="count"></p>
  </section>

  <section>
    <h2>Create another split</h2>
    <p>The four manifests are defaults. The source tables contain enough information to create a different split. <a href="splits/time_limits.csv"><code>splits/time_limits.csv</code></a> records the task, instance, scale label, Gurobi runtime source, checker status, and execution limit. <a href="data/task_catalog.csv"><code>data/task_catalog.csv</code></a> adds problem class, formulation, application, year, and difficulty.</p>
    <p>This command creates a deterministic 3/2 split of the five large replicas:</p>
    <pre><code>python examples/make_custom_split.py \\
  --train-count 3 --test-count 2 \\
  --incomplete-policy proportional \\
  --output my_large_replica_split.csv</code></pre>
    <p>Choose <code>--incomplete-policy skip</code> for an exact 3/2 split over 179 tasks. Choose <code>proportional</code> to retain <code>segundo2019</code> with the 2/1 exception described above. A fixed hash seed makes the assignment reproducible.</p>
    <div class="actions"><a class="button" href="examples/make_custom_split.py">Download the split script</a><a class="button secondary" href="https://huggingface.co/datasets/{SOURCE_REPO}/tree/{SOURCE_REVISION}">View the pinned Hugging Face revision</a></div>
  </section>

  <section>
    <h2>Data and reproducibility</h2>
    <p>The splits use <a href="https://huggingface.co/datasets/{SOURCE_REPO}/tree/{SOURCE_REVISION}">{SOURCE_REPO}@{SOURCE_REVISION[:12]}</a>. All 1,095 reference solutions pass their checker, and every row keeps the source of its runtime measurement.</p>
    <p>Run <code>python scripts/build_splits.py</code> to rebuild the files and <code>python scripts/validate_splits.py</code> to check them. <a href="MANIFEST.sha256">MANIFEST.sha256</a> contains the generated-file hashes.</p>
  </section>
  <footer>FrontierOR RL Splits · seed {SEED} · source revision {SOURCE_REVISION[:12]}</footer>
</main>
<script>
const tasks={task_json}; const splitSummary={split_json};
const body=document.getElementById('taskRows'), search=document.getElementById('search'), phase=document.getElementById('phase'), klass=document.getElementById('klass'), count=document.getElementById('count');
[...new Set(tasks.map(x=>x.problem_class))].sort().forEach(x=>{{const o=document.createElement('option');o.textContent=x;klass.appendChild(o);}});
function render(){{const q=search.value.toLowerCase();const rows=tasks.filter(x=>(!phase.value||x.partition===phase.value)&&(!klass.value||x.problem_class===klass.value)&&JSON.stringify(x).toLowerCase().includes(q));body.innerHTML=rows.map(x=>`<tr><td><code>${{x.case_id}}</code></td><td><span class="pill ${{x.partition}}">${{x.partition}}</span></td><td>${{x.problem_class}}</td><td>${{x.formulation_type}}</td><td>${{x.application_field}}</td><td>${{x.runtime_quartile}}</td></tr>`).join('');count.textContent=`Showing ${{rows.length}} of ${{tasks.length}} tasks`;}}
[search,phase,klass].forEach(x=>x.addEventListener('input',render));render();
</script></body></html>
"""


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    tasks_raw, runtime_records, runtime_payload = load_source(args.dataset_root.resolve())
    tasks = enrich_tasks(tasks_raw, runtime_records)
    train_pool, test_tasks, test_rationale = choose_task_partition(tasks)
    train_tasks, holdout_tasks, holdout_rationale = choose_holdout_partition(
        tasks, train_pool, test_tasks
    )
    rationale = {
        "partition": "130 train / 20 holdout / 30 final test",
        "test_selection": test_rationale,
        "holdout_selection": holdout_rationale,
        "balance": holdout_rationale["balance"],
    }
    instances = prepare_instances(runtime_records)
    splits = build_splits(instances, train_tasks, holdout_tasks, test_tasks)
    catalog = task_catalog(tasks, train_tasks, holdout_tasks)
    replica_audit = large_replica_audit(args.dataset_root.resolve())

    provenance = {
        "format": "frontieror-rl-split-provenance-v1",
        "generated_at": runtime_payload["generated_at"],
        "source_dataset": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_runtime_format": runtime_payload["format"],
        "source_runtime_summary": runtime_payload["summary"],
        "builder_seed": SEED,
        "large_replica_audit": replica_audit,
    }
    write_json(root / "data/source_provenance.json", provenance)

    catalog_fields = list(catalog[0])
    write_csv(root / "data/task_catalog.csv", catalog, catalog_fields)
    write_json(
        root / "splits/task_partition.json",
        {
            "format": "frontieror-task-partition-v2",
            "source_revision": SOURCE_REVISION,
            "train_tasks": sorted(train_tasks),
            "holdout_tasks": sorted(holdout_tasks),
            "test_tasks": sorted(test_tasks),
            "rationale": rationale,
        },
    )
    partition_rows = [
        {
            "case_id": row["case_id"],
            "partition": row["partition"],
            "canonical_group": row["canonical_group"],
            "problem_class": row["problem_class"],
            "formulation_type": row["formulation_type"],
            "application_field": row["application_field"],
            "category": row["category"],
            "direction": row["direction"],
            "runtime_quartile": row["runtime_quartile"],
            "size_quartile": row["size_quartile"],
        }
        for row in catalog
    ]
    write_csv(root / "splits/task_partition.csv", partition_rows, list(partition_rows[0]))

    instance_fields = [
        "phase",
        "case_id",
        "instance",
        "solution",
        "scale",
        "gurobi_runtime_seconds",
        "gurobi_runtime_status",
        "gurobi_runtime_source",
        "time_limit_seconds",
        "time_limit_rule",
        "eval_time_limit_seconds",
        "checker_accepted",
    ]
    for split in splits:
        payload = {key: value for key, value in split.items() if key != "records"}
        payload.update(
            {
                "format": "frontieror-rl-split-v1",
                "source_revision": SOURCE_REVISION,
                "records": split["records"],
            }
        )
        write_json(root / f"splits/{split['id']}.json", payload)
        write_csv(root / f"splits/{split['id']}.csv", split["records"], instance_fields)

    evaluation_rows = [
        {
            "split_id": split["id"],
            "title": split["title"],
            "holdout_instance_count": split["summary"]["holdout"]["instance_count"],
            "test_instance_count": split["summary"]["test"]["instance_count"],
            **split["evaluation_analysis"],
        }
        for split in splits
    ]
    write_json(
        root / "splits/evaluation_protocol.json",
        {
            "format": "frontieror-evaluation-protocol-v1",
            "source_revision": SOURCE_REVISION,
            "holdout_role": (
                "model selection, prompt/reward tuning, and checkpoint selection only"
            ),
            "final_test_rule": "use final-test results only after the evaluation configuration is frozen",
            "visibility_note": (
                "the published manifests are procedurally held out, not cryptographically hidden"
            ),
            "uniform_eval_limit_seconds_per_instance": UNIFIED_EVAL_LIMIT_SECONDS,
            "primary_attempts_per_instance": 1,
            "repeat_rule": (
                "optional repeated-seed evaluation must use the same repeat count for every method"
            ),
            "timeout_result": "retain and report as an explicit failure",
            "aggregation": (
                "aggregate instance metrics within task, then macro-average across tasks"
            ),
            "required_metrics": [
                "checker-accepted feasibility rate",
                "objective quality",
                "timeout rate",
                "checker-failure rate",
                "wall-clock compute",
            ],
            "split_analysis": evaluation_rows,
        },
    )

    time_fields = [
        field
        for field in instance_fields
        if field not in {"phase", "eval_time_limit_seconds"}
    ]
    write_csv(root / "splits/time_limits.csv", instances, time_fields)
    write_text(root / "index.html", make_index(splits, catalog, rationale, replica_audit))

    generated = sorted(
        path
        for folder in (root / "data", root / "splits")
        for path in folder.glob("*")
        if path.is_file()
    ) + [root / "index.html"]
    manifest_lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in generated]
    write_text(root / "MANIFEST.sha256", "\n".join(manifest_lines) + "\n")

    print(
        "Task partition:",
        len(train_tasks),
        "train /",
        len(holdout_tasks),
        "holdout /",
        len(test_tasks),
        "test",
    )
    for split in splits:
        print(split["id"], json.dumps(split["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
