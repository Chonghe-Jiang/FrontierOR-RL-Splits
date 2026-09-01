#!/usr/bin/env python3
"""Build four reproducible FrontierOR RL train/test split manifests."""

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
from datetime import datetime, timezone
from pathlib import Path


SOURCE_REVISION = "a6fe77d0c79184bbea1e8f72ca6efd1a75eec1cf"
SOURCE_REPO = "LeoJiangOR/FrontierOR-Audited-180"
SEED = 20260831
TEST_TASK_COUNT = 30
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
        "objective_score": best_score,
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


def summarize_records(records: list[dict]) -> dict:
    result = {}
    for phase in ("train", "test"):
        selected = [row for row in records if row["phase"] == phase]
        limits = [row["time_limit_seconds"] for row in selected]
        observed = [
            row["gurobi_runtime_seconds"]
            for row in selected
            if row["gurobi_runtime_seconds"] is not None
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
            "median_gurobi_runtime_seconds": round(statistics.median(observed), 6),
            "time_limit_tier_counts": dict(sorted(Counter(limits).items())),
        }
    return result


def build_splits(
    instances: list[dict], train_tasks: set[str], test_tasks: set[str]
) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in instances:
        by_task[row["case_id"]].append(row)

    definitions = [
        {
            "id": "scale_ood",
            "title": "Scale-OOD",
            "question": "Can a policy learned on small instances scale to large instances of the same tasks?",
            "train_rule": "All small instances from all 180 tasks",
            "test_rule": "All large instances from the same 180 tasks",
            "records": [
                {**row, "phase": "train" if row["scale"] == "small" else "test"}
                for row in instances
            ],
        },
        {
            "id": "task_ood_full",
            "title": "Task-OOD Full",
            "question": "Can a policy transfer to unseen tasks when all available training instances are used?",
            "train_rule": "All available instances from the 150 train tasks",
            "test_rule": "All available instances from the 30 held-out tasks",
            "records": [
                {**row, "phase": "train" if row["case_id"] in train_tasks else "test"}
                for row in instances
            ],
        },
        {
            "id": "task_ood_low_resource",
            "title": "Task-OOD Low-Resource",
            "question": "Can a policy transfer to unseen tasks with only two representative instances per task?",
            "train_rule": "Exactly one canonical small and one median-runtime large instance per train task",
            "test_rule": "The same two-instance selection rule on each held-out task",
            "records": [
                {**row, "phase": "train" if case_id in train_tasks else "test"}
                for case_id in sorted(by_task)
                for row in representative_pair(by_task[case_id])
            ],
        },
        {
            "id": "joint_ood",
            "title": "Joint-OOD",
            "question": "Can a policy generalize simultaneously to unseen tasks and larger instances?",
            "train_rule": "All small instances from the 150 train tasks",
            "test_rule": "All large instances from the 30 held-out tasks",
            "records": [
                {**row, "phase": "train"}
                for row in instances
                if row["case_id"] in train_tasks and row["scale"] == "small"
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
        definition["timeout_policy"] = {
            "scope": "per instance, identical in train and test",
            "maximum_seconds": 900,
            "tiers": [
                {"gurobi_runtime_range": "[0, 5] seconds", "time_limit_seconds": 60},
                {"gurobi_runtime_range": "(5, 30] seconds", "time_limit_seconds": 120},
                {"gurobi_runtime_range": "(30, 120] seconds", "time_limit_seconds": 300},
                {"gurobi_runtime_range": "(120, 300] seconds", "time_limit_seconds": 600},
                {"gurobi_runtime_range": "> 300 seconds or timeout-censored", "time_limit_seconds": 900},
            ],
            "reason": (
                "Coarse tiers avoid false precision across heterogeneous runtime sources while "
                "giving slower Gurobi instances more wall-clock budget."
            ),
        }
    return definitions


def task_catalog(tasks: list[dict], train_tasks: set[str]) -> list[dict]:
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
        row = {"case_id": task["paper_id"], "partition": "train" if task["paper_id"] in train_tasks else "test"}
        row.update({field: task.get(field) for field in fields if field not in row})
        rows.append(row)
    return rows


def make_index(
    split_defs: list[dict], catalog: list[dict], rationale: dict, replica_audit: dict
) -> str:
    cards = []
    for split in split_defs:
        train = split["summary"]["train"]
        test = split["summary"]["test"]
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{html.escape(split['id'])}</div>
              <h3>{html.escape(split['title'])}</h3>
              <p>{html.escape(split['question'])}</p>
              <div class="metrics"><span><b>{train['instance_count']}</b> train</span><span><b>{test['instance_count']}</b> test</span></div>
              <p class="rule"><strong>Train:</strong> {html.escape(split['train_rule'])}</p>
              <p class="rule"><strong>Test:</strong> {html.escape(split['test_rule'])}</p>
              <a href="splits/{split['id']}.json">JSON</a> · <a href="splits/{split['id']}.csv">CSV</a>
            </article>"""
        )
    task_json = json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/")
    split_json = json.dumps(
        [{"id": s["id"], "title": s["title"], "summary": s["summary"]} for s in split_defs]
    )
    test_preview = [row for row in catalog if row["partition"] == "test"]
    monotone_percent = 100 * replica_audit["byte_size_monotone_fraction"]
    balance_rows = "".join(
        f"<tr><td>{html.escape(row['dimension'])}</td><td>{html.escape(row['value'])}</td>"
        f"<td>{row['all_tasks']}</td><td>{row['train_tasks']}</td><td>{row['test_tasks']}</td></tr>"
        for row in rationale["balance"]
        if row["dimension"] in {"category", "formulation", "runtime", "direction"}
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Four reproducible RL train/test splits for FrontierOR.">
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
    .metrics {{ display:flex; gap:8px; margin:18px 0; }} .metrics span {{ background:#e7eef0; color:var(--blue); border-radius:8px; padding:8px 12px; }}
    .rule {{ margin:.45rem 0; color:var(--muted); }} a {{ color:var(--blue); }}
    .matrix {{ display:grid; grid-template-columns:160px repeat(2,1fr); border:1px solid var(--line); border-radius:14px; overflow:hidden; background:var(--panel); }}
    .matrix > div {{ padding:16px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); }} .matrix .head {{ font-weight:800; background:#e8e2d8; }}
    .callout {{ border-left:5px solid var(--orange); padding:16px 20px; background:#fff6ed; border-radius:0 12px 12px 0; }}
    .compare {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .compare article {{ border-top:5px solid var(--green); }} .compare article:last-child {{ border-top-color:var(--orange); }}
    pre {{ overflow:auto; padding:18px; border-radius:12px; background:#17212b; color:#edf5f2; font-size:.86rem; }} code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .tiers {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }} .tier {{ padding:15px; border-radius:12px; background:var(--panel); border:1px solid var(--line); }} .tier b {{ display:block; font-size:1.55rem; color:var(--green); }}
    .controls {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }} input,select {{ padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:white; font:inherit; }} input {{ min-width:280px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:white; max-height:560px; }} table {{ border-collapse:collapse; width:100%; font-size:.88rem; }} th,td {{ padding:9px 11px; border-bottom:1px solid #eee8dd; text-align:left; white-space:nowrap; }} th {{ position:sticky; top:0; background:#ece7dd; z-index:1; }}
    .pill {{ padding:3px 8px; border-radius:99px; font-weight:700; }} .train {{ background:#e3f0e8; color:#245a3f; }} .test {{ background:#fde6db; color:#8b3f20; }}
    footer {{ color:var(--muted); padding-top:48px; }}
    @media(max-width:760px) {{ .grid,.compare {{ grid-template-columns:1fr; }} .tiers {{ grid-template-columns:repeat(2,1fr); }} .matrix {{ grid-template-columns:110px repeat(2,1fr); font-size:.86rem; }} }}
  </style>
</head>
<body><main>
  <header>
    <div class="kicker">FrontierOR · RL evaluation protocol</div>
    <h1>Four splits.<br>Two kinds of generalization.</h1>
    <p class="lede">A reproducible train/test protocol for asking whether optimization agents generalize across instance scale, across problem tasks, or across both at once.</p>
    <div class="stamp"><span>180 task directories</span><span>1,095 verified instances</span><span>150 / 30 task partition</span><span>≤ 15 min per run</span></div>
    <div class="actions"><a class="button" href="https://huggingface.co/datasets/{SOURCE_REPO}">Open the dataset on Hugging Face</a><a class="button secondary" href="https://github.com/Chonghe-Jiang/FrontierOR-RL-Splits">View the GitHub repository</a></div>
  </header>

  <section>
    <h2>The evaluation matrix</h2>
    <div class="matrix">
      <div class="head"></div><div class="head">Small instance</div><div class="head">Large instance</div>
      <div class="head">Seen task</div><div>Within-distribution control</div><div><strong>Scale-OOD</strong><br>same task, larger instance</div>
      <div class="head">Unseen task</div><div><strong>Task-OOD</strong><br>new task, familiar scale mix</div><div><strong>Joint-OOD</strong><br>new task and larger instance</div>
    </div>
  </section>

  <section>
    <h2>Scale is not the same as a 3 / 2 replicate holdout</h2>
    <p>Each standard task has one tiny instance and five nominally large replicas. The suffixes <code>large_1</code> through <code>large_5</code> identify replicas; they do not define five increasing scales. In a serialized-byte-size audit, only <strong>{replica_audit['byte_size_monotone_task_count']} of {replica_audit['complete_five_large_task_count']}</strong> complete tasks ({monotone_percent:.1f}%) increase monotonically from 1 to 5, and the median adjacent size ratio is {replica_audit['median_adjacent_byte_size_ratio']:.3f}.</p>
    <div class="compare">
      <article class="card"><div class="eyebrow">Published protocol</div><h3>True Scale-OOD</h3><p>Train on tiny instances and test on large instances of the same tasks. This measures extrapolation across the explicit small/large boundary.</p></article>
      <article class="card"><div class="eyebrow">Optional custom protocol</div><h3>Large-replica holdout 3 / 2</h3><p>Train on three large replicas and test on two other large replicas. This is useful, but it measures held-out-instance or random-seed generalization—not scale extrapolation.</p></article>
    </div>
    <p class="callout"><strong>Dataset exception:</strong> <code>segundo2019</code> retains only <code>large_1</code>, <code>large_2</code>, and <code>large_5</code>. An exact 3/2 large-only protocol therefore covers 179 complete tasks; retaining all 180 requires a documented proportional 2/1 exception.</p>
  </section>

  <section><h2>The four published splits</h2><div class="grid">{''.join(cards)}</div></section>

  <section>
    <h2>How the 150 / 30 task boundary was chosen</h2>
    <p>The 30 held-out tasks are not a random sample. A deterministic local-search partition balances category, formulation, application field, optimization direction, runtime quartile, model-size quartile, publication era, and sufficiently represented problem classes.</p>
    <div class="callout"><strong>Leakage guard:</strong> <code>earl2005</code> and its legacy alias <code>ostrowski2012</code> are forced into train together. The release has 180 directories but 179 independent benchmark identities.</div>
    <p><a href="splits/task_partition.json">Full partition rationale and balance table</a> · Test preview: {', '.join(html.escape(row['case_id']) for row in test_preview[:8])}…</p>
    <details><summary>Selected balance table</summary><div class="table-wrap"><table><thead><tr><th>Dimension</th><th>Value</th><th>All</th><th>Train</th><th>Test</th></tr></thead><tbody>{balance_rows}</tbody></table></div></details>
  </section>

  <section>
    <h2>Runtime-aware limits</h2>
    <p>Every instance receives the same limit wherever it appears, whether in train or test. Coarse tiers are intentional: the runtime metadata combines exact current reruns and historical measurements from different environments, so fine-grained scaling would imply false precision.</p>
    <div class="tiers"><div class="tier"><b>60s</b>Gurobi ≤ 5s</div><div class="tier"><b>120s</b>5–30s</div><div class="tier"><b>300s</b>30–120s</div><div class="tier"><b>600s</b>120–300s</div><div class="tier"><b>900s</b>&gt;300s or censored</div></div>
    <p class="callout"><strong>Hard rule:</strong> no single training rollout or evaluation attempt may exceed 900 seconds (15 minutes). Report aggregate compute separately; the manifests include serial upper bounds for planning.</p>
    <p><a href="splits/time_limits.csv">Download all 1,095 instance limits</a></p>
  </section>

  <section>
    <h2>Inspect the task partition</h2>
    <div class="controls"><input id="search" placeholder="Search task, class, application…"><select id="phase"><option value="">All partitions</option><option>train</option><option>test</option></select><select id="klass"><option value="">All problem classes</option></select></div>
    <div class="table-wrap"><table><thead><tr><th>Task</th><th>Split</th><th>Problem class</th><th>Formulation</th><th>Application</th><th>Runtime</th></tr></thead><tbody id="taskRows"></tbody></table></div>
    <p id="count"></p>
  </section>

  <section>
    <h2>You can define your own split</h2>
    <p>The four manifests are recommended evaluation protocols, not restrictions. Every row in <a href="splits/time_limits.csv"><code>splits/time_limits.csv</code></a> includes the task ID, instance, small/large label, Gurobi runtime provenance, checker status, and runtime-aware execution cap. Combine it with <a href="data/task_catalog.csv"><code>data/task_catalog.csv</code></a> to split by problem class, formulation, application, publication era, difficulty, or a custom task list.</p>
    <p>For a deterministic three-train/two-test split of the five large replicas:</p>
    <pre><code>python examples/make_custom_split.py \\
  --train-count 3 --test-count 2 \\
  --incomplete-policy proportional \\
  --output my_large_replica_split.csv</code></pre>
    <p>Use <code>--incomplete-policy skip</code> for an exact 3/2 protocol over the 179 complete tasks, or <code>proportional</code> to retain <code>segundo2019</code> as a documented 2/1 exception. The script uses a fixed hash seed, so independent users obtain the same assignment.</p>
    <div class="actions"><a class="button" href="examples/make_custom_split.py">Download the custom split script</a><a class="button secondary" href="https://huggingface.co/datasets/{SOURCE_REPO}/tree/{SOURCE_REVISION}">Browse the pinned Hugging Face revision</a></div>
  </section>

  <section>
    <h2>Reproducibility</h2>
    <p>Source dataset: <a href="https://huggingface.co/datasets/{SOURCE_REPO}/tree/{SOURCE_REVISION}">{SOURCE_REPO}@{SOURCE_REVISION[:12]}</a>. All 1,095 reference solutions pass their checker. Runtime source provenance is retained in every row.</p>
    <p>Rebuild with <code>python scripts/build_splits.py</code>; verify with <code>python scripts/validate_splits.py</code>. Generated-file hashes are recorded in <a href="MANIFEST.sha256">MANIFEST.sha256</a>.</p>
  </section>
  <footer>FrontierOR RL Splits · deterministic seed {SEED} · generated from immutable source revision {SOURCE_REVISION[:12]}</footer>
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
    train_tasks, test_tasks, rationale = choose_task_partition(tasks)
    instances = prepare_instances(runtime_records)
    splits = build_splits(instances, train_tasks, test_tasks)
    catalog = task_catalog(tasks, train_tasks)
    replica_audit = large_replica_audit(args.dataset_root.resolve())

    provenance = {
        "format": "frontieror-rl-split-provenance-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
            "format": "frontieror-task-partition-v1",
            "source_revision": SOURCE_REVISION,
            "train_tasks": sorted(train_tasks),
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

    time_fields = [field for field in instance_fields if field != "phase"]
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

    print("Task partition:", len(train_tasks), "train /", len(test_tasks), "test")
    for split in splits:
        print(split["id"], json.dumps(split["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
