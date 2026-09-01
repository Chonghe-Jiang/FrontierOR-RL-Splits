# FrontierOR RL Splits

Four reproducible train/holdout/test protocols for reinforcement-learning experiments
on [FrontierOR-Audited-180](https://huggingface.co/datasets/LeoJiangOR/FrontierOR-Audited-180).

Open **[the visual guide](https://chonghe-jiang.github.io/FrontierOR-RL-Splits/)**
for the motivation, exact instance counts, task browser, holdout policy, and evaluation
budget analysis.

## Published protocols

| ID | Train | Holdout for model selection | Final test | Generalization question |
|---|---:|---:|---:|---|
| `scale_ood` | 197 small instances | 180 median-runtime large replicas | 718 remaining large replicas | Scale extrapolation on seen tasks |
| `task_ood_full` | 794 instances from 130 tasks | 120 instances from 20 tasks | 181 instances from 30 tasks | New-task transfer with full data |
| `task_ood_low_resource` | 260 instances | 40 instances | 60 instances | New-task transfer with two examples per task |
| `joint_ood` | 144 small instances from 130 tasks | 100 large instances from 20 tasks | 148 large instances from 30 tasks | Simultaneous task and scale OOD |

For the task-based protocols, the original 30 final-test tasks remain unchanged. The 20 holdout tasks were selected
only from the former 150-task training pool, leaving a fixed **130 train / 20 holdout /
30 final-test** partition. The deterministic search balances problem metadata and
difficulty proxies while ensuring that every test- or holdout-side type retains a train
example. The `earl2005` / `ostrowski2012` alias pair stays together in train.

Scale-OOD uses a different holdout boundary because it evaluates scale rather than task
transfer. Every task contributes its median-runtime large replica to holdout; the other
large replicas remain untouched for final testing.

## Uniform evaluation limit

The official limit is **900 wall-clock seconds per holdout or final-test instance**, with
one attempt per instance. It is the same for every protocol and method.

| Protocol | Holdout + test | Gurobi runtime p90 | Runtime >300s | Serial maximum at 900s each |
|---|---:|---:|---:|---:|
| Scale-OOD | 898 | 3,609.59s | 63.9% | 224.50h |
| Task-OOD Full | 301 | 3,603.23s | 56.1% | 75.25h |
| Task-OOD Low-Resource | 100 | 3,601.68s | 34.0% | 25.00h |
| Joint-OOD | 248 | 3,606.86s | 68.1% | 62.00h |

All four p90 values fall in the hardest available runtime tier, so a split-specific rule
would select 900 seconds in every case. The single cap is easier to audit and keeps scores
comparable. The serial maximum is a compute-planning bound, not expected wall time;
parallel workers can reduce elapsed time.

The 60/120/300/600/900-second columns in `splits/time_limits.csv` remain available for
optional training-rollout scheduling. They are not the official evaluation cap.

## Evaluation protocol

1. Use `holdout` for prompt, reward, hyperparameter, and checkpoint selection.
2. Freeze the complete agent configuration before using `test` results.
3. Give each evaluated instance one primary attempt and terminate it at 900 wall-clock
   seconds. If repeated seeds are reported, use the same repeat count for every method.
4. Validate the output schema and run the task checker. Preserve malformed output,
   checker failure, and timeout as explicit failures.
5. Report feasibility, objective quality, timeout rate, checker-failure rate, and total
   wall-clock compute.
6. Aggregate instances within each task first, then macro-average across tasks.

Do not expose held-out Gurobi code or reference solutions to the policy. Decide whether
checker source is visible before comparison and keep that policy identical across methods.
The published manifests are procedurally held out, not technically secret; a competition
can use the same IDs against a private dataset copy for a genuinely hidden final test.

## Files

- `index.html`: self-contained visual guide and interactive task browser.
- `splits/*.json`: complete protocol definitions, summaries, and instance records.
- `splits/*.csv`: flat train/holdout/test manifests.
- `splits/evaluation_protocol.json`: common cap, reporting rules, and runtime analysis.
- `splits/task_partition.*`: fixed 130/20/30 task boundary and rationale.
- `splits/time_limits.csv`: runtime provenance and optional scheduling tiers for all 1,095 instances.
- `data/task_catalog.csv`: task taxonomy, partition, and difficulty bins.
- `examples/make_custom_split.py`: deterministic large-replica split generator.
- `MANIFEST.sha256`: hashes of every generated artifact.

## Rebuild and validate

The builder uses only the Python standard library. It expects a local checkout of the
audited dataset by default; override it with `--dataset-root` when needed.

```bash
python scripts/build_splits.py --dataset-root /path/to/FrontierOR_Audited_180_HF
python scripts/validate_splits.py
```

Source revision:
[`a6fe77d0c79184bbea1e8f72ca6efd1a75eec1cf`](https://huggingface.co/datasets/LeoJiangOR/FrontierOR-Audited-180/tree/a6fe77d0c79184bbea1e8f72ca6efd1a75eec1cf).

## Optional large-replica split

The five `large_1` through `large_5` files are replicas, not reliable ordered scale
levels. A three-train/two-test split therefore measures replica holdout rather than
strict Scale-OOD. Generate it with:

```bash
python examples/make_custom_split.py \
  --train-count 3 --test-count 2 \
  --incomplete-policy proportional \
  --output my_large_replica_split.csv
```

## License

Split manifests and repository code are released under the MIT License. The underlying
FrontierOR task data remains governed by its own dataset license.
