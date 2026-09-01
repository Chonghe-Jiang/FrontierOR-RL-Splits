# FrontierOR RL Splits

Four reproducible train/test protocols for reinforcement-learning experiments on
[FrontierOR-Audited-180](https://huggingface.co/datasets/LeoJiangOR/FrontierOR-Audited-180).

Open **[the visual guide](https://chonghe-jiang.github.io/FrontierOR-RL-Splits/)**
for the motivation, task-partition rationale, exact instance counts, and runtime policy.
The guide also explains why a three-train/two-test split across the five large replicas
is an instance holdout rather than strict Scale-OOD, and shows how to build custom splits.

## Published protocols

| ID | Train | Test | Generalization question |
|---|---|---|---|
| `scale_ood` | Small instances, all tasks | Large instances, same tasks | Scale extrapolation |
| `task_ood_full` | All instances, 150 tasks | All instances, 30 held-out tasks | New-task transfer with full data |
| `task_ood_low_resource` | Two instances per train task | Two instances per held-out task | New-task transfer with limited data |
| `joint_ood` | Small instances, 150 tasks | Large instances, 30 held-out tasks | Simultaneous task and scale OOD |

The task partition is deterministic and stratified over problem family, formulation,
application field, optimization direction, runtime quartile, model-size quartile, and
publication era. The `earl2005` / `ostrowski2012` alias pair is kept together in train.
Every problem type represented in test retains at least one train exemplar.

## Runtime policy

Each instance has one role-independent execution limit derived from its best-available
Gurobi runtime metadata: 60, 120, 300, 600, or 900 seconds. The hard maximum is
**900 seconds (15 minutes) per rollout/evaluation attempt**. The source runtimes combine
current reruns and provenance-labeled historical measurements, so coarse tiers are used
instead of false precision.

## Files

- `index.html`: self-contained visual explanation and interactive task browser.
- `splits/*.json`: complete protocol definitions, summaries, and instance records.
- `splits/*.csv`: flat training/evaluation manifests.
- `splits/task_partition.*`: the fixed 150/30 task boundary and rationale.
- `splits/time_limits.csv`: role-independent time limits for all 1,095 instances.
- `data/task_catalog.csv`: task taxonomy, split assignment, and difficulty bins.
- `examples/make_custom_split.py`: deterministic large-replica 3/2 split generator.
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

## Evaluation hygiene

- Do not expose held-out Gurobi code or reference solutions to the policy.
- Decide once whether checker source is visible; keep that policy identical across methods.
- Tune on training-side held-out rollouts, not on the 30 final test tasks.
- Aggregate instance results within task first, then average across tasks.
- Report timeout rates and total compute in addition to reward or feasibility.

## License

Split manifests and repository code are released under the MIT License. The underlying
FrontierOR task data remains governed by its own dataset license.
