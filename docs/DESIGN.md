# Split design and evaluation semantics

## What the protocols measure

Scale and task generalization are separate axes. A model may learn an effective policy
for a task yet fail when its instance grows, or it may handle larger instances of a
known formulation while failing to transfer to a new formulation. The four protocols
make those outcomes distinguishable.

### 1. Scale-OOD

- Train: every `tiny_instance*.json` in all 180 task directories.
- Test: every `large_instance*.json` in the same task directories.
- Current size: 197 train instances and 898 test instances.
- Interpretation: size extrapolation with task description, schema, and semantics seen.

This is not a new-task test. It intentionally asks whether experience acquired from
small examples of a known problem transfers to computationally larger examples.

### 2. Task-OOD Full

- Train: all 914 available instances in 150 train tasks.
- Test: all 181 available instances in 30 held-out tasks.
- Interpretation: transfer to unseen tasks when training uses every available instance.

The exact counts are not `150 × 6` and `30 × 6` because the audited release contains
17 extra `tiny_instance_2.json` files and `segundo2019` has only three retained large
instances. No synthetic padding or silent dropping is used.

### 3. Task-OOD Low-Resource

- Train: exactly two instances per train task, 300 total.
- Test: exactly two instances per held-out task, 60 total.
- Selection: the canonical `tiny_instance.json` plus the median-runtime large instance.
- Interpretation: task transfer under a controlled, low-data budget.

The large instance is selected by the within-task median Gurobi runtime, with the
instance name as a deterministic tie-breaker. This avoids selecting only the easiest or
hardest large case.

### 4. Joint-OOD

- Train: all 164 small instances belonging to the 150 train tasks.
- Test: all 148 large instances belonging to the 30 held-out tasks.
- Interpretation: simultaneous new-task and size extrapolation.

This is the hardest protocol and intentionally compounds the two distribution shifts.

## Why these 30 test tasks

The task boundary is a grouped, stratified split—not a leave-entire-family-out split.
It measures transfer to new task specifications while retaining train exemplars for
every problem class, formulation type, and application field that occurs in test.

The deterministic optimizer minimizes deviation from a 1/6 test proportion over:

1. operational/planning/strategic category;
2. formulation type;
3. application field;
4. minimization/maximization direction;
5. quartile of median per-task Gurobi runtime;
6. quartile of model size derived from variables and constraints;
7. publication-year bin; and
8. problem classes represented by at least four tasks.

Singleton problem classes, formulations, and application fields remain in train. The
optimizer also enforces at least one train exemplar for every test-side type. The
`earl2005` and `ostrowski2012` directories represent one benchmark identity and are
forced into train together. This produces 150 directories in train and 30 in test while
preventing alias leakage.

The full feature-balance table and selected IDs are in `splits/task_partition.json`.
Generation is deterministic with seed `20260831`.

## Runtime-aware execution policy

The time limit is attached to an instance, not to its role in one split. An instance
therefore receives the same wall-clock allowance wherever it appears.

| Best-available Gurobi runtime | RL execution limit |
|---:|---:|
| 0–5 seconds | 60 seconds |
| >5–30 seconds | 120 seconds |
| >30–120 seconds | 300 seconds |
| >120–300 seconds | 600 seconds |
| >300 seconds or timeout-censored | 900 seconds |

The hard maximum is 900 seconds (15 minutes) per rollout or evaluation attempt. These
are coarse tiers because the underlying metadata combines exact current reruns with
provenance-labeled historical runs from different environments. The tiers use runtime
as a difficulty signal without claiming hardware-normalized precision.

The manifests report `serial_upper_bound_hours` only for compute planning. It is the
sum of per-instance caps, not a requirement to execute serially and not a benchmark
score.

## Recommended RL hygiene

1. Freeze the 30 test tasks before prompt, reward, or hyperparameter tuning.
2. Do not expose held-out reference solutions or Gurobi implementations to the policy.
3. State whether checker source is visible. A hidden checker is safer against reward
   hacking; whichever policy is selected must be identical across methods.
4. Treat timeout as an explicit outcome. Do not silently exclude timed-out instances.
5. Aggregate instance metrics within task, then average across tasks, so directories
   with extra instances do not receive more weight.
6. Report feasibility, objective quality, timeout rate, wall-clock compute, and checker
   failures separately.
7. Use training-side instances or a documented internal validation fold for tuning;
   do not repeatedly query the final 30-task test set.

## What is intentionally not claimed

- The merged Gurobi runtime column is not a controlled solver-speed benchmark.
- Scale-OOD does not measure unseen-problem generalization.
- Task-OOD is not leave-one-problem-family-out; that would be a distinct fifth protocol.
- A reference solution passing its checker establishes usable reward/evaluation data,
  not necessarily a proof of global optimality for every instance.
