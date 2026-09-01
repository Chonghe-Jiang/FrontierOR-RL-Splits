# Split design and evaluation semantics

## Three data roles

- **Train** may be used for learning, prompt development, and reward design.
- **Holdout** is validation data for hyperparameter tuning, checkpoint selection, and
  other model-selection decisions.
- **Final test** is used only after the full evaluation configuration is frozen.

For task-based protocols, the existing 30 final-test tasks were not changed. We selected 20 holdout tasks only
from the former 150-task training pool, producing 130 train, 20 holdout, and 30 final-test
tasks. This prevents repeated final-test queries from becoming an undeclared tuning loop.

## What the four protocols measure

### 1. Scale-OOD

- Train: all 197 `tiny_instance*.json` files across all 180 tasks.
- Holdout: one median-Gurobi-runtime large replica per task, 180 total.
- Final test: the remaining large replicas, 718 total.
- Interpretation: size extrapolation with task description, schema, and semantics seen.

The median-runtime rule makes the validation sample representative without selecting
only the easiest or hardest large replica. The files named `large_1` through `large_5`
are replicas, not a reliable increasing-size sequence. A 3/2 split across them is useful
as a replica holdout, but it is not strict Scale-OOD.

### 2. Task-OOD Full

- Train: all 794 instances in 130 train tasks.
- Holdout: all 120 instances in 20 validation tasks.
- Final test: all 181 instances in 30 unseen tasks.
- Interpretation: transfer to unseen tasks when every training-task instance is available.

Counts are not simple multiples of six because 17 tasks have an extra
`tiny_instance_2.json`, while `segundo2019` retains only three large instances.

### 3. Task-OOD Low-Resource

- Train: two instances per train task, 260 total.
- Holdout: two instances per validation task, 40 total.
- Final test: two instances per final-test task, 60 total.
- Selection: canonical `tiny_instance.json` plus the median-runtime large instance.
- Interpretation: new-task transfer under a controlled two-example data budget.

### 4. Joint-OOD

- Train: all 144 small instances in the 130 train tasks.
- Holdout: all 100 large instances in the 20 unseen validation tasks.
- Final test: all 148 large instances in the 30 unseen final-test tasks.
- Interpretation: simultaneous transfer to unseen task specifications and larger instances.

## Task partition construction

The final-test boundary is the previously published deterministic 150/30 grouped,
stratified split. A second deterministic search with seed `20260832` chooses 20 holdout
tasks from the 150-task training side. It minimizes deviation from a 20/150 holdout
proportion over:

1. operational/planning/strategic category;
2. formulation type;
3. application field;
4. minimization/maximization direction;
5. task-median Gurobi runtime quartile;
6. model-size quartile from variable and constraint counts;
7. publication-year bin; and
8. problem classes represented by at least four tasks.

Every problem class, formulation type, and application field present in holdout or final
test retains at least one train example. The `earl2005` / `ostrowski2012` alias pair stays
together in train. Full IDs, constraints, and balance tables are in
`splits/task_partition.json`.

## Why the official evaluation limit is 900 seconds

We analyzed best-available Gurobi runtime metadata over each protocol's combined holdout
and final-test records:

| Protocol | Evaluated instances | Runtime p90 | Runtime >300s | 900s serial maximum |
|---|---:|---:|---:|---:|
| Scale-OOD | 898 | 3,609.59s | 63.9% | 224.50h |
| Task-OOD Full | 301 | 3,603.23s | 56.1% | 75.25h |
| Task-OOD Low-Resource | 100 | 3,601.68s | 34.0% | 25.00h |
| Joint-OOD | 248 | 3,606.86s | 68.1% | 62.00h |

Every p90 lies above 300 seconds, the threshold for the hardest tier. We therefore use
one official limit for every method and protocol: **900 wall-clock seconds per holdout or
test instance**. A timeout is retained as an explicit result.

This 15-minute limit applies to one instance, not an entire manifest. Serial upper bound
is evaluated instance count multiplied by 900 seconds. Parallel execution reduces elapsed
wall time but does not reduce total compute.

The source timing table also contains 60/120/300/600/900-second recommendations. Those
coarse tiers remain useful for training-rollout scheduling, but using them in official
evaluation would give instances different budgets. The merged timings come from current
reruns and provenance-labeled historical measurements on different systems, so they are
difficulty indicators rather than a controlled solver-speed benchmark.

## Evaluation procedure

1. Use holdout results for every model-selection decision.
2. Freeze weights, prompts, reward logic, decoding parameters, checker visibility, and
   all other inference settings before using final-test results.
3. Give each evaluated instance one primary attempt, with a 900-second wall-clock timeout.
   If repeated seeds are reported, use the same repeat count for every compared method.
4. Validate solution syntax/schema, then run the task's checker.
5. Preserve malformed outputs, checker failures, and timeouts as explicit failures.
6. Report checker-accepted feasibility rate, objective quality, timeout rate,
   checker-failure rate, and wall-clock compute.
7. Aggregate instance results within task first and macro-average across tasks so tasks
   with extra instances do not receive additional weight.

The repository does not impose a single composite score. Feasibility and objective
quality answer different questions and should remain separately visible. If a study adds
a normalized objective-gap score, it must state the minimization/maximization convention,
zero-reference handling, clipping policy, and treatment of references without optimality
certificates.

Because this repository publishes its manifests, the holdout/final-test distinction is
procedural rather than cryptographic. A competition can reuse the same IDs against a
private copy of the instances or references to implement a genuinely hidden final test.

## Leakage and reporting rules

- Do not expose holdout/final-test Gurobi code or reference solutions to the policy.
- State whether checker source is visible, and use the same policy for every method.
- Do not silently exclude timed-out, malformed, or checker-rejected attempts.
- Report the number of attempts and total compute alongside quality metrics.
- A reference solution passing its checker establishes usable reward/evaluation data,
  not necessarily a proof of global optimality for every instance.
