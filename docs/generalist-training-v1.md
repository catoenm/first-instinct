# A longer supervised run toward a usable decision demo

This is a prospective continuation from the original Qwen3.5-9B supervised
step-2,742 adapter. It reuses the admitted 371,278-question release corpus and
the existing balanced trainer. The separate paired-capacity experiment continues
under its own frozen recipe. Neither experiment changes the selected demo model
without a qualifying result.

## Why spend more time training?

The [previous balanced run](release-balanced-v1-results.md) consumed 10,240
presentations, covering 10,032 unique questions. Forecast scores improved, tool
accuracy barely changed, and general accuracy approached its retention floor.
Only half of that run's presentations were general replay. The next run increases
general replay, lowers the learning rate, and allows sustained improvements in
tool log loss to justify continued learning before the release criteria are met.
These changes are a combined recipe test, not an isolated causal experiment.

## Fixed recipe

- Same original foundation revision and internal adapters; no larger model.
- Up to **3,132 updates / 400,896 presentations**: 112 general, 12 tool, and
  four verified questions per update. These are planned counts, not consumption.
- Nearly one additional pass over the admitted general training set. At most
  one visit per general question, three per tool question and six per verified
  question. Visit every question in a pool before repeating any of them.
  Group-balanced sampling preserves the existing source memberships.
- Learning rate 0.000002, linear warmup for 64 updates, microbatch two, maximum
  complete prompt 4,096 tokens. No evidence is cropped to fit.
- Evaluate the same frozen general, tool, outcome and application development
  cohorts every 128 updates. Preserve all original development/transfer roles.
- Continue ordinary learning for at least 512 updates unless a safety check,
  numerical failure or time limit stops it. After that, stop following three
  evaluations without at least 0.0001 improvement in tool macro log loss.
- General accuracy, general log loss and product-slice retention bounds are
  unchanged and apply from the first evaluation. The original release gates,
  including a three-point tool accuracy improvement and no worsening of each
  declared outcome group's proper scores, remain requirements for promotion.
- Preserve full optimizer/random state and qualify a separate-process restart
  after update one. Reserve 30 minutes inside the training deadline to evaluate
  and save the actual final weights when time expires between checks.

This run uses the **existing admitted corpus**. It does not claim newly collected
worlds, new counterfactual executions, or that teacher tool choices are verified
outcomes. Data prepared, unique questions consumed, repeated presentations and
optimizer updates are reported separately. Reserved transfer is not used to
decide training duration.

## Spending and serving

The user explicitly authorized up to $200 for the next run. Reserve at most $5
for a six-hour inference demonstration and at most $195 for this training,
evaluation and recovery. The earlier paired-capacity rental remains charged to
its previous allocation. Reconcile live prices before allocating; one H200 may
run for at most 36 hours at no more than $5/hour, with the remaining $15 reserved
for storage and recovery. Setup, evaluation and recovery fit inside that deadline.
An independent provider shutdown guard must be installed before model setup.

The demo initially serves the already released checkpoint on a separate 48 GB
GPU through a local SSH tunnel. It accepts arbitrary questions and supplied
answers. The Mac runs no foundation model. This temporary demonstration is not
an authorization for indefinite hosting.

Prepare with `python -m release_lab.balanced_plan --recipe generalist-training-v1
--output output/generalist-training-v1-data`. Preparation streams token rows to
keep local memory bounded. Launch only after the resulting schedule, source
hashes, actual-device gradients, and restart/recovery checks pass.
