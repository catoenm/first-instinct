# Scaling the verified outcome experiment

**Status: paused before training began.** The user clarified that the primary
goal is a general model for dynamically supplied questions and constrained
outputs, with reinforcement learning updating the language network. The new
rental was deleted before its bootstrap completed. Both model sizes' prepared
inputs were verified identical, and the SWE-smith source data was downloaded;
neither a full-pass scaling run nor a new nine-billion-parameter run was trained.
The [mini-Jev research plan](mini-jev-plan.md) supersedes this as the immediate
direction. This document preserves the proposed narrow comparison for later use.

This protocol was written before starting the next cloud run. The current
GitHub release remains the reference and the local demo remains on that model.
The authorized experiment budget is US$100 including a reserve for storage;
provisioning and automatic stop deadlines are tracked privately. No employer
infrastructure is used.

## First comparison: use the prepared data

Train a fresh Qwen3.5-4B adapter on all 36,189 prepared training views for one
pass (34,571,820 input tokens). Use seed 41, effective batch size 32, the original
length-bucketing algorithm and 2,048-token limit. There are 1,131 updates.
Retain the original learning-rate rule, with its warmup and decay stretched over
the full pass. This compares training regimens; the schedule changes along with
the number of examples consumed. It is not an isolated continuation of the
published checkpoint's optimizer state.

Use the same 256 checkpoint-selection validation views, evaluate every 150
updates and at completion, and select the lowest validation log loss, including
the original foundation. Keep the independent calibration population fixed at
the original six validation source groups. Reuse the existing 128-candidate
ordinary and family test samples for a clearly labeled follow-up comparison.
Those test results are already known; they are not a new untouched benchmark.

Measure real input tokens, distinct views, program variants, source groups,
training time, memory, probability error, accuracy, copied-evidence sensitivity
and workflow reward. Report all completed runs, including regressions. Select
checkpoints on validation loss, never on the test results.

## Model-size comparison

Use pinned Qwen3.5-9B (revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`) with a fresh adapter and the same
effective batch size, adapter configuration, example order and full-pass
schedule. Independently prepare its inputs and verify exact population and token
agreement before describing this as a matched-token comparison. If tokenization
differs, use the common eligible population and disclose the difference.

Benchmark memory and throughput before committing to the full larger run.
Microbatch sizes may differ to fit memory; record them. The initial comparison
has one training seed per size and does not estimate seed-to-seed uncertainty.
Do not replace the demo solely because a model is larger: compare held-out
probability quality, workflow reward and local inference cost first.

## Data expansion and reinforcement learning

Audit public repository-level software tasks, beginning with SWE-smith. A bug
description, generated patch or tool response is not itself a verified outcome.
Reproduce test execution, preserve source licenses and versions, distinguish
visible from hidden evidence, and quarantine unstable or incompatible cases.
Reserve whole repositories and bug generators for evaluation before constructing
training variants. Keep the new repository holdout sealed during development.

Add realistic failure mechanisms and independently written checks. Count source
repositories and original functions separately from mutations and evidence views.
Publish a data audit even if the candidate source proves unsuitable.

The subsequent reinforcement experiment should train acquisition with the chosen
forecaster. Retain direct outcome supervision for probability reports while
using Proximal Policy Optimization for evidence acquisition. Compare with a
frozen forecaster, stopping immediately, a fixed inspection rule and the empirical
planner at matched interaction budgets. Continue scoring common evidence states
alongside the states selected by each policy. This phase is not complete merely
because a larger outcome-supervised adapter has been trained.
