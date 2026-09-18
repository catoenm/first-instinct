# Outcome decisions v2: prospective training protocol

This experiment is separate from the completed sensor-world comparison. Its
purpose is to improve a small general typed-question model through verified
outcomes, not to reconstruct Jev's undisclosed training recipe. No new-model
held-out predictions informed this protocol. A checksum freeze accompanies the
run before model inference on these evaluation worlds.

## Deployed behavior and question contract

Keep the selected Qwen3.5-9B supervised adapter (step 2,742) as the common start.
The foundation revision remains `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
The starting adapter file SHA-256 is
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.
Internal rank-16 language adapters train; the original foundation matrices stay
frozen. The same language network scores user-described choices and forecasts.
There is no generated rationale or classifier reading generated text.

For every available action, ask two independent typed questions: the categorical
terminal outcome and the remaining action cost. Both specify the offered action,
deadline, visible history and a fixed, executable continuation. One freshly
sampled conditional rollout supplies the outcome and cost labels together.
Different actions receive independent hidden draws conditional on the same
visible state. Exact probabilities remain verifier-only.

The primary controller computes expected terminal utility minus expected future
cost and takes the highest-value action. It repeats this process after the next
observation. Marginal distributions suffice because utility is additive. A
single success probability would omit the different penalties of missed and
duplicate jobs. A stop-now continuation would fail to value useful inspection.

These forecasts still describe the stated fixed continuation. They are **not**
end-to-end success probabilities for the controller that repeatedly replans.
Evaluate that controller by executing its actual full trajectory. Questions are
separate full-state forwards, batched when possible; this is not an implementation
of Jev's private shared-state architecture or a claim of matching its latency.
Public singleton menus are answered deterministically without model inference.

## New data and split ownership

Two authored executable mechanisms supply the data:

- [SQLite retry](retry-v2-design.md): ambiguous acknowledgements, delayed delivery,
  stale receipts, idempotency and duplicate side effects. Outcomes are zero,
  exactly one or two completed jobs; utilities are −100, +100 and −200 cents.
- [Workshop workflow](workflow-environment-design.md): inspection, replacement,
  prerequisites, assembly, submission and irreversible casing damage. Terminal
  assertions distinguish completed, damaged and unfinished work.

Each environment reserves two entire factor combinations for validation and two
for testing; four combinations train. Every individual factor and every factor
pair occurs in training. Related cost/prior/seed variants retain their mechanism
owner. The retry validation/test sets both include keyed and unkeyed requests;
the workflow sets both include reusable and consumable casings. Some factors
are asymmetric across holdouts, as detailed in their design documents.

The independent exploration pool has **8,192 training root worlds**, 128 validation
roots and 256 test roots. One uniformly selected visited nonterminal state per
root provides forecasts for every currently available action. Random exploration
records failures as well as successes; it never queries an oracle for an action.
There are **59,993 training questions / 42,897,203 input tokens**, 933 validation
questions and 1,900 test questions. All training outcomes are realized execution
labels. Known singleton costs contribute no training question. The longest
prepared input is 933 tokens; prompts are never truncated.

All 8,576 exploration trajectories and their independent forecast rollouts were
regenerated; every data-file checksum reproduced. Semantic tests separately
check what those executions mean. The prepared-data manifest also pins a
4,096-row reservoir from the previous **training** split for general replay,
and 622 **validation** questions across 157 tasks for retention. Previously
inspected general test/prose results remain historical evidence, not a newly
untouched benchmark.

## Matched controls

Run seeds **77 and 83**, starting from the identical supervised adapter:

| Arm | Language-network objectives |
| --- | --- |
| Outcome | Categorical outcome log loss + future-cost log loss + general replay |
| Reward | Clipped Proximal Policy Optimization + entropy + general replay |
| Hybrid | The same reward objectives + the same outcome and future-cost losses |

All arms use language learning rate **0.000003**, AdamW without weight decay,
up to 60 rollout updates and two optimizer epochs per update. Each update takes
32 distinct forecast roots from a seeded shuffled pool and 32 general replay
rows. Outcome and hybrid use exactly the same labels/order/reuse for a given
seed. The reward arm receives no outcome labels in optimization. The two reward
arms sample 16 training episodes per environment per update on paired root tapes.
Policy-chosen actions can differ after learning. Reusing an outcome in two epochs
does not create two independent labels. Record actual roots, labels, transitions,
optimizer steps, prompt tokens and model calls; hybrid has additional simulator
and supervision work, not an equal-total-data budget.

Outcome log loss has weight 1, future-cost log loss 0.25, general replay 0.25,
entropy 0.01 and critic squared-error loss 0.5. Returns use reward cents divided
by 100, with undiscounted finite-horizon returns and full-rollout standardized
advantages. Proximal clipping is 0.2. Microbatch 16 and the existing 1,536-token
limit are defaults; reduce only the microbatch if memory requires it, preserving
objective averaging and logging the change.

The value head reads **detached** language features and has learning rate 0.0001.
Its loss cannot alter language adapters. Language and critic gradients are
clipped separately to norm 1. Before the first update, audit full-batch pure
actor/outcome/cost/replay gradients and their pairwise cosines, verify zero
critic-to-language gradient, and verify unchanged-policy ratios in actual
training mode. The diagnostic performs no optimizer step. Outcome-only also
collects one clearly labeled diagnostic policy batch; it does not optimize it.

After every committed optimizer step, rescore the full rollout distribution.
If mean old-policy divergence exceeds 0.02, stop optimization and preserve the
actual changed weights and partial-epoch receipt. This guards local steps, not
cumulative drift. A committed step is logged before interruptible diagnostics.

## Selection, stopping and evaluation

The sole primary selection metric for every arm is **forecast-controller
validation reward**, macro-averaged equally across the two environments. The
same controller, question budget and cost rules apply to every arm. Validation
executes 32 root tapes per environment and audits the fixed independent
64-root-per-environment forecast stream. Check at update zero, every 20 updates,
and the last committed update. Update zero remains a legitimate selection.

A new selection must also retain general-validation macro accuracy within
3 percentage points and macro log loss within 0.10 of the starting adapter.
These are engineering gates, not guarantees of retained ability. If two
successive checks make no eligible improvement and at least 40 updates have
committed, stop the arm. Report unequal actual training lengths honestly; do not
attribute a difference in optimization dose solely to the presence of reward.
Each process also has a two-hour bound, including its evaluation.

Lock selection before first test inference. Evaluate selected and latest weights
on 128 new test root tapes per environment, plus the independent test forecast
stream. Compare native argmax actions, the primary forecast controller, the
fixed continuation and a controller using exact conditional probabilities. The
last is one-step rollout improvement with replanning, **not a fully optimal
planner**. All four execute on the same sampled root tapes. Scores are realized
returns, not exact latent-averaged policy expectations.

Report per-environment reward, costs, terminal outcomes, independent multiclass
Brier/log loss, exact-distribution error and utility error. Keep root-level
traces for paired bootstrap intervals. Roots within the same mechanism remain
related; intervals over root draws cannot establish transfer to other mechanisms.
Two training seeds are a small reproducibility check. Test results do not select
checkpoints, learning rates, objective weights or environment definitions.

## Compute and scope

Use personal Runpod only. No Phantom resources or configured model API endpoint.
The previous phase used estimated compute of $64.27; the overall authorized cap
remains $500. This experiment has an **additional $100 ceiling**, including setup,
evaluation, recovery and storage. Rent at most two H200 devices, check total rate
is at most $10/hour, and enforce an eight-hour provider stop deadline both
locally and on the rental. At that rate, scheduled compute is at most $80,
leaving storage/recovery margin. No automatic balance top-up.

Two workers may run one seed each, with outcome/reward/hybrid arms sequentially.
Any mechanical/source/numerical failure ends the pipeline, saves artifacts and
triggers collection; no automatic hyperparameter search or replacement rental.
Verify recovered files before deleting the rental. Keep the existing supervised
demo until a new checkpoint earns selection and its final evidence is reviewed.

## Reproduce the data and run

The new environment data needs only Python and this checkout:

```bash
python -m general_lab.outcome_data --output output/outcome-data-v2
python -m unittest test_retry_v2 test_workflow_environment test_outcome_data -v
```

Preparing model inputs also needs the pinned tokenizer and the previous
[general training mixture](general-training-v1-protocol.md), including its
training/validation files and manifest. Those prior prepared inputs are not
silently replaced with test examples:

```bash
python -m general_lab.outcome_prepare \
  --data output/outcome-data-v2 \
  --replay-data output/general-qwen35-9b-v2 \
  --output output/outcome-prepared-v2
```

On a configured graphics-processor machine, one arm is:

```bash
python -m general_lab.outcome_train \
  --adapter /path/to/supervised/best \
  --data output/outcome-prepared-v2 --raw-data output/outcome-data-v2 \
  --freeze results/outcome-v2/freeze.json \
  --arm hybrid --seed 77 --output output/outcome-v2/runs/hybrid-s77
```

The [freeze](../results/outcome-v2/freeze.json) verifies source, protocol,
prepared-data manifest and every starting-adapter file before loading the model.
Use the exact starting artifacts for a frozen reproduction. A different dataset
or checkpoint is a separate experiment. The cloud runner
`scripts/run_outcome_experiment.py --help` describes the two-worker launch and
recovery archive; it does not provision infrastructure or replace the independent
provider stop guard. `general_lab.outcome_monitor` exports training and validation
metrics to TensorBoard; test results are excluded from live charts.
