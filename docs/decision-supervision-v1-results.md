# Decision-focused supervision

**The model learned the supervised questions, but did not qualify a better
workflow policy.** The [fixed comparison](decision-supervision-v1-protocol.md)
stopped at update 96 when database forecast error exceeded its preset limit.
Completed-task rate fell from 52.8% to 50.0%. The original Qwen3.5-9B step-2742
checkpoint remains selected; the demo has not been promoted to this candidate.

This was decision and inspection supervision with general-task replay. It used
**no reinforcement-learning updates and no forecast-training presentations**.
It does not establish that reinforcement learning cannot work.

[Final aggregate and audit identities](../results/decision-supervision-v1/connected-final.json).
The [baseline/start receipt](../results/decision-supervision-v1/connected-start.json)
and earlier failed-run receipts remain unchanged.

## Measured results

Each measurement executed the same 128 database, report, and calendar episodes,
then scored both database question panels, report forecasts, and general panels.

| Measure | Original | Update 32 | Update 64 | Update 96 |
| --- | ---: | ---: | ---: | ---: |
| Completed tasks, equal weight per mechanism | **52.8%** | 52.8% | 50.0% | **50.0%** |
| Database completion, 12 episodes | 41.7% | 41.7% | 33.3% | 33.3% |
| Report completion, 36 episodes | 50.0% | 50.0% | 50.0% | 50.0% |
| Calendar completion, 80 episodes | 66.7% | 66.7% | 66.7% | 66.7% |
| Database action-question accuracy | 41.2% | 52.5% | 54.9% | **56.6%** |
| Database inspection-question accuracy | 67.6% | 81.4% | 86.3% | 86.6% |
| Database forecast Brier error, lower is better | 0.6248 | 0.6337 | 0.6423 | **0.6471** |
| General macro accuracy, 3,465 questions | 87.21% | 87.30% | 86.96% | 87.15% |
| Mean reward after action costs | 0.1608 | 0.1664 | 0.1672 | 0.1674 |

Rates retain the predefined weighting within each mechanism; the overall
completion measure is not a pooled episode count. Raw completions changed from
5 to 4 out of 12 database episodes; report stayed at 18/36 and calendar at 50/80.
Database question metrics above use the canonical option order. Both orders,
raw counts, retention results, and product slices are in the aggregate.

At update 96, forecast Brier error had increased by **0.02226**, crossing the
allowed increase of 0.02. Learning stopped as specified. No update passed the
completion-improvement gate; general capabilities remained within their safety
bounds. This was a deliberate safety stop before the planned 128 updates,
not an execution failure or an exhausted runtime allowance. It is evidence
against this recipe under its fixed constraints, not against learnability.

## What the failures reveal

The starting database decisions remain a bottleneck. On the six distinct
starting observations, exact optimal-action selection rose from **0/6 to 1/6**
in the canonical order. Mean probability assigned to the optimal action rose
from 16.3% to 39.9%, so learning occurred without usually changing the executed
choice to the best action. Later-state question accuracy improved more.

These starting cases were not absent from training. The consumption audit
confirms **96 presentations of each starting action question**, covering six
cyclic menu orders sixteen times each. Their encoded inputs and correct targets
match the evaluation. The first live decision also has exactly the same input,
token sequence, and recorded probabilities as its corresponding panel question.
That rules out missing starting examples and this particular offline/live
interface mismatch; it does not identify the cause of the remaining errors.

One example explains the completion/return divergence. With a cheap database
read and a goal of incrementing the latest value, the original model immediately
wrote a cached value. That succeeded in one hidden world and corrupted the other.
The candidate stopped in both worlds. It avoided the bad write but also lost a
completion; the execution-derived best action was to read, then continue using
the result. Stopping *was* correct when writing was prohibitively expensive,
and the candidate learned that separate case. Merely suppressing action is not
enough to learn when evidence is worth acquiring.

Option order remains another weakness. Reversing the answer menu changed the
selected action on **17.2% before training and 40.8% afterward** under the
existing weighted database-panel measure. These changes are not all necessarily
large utility losses, but similar aggregate accuracy in both orders hides
substantial disagreement on individual questions. Position rotation alone did
not establish stable selection.

These are post-hoc diagnostics of known development cases, not new selection
rules or fresh evidence of transfer. The next useful learning check is small:
demonstrate reliable selection on these starting states across menu orders,
then verify the resulting continuations while retaining consequence forecasts.
Measure correct-action probability, execution utility, completion, and forecast
error together. A focused comparison can test whether the limiting factor is
optimization strength or interference between objectives before spending on
broader data. It needs a new qualified protocol; this run is closed and has not
been extended, restarted, or retroactively given looser gates.

## Data actually consumed and checkpoint lineage

All **96 optimizer attempts were accepted**. The learning ledger records:

| Unit | Prepared or planned | Actually consumed |
| --- | ---: | ---: |
| Canonical decision/inspection questions | 742 | 742 |
| Teacher presentations | 6,144 planned | 4,608 |
| Distinct teacher position variants | — | 1,594 |
| Repeated presentations of the same position variant | — | 3,014 |
| General replay presentations | 4,096 planned | 3,072, all distinct |
| Forecast-training presentations | 0 planned | 0 |
| Reinforcement-learning updates | 0 planned | 0 |

The questions span six authored families. Questions, menu variants, and repeated
presentations are not independent underlying tasks. No new environments or
counterfactual branches were generated for training in this run. Four complete
measurements executed 512 episodes over the same 128 specifications. The final
receipt explicitly reuses update 96, adding no further evaluation episodes.
Infrastructure qualification executions are excluded from learning counts.

The candidate changed all 496 adapter tensors: 43,278,170 of 43,278,336 trainable
values differ from the parent. Base matrices stayed frozen; the adapters change
effective transformations throughout the language network. The selected
checkpoint is byte-identical to the original. Original and update-96 weights
were recovered; update-32/64 metrics and trajectories were recovered, but their
intermediate weight directories were not recovered locally. Neither qualified.

The final audit verified 821 artifact files, frozen sources/data, recorded
probability metrics, executed receipts, consumption, stopping/selection logic,
and saved adapter arrays. It did not independently rerun foundation inference
or replay optimizer steps. One seed and previously exposed development mechanisms
cannot establish a general mini-Jev release.

## Runtime and cost

The cloud pipeline finished in about **84 minutes**. The H100 then remained
allocated for approximately **nine more hours** while artifact recovery was
delayed. Transfer failures and a timeout are recorded; the precise contribution
of laptop or network availability is not established. This was wasted allocation
after the experiment finished, not additional training.

All artifacts were ultimately verified, and the owned H100 was stopped and
deleted. Posted billing is **$36.93 for this rental** and **$41.45 including the
earlier failed attempts**, within the $48 rental cap and existing $60 comparison
hold. These are posted charges at the recorded billing check, not a new budget.
The protected older stopped volume remains untouched.

Before another rental, durable cloud artifact recovery and stopping the GPU on
completion must be separated from downloading results to the laptop. The
independent maximum-runtime guard remains necessary, but does not by itself
prevent this kind of idle cost.

## Earlier startup failures

The first connected H100 attempt loaded the original adapter and passed its
forward/backward qualification, then failed during baseline calendar evaluation.
It completed **zero optimizer updates** and consumed **zero training
presentations**, providing no evidence for or against the learning intervention.

### Missing timezone asset

The source bundle omitted `tool_lab/calendar_assets/America_New_York.tzif`.
Its builder included tracked Python files and explicitly frozen sources, but
the new experiment's source list did not include this non-Python dependency.
The existing bundle tests checked schedules, update behavior, hashes, and
controller recovery without executing the calendar worker from that bundle.
An incomplete manifest can pass every listed hash check.

The failure happened before the first calendar episode. The partial baseline
contains twelve database and 36 report trajectories, two 2,064-question database
panels, 308 report forecasts, and 622 retention predictions. Calendar and full
general evaluation were not completed, so there is **no valid complete baseline,
candidate comparison, or selected new checkpoint**. Qualification gradients
are not optimizer updates or consumed training examples.

### Recovery and correction

The collector recovered the final archive, verified all 765 recorded files, and
stopped and deleted the owned rental. A separate audit rehashed those files,
the frozen source/data inputs, the original adapter, and the archive. The older
stopped paired-experiment volume remains untouched. Fresh provider inspection
found no running training or inference rental.

The correction adds the pinned timezone asset and its provenance to future
source manifests. Before loading foundation weights, the trainer now executes
all 80 calendar cases through the same live episode adapter and independent
verifier, using a fixed scripted continuation. This is an infrastructure check,
not a model benchmark or additional training data.

A disposable extraction of the original archive reproduced the missing-asset
failure, rejected a corrupted asset, and passed after the correction: 80
episodes, 176 offered actions including stops and prefixes, and 136 actual tool
commands. No model was loaded locally. Historical archives, source snapshots,
receipts, checkpoints, and hashes were preserved. The disposable extraction is
not a newly frozen or launch-qualified training bundle.

Full test discovery passed: 1,189 tests, with 29 skipped. Regression coverage
includes missing and corrupted assets, incomplete cohorts, actual worker
execution, and a trainer check that fails before foundation loading. The guarded
test process peaked at about 1 GB of memory with no added swap.

The rental's compute estimate was about $1.29 excluding storage; provider billing
can arrive later. This remains within the existing comparison allowance. The
attempt closed without an automatic replacement or budget extension.

[Aggregate audit receipt](../results/decision-supervision-v1/summary.json).

### Corrected restart: evaluator interface failure

A subsequent user-authorized restart included the missing asset. Its complete
scripted calendar preflight, dependency setup, 31 startup tests, model loading,
and device qualification passed. Baseline evaluation then failed at the first
calendar policy call. Again, there were **zero optimizer updates, zero training
presentations, and no complete baseline or candidate result**.

The calendar evaluator called a legacy collector directly. That collector
inserts an unused first-option target placeholder; the current decision policy
correctly rejects labeled live actions. The report evaluator already uses an
existing adapter that removes this placeholder and verifies the executed
receipts. Calendar evaluation now uses that same adapter. The policy's rejection
check remains intact.

The earlier tests exercised the environment and model independently but missed
their connection. The added integration test reproduces that rejected call,
then runs the production calendar evaluation function across all 80 cases with
the actual decision-policy class over a tiny test network and a synthetic
tokenizer. It independently audits the executions and encoded inputs, recomputes
the recorded action probabilities, and checks that weights remain unchanged and
no gradients accumulate. This establishes interface compatibility; it does not
measure the 9B model's capabilities.

The same connected test passed from a disposable extraction of the recovered
archive with only the corrected source files substituted. Full test discovery
passed: 1,190 tests, with 29 skipped. No historical freeze was rewritten. At that
point, no new launch bundle had been qualified.

All 770 restart artifacts were recovered and the rental was deleted. Its compute
estimate was about $1.03 excluding storage. No replacement was rented at that
closure. Both failed attempts left the learning comparison unanswered.
Historical failed-run sources and hashes remain unchanged.

[Restart audit receipt](../results/decision-supervision-v1/fixed-startup-failure.json).
