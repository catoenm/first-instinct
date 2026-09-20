# Development-only audit while the comparison runs

This is a bounded snapshot of the **first forecast-only arm, seed 1507**, not the
final six-arm comparison. The calendar and general-transfer predictions and
scores are deliberately absent. The training recipe remains unchanged.

Independent offline reconstruction checked all 40 accepted transactions, both
probability-divergence guard receipts, development predictions against frozen
execution-derived targets, 180 development trajectories, actor token inputs,
checkpoint selection and required stopping rules. Twelve new tests cover false
labels, changed metrics, missing or repeated predictions, uncertainty floors,
structure weighting, inflated task counts and invalid selection/stop histories.
The audit runs no model and executes no new environment branches.

Actual optimizer consumption was **800 forecast presentations from 800 distinct
inputs**, balanced at 160 per training mechanism. Of those inputs, 130 retain
uncertain outcome targets. The prepared pool has 2,280 unique forecasts; this arm
did not consume all of them. General replay contributed 640 presentations from
589 unique questions, including 51 repeats. Another 36 backward presentations
were diagnostic component-gradient measurements, with no optimizer steps.
This forecast-only arm collected no live **training** episodes. Its audited
180 development episodes and the still-sealed final evaluation are separate.

The selected checkpoint was update 40. Its development return was 0.318889
versus 0.262222 at the original start, and expected Brier score was 0.490940
versus 0.641098. General retention macro accuracy was 0.867304 versus 0.870488.
These are checkpoint-selection measurements on one authored development
mechanism. **They do not establish transfer or joint pilot success.** Wait for
all three methods, both seeds, unchanged-control evaluation and recovered-checkpoint
audits before drawing that conclusion.

`outcome-1507-receipts.tar.gz` contains exactly the 25 files in
`snapshot-manifest.json`, selected through a development-only filename allow-list.
Its SHA256 is `776a46693b083154b5430aa365e0e222d55b4e8c57eb8df4df532cd3f528ad35`.
The two audit reports record their own source hashes. `hashes.json` covers the
receipt archive and reports; no weights or final-test metrics are included.

Audit entry points are `tool_lab.expanded_evaluation_audit` and
`tool_lab.expanded_selection_audit`. The former intentionally refuses incomplete
learning arms and never opens a final-test result. It reconstructs probability
losses from saved probabilities and rejects unrecoverable target-probability
underflow rather than silently clipping it. Selected checkpoint bytes still
require the full recovery audit after the pilot finishes.

## Reward-only arm, seed 1507

The same two auditors also passed the completed reward-only arm. It performed
40 accepted updates with no rejection, retained update 20, and correctly recorded
`validation_plateau` after two subsequent non-improving development checks.
All 1,200 live training episodes were independently reconstructed: 3,028 actual
actor decisions across all five mechanisms, 268 context cases and 40 underlying
world-and-goal tasks. This includes the new filesystem and actual SQLite
reservation adapters, their visible token inputs, executed commands, reward
assignments and sampled action likelihoods.

The optimizer consumed 3,028 policy presentations from 1,670 distinct question
identities (1,358 repeats), plus the same 640 replay presentations from 589 unique
questions. Reward-only consumed zero forecast-supervision examples. Its 190
additional diagnostic backward presentations comprise 87 actor, 87 value and
16 replay measurements. They are excluded from optimizer consumption.

At selected update 20, development return was 0.315556 and expected Brier was
0.642762, versus the original 0.262222 and 0.641098. Retention macro accuracy
was 0.871550 versus 0.870488. This is a small development decision improvement
without an improvement in forecast error. Final transfer evidence remains sealed
while other arms are selecting checkpoints; this is not the final comparison.

The reward-only snapshot contains 25 allowed development/training files and is
4,128,581 compressed bytes, SHA256
`d0892614a856046a571023876c55208ccd18ca616530d2da8d925f28ffecea73`.
`reward-1507-hashes.json` and `reward-1507-snapshot-manifest.json` cover its receipts
and audit reports. The earlier forecast-only snapshot remains unchanged.

## Combined arm, seed 1507

The same independent development and selection audits passed the first combined
arm: 40 accepted updates, zero rejections, selected update 40. It executed 1,200
live training episodes across the same 268 cases and 40 world-and-goal tasks.
The optimizer consumed 2,989 policy presentations from 1,643 question identities
(1,346 repeats), 800 distinct forecast inputs, and 640 replay presentations from
589 unique questions. Its 210 diagnostic backward presentations are separate.
The paired forecast and replay schedules match the other methods at this seed.

Selected development return was 0.401667, expected Brier was 0.538458 and
retention macro accuracy was 0.865180. The original values were 0.262222,
0.641098 and 0.870488. Both development decision return and forecast error
improved, within the retention limit. **This remains checkpoint-selection
evidence from one seed.** The second seed and sealed transfer evaluations are
needed before applying the prospective joint advancement gate.

Its 25-file development snapshot is 4,088,176 compressed bytes, SHA256
`6070de43f08cf9bb2c25d1c071fbf8f706006ddb2b052715fa50245fef031a04`.
The `hybrid-1507` manifests and audit reports preserve those receipts separately.
No final-test scores or model weights are included in this partial-study folder.

## Second-seed development

The completed combined and reward-only arms at seed 1609 also passed the same
independent evaluation, consumption, selection and stop audits. Each completed
40 accepted updates without rejection and selected update 40. The final transfer
results remain sealed at the time these development snapshots were taken.

| Arm | Live episodes | Policy presentations / IDs | Forecast presentations / IDs | Replay presentations / IDs | Development return | Expected Brier | Retention accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined / 1609 | 1,200 | 2,956 / 1,629 | 800 / 800 | 640 / 592 | 0.403889 | 0.540015 | 0.868365 |
| Reward-only / 1609 | 1,200 | 2,965 / 1,645 | 0 / 0 | 640 / 592 | 0.402778 | 0.635309 | 0.866773 |

Each arm covers all five training mechanisms, 268 cases and 40 world-and-goal
tasks. The per-arm manifests and hashes bind each 25-file snapshot to its audits.
Diagnostic backward presentations remain separate from optimizer consumption.
These are development results used for selection, not evidence of transfer.
