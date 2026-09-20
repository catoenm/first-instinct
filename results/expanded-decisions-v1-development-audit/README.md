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
