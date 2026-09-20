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
