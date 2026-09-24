# Decision-focused supervision

The [fixed comparison](decision-supervision-v1-protocol.md) has reached actual
training after the two startup failures documented below. The corrected source
bundle passed 32 tests locally and on the H100, including the complete calendar
evaluation through the decision policy. The original Qwen3.5-9B step-2742 adapter
passed forward/backward qualification, then completed every baseline panel.

An independent audit reconstructed all 128 workflow episodes, both database
question panels, report forecasts, and both general evaluation panels:

| Baseline measure | Result |
| --- | ---: |
| Database completion, 12 episodes | 41.7% |
| Report completion, 36 episodes | 50.0% |
| Calendar completion, 80 episodes | 66.7% |
| Equal-mechanism completion | **52.8%** |
| General accuracy, 3,465 questions | **87.2%** |

Completion rates preserve each mechanism's existing weighting; they are not
pooled raw episode counts. These previously exposed development cohorts do not
establish fresh transfer.

A copied learning ledger independently confirms the first accepted optimizer
update: **48 decision-teacher and 32 general-replay presentations**. The next
update was in progress when that snapshot was taken; its partial presentations
are recorded separately. The teacher pool still contains 742 canonical
questions, and forecast-training presentations remain zero. The maximum planned
dose is 128 updates, with the first candidate measurement at update 32.

**There is no measured improvement yet.** The original checkpoint remains
selected until a trained candidate passes the fixed completion and safety
gates. The corrected restart has a $48 ceiling and twelve-hour hard stop within
the existing allowance; historical failures and artifacts remain preserved.

[Audited baseline and first-update receipt](../results/decision-supervision-v1/connected-start.json).

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
