# Decision supervision: startup failure, no training result

The [fixed decision-focused comparison](decision-supervision-v1-protocol.md)
did not reach training. The H100 attempt loaded the original Qwen3.5-9B adapter
and passed its actual forward/backward qualification, then failed during the
baseline calendar evaluation. It completed **zero optimizer updates** and
consumed **zero training presentations**. The original step-2742 checkpoint
remains selected. This attempt provides no evidence for or against the proposed
learning intervention.

## What failed

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

## Recovery and correction

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
experiment is closed without an automatic replacement or budget extension.

[Aggregate audit receipt](../results/decision-supervision-v1/summary.json).
