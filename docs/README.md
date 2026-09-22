# Project guide

Start with the [project README](../README.md) for the model and quickstart.
This page separates the current work from earlier experiments. Detailed reports
and their original results remain at their existing paths.

## Use and understand the current model

- [Demo, command-line interface, and hardware requirements](general-demo.md)
- [9B supervised model: measurements and limitations](general-supervised-results.md)
- [General training protocol](general-training-v1-protocol.md) and [data sources](general-data-sources.md)
- [Completed reinforcement-learning comparison](general-reinforcement-results.md)
- [Training monitoring](training-monitor.md)
- [Release candidate requirements](release-candidate-v1-plan.md)

## Active decision and forecast work

| Question | Read |
| --- | --- |
| How are we training toward the working demo? | [Broader generalist continuation](generalist-training-v1.md), using the existing release corpus |
| What is the longer supervised experiment testing? | [Paired-capacity run](paired-capacity-v1-launch.md) and [protocol](paired-capacity-v1-protocol.md) |
| What data is admitted, and what has only been prepared? | [Paired-data readiness](paired-training-v1-readiness.md) and [curriculum census](paired-curriculum-v1-results.md) |
| Did the previous learning runs improve decisions? | [Capacity comparison](oracle-capacity-probe-v2-results.md) and [final checkpoint evaluations](oracle-capacity-completion-v1-results.md) |
| How do we derive decision and continuation labels? | [Execution-derived oracle](revisioned-oracle-v1-results.md) |
| Can clearer tool descriptions help? | [Controlled tool-description comparison](report-contract-paired-v1-results.md) |
| What happened on unfamiliar mechanisms? | [Expanded curriculum results](expanded-decisions-v1-results.md) |

The current training entry point is
[`tool_lab.paired_capacity_train`](../tool_lab/paired_capacity_train.py).
Its [loss consumer](../tool_lab/paired_learning.py),
[guarded update](../tool_lab/paired_update.py), and
[consumption accounting](../tool_lab/paired_capacity_accounting.py) separate
verified targets, accepted updates, and repeated presentations.

## Environments and data

- **Shell and database:** [Harbor adapter](../tool_lab/README.md),
  [costly evidence](evidence-decisions-v2-results.md),
  [paired decision curriculum](decision-curriculum-v3-results.md),
  [concurrent database changes](revisioned-live-v1-results.md)
- **Filesystems:** [Mutation scope and hard-link side effects](filesystem-decisions-v1-results.md)
- **Applications:** [ToolSandbox dependency chains](application-curriculum-v1-results.md),
  [AppWorld qualification](appworld-local-v1-results.md),
  [AppWorld transfer evaluation](appworld-transfer-evaluation-v2-results.md)
- **Retail:** [Independent financial verification](retail-ledger-v1-results.md),
  [evidence and prerequisites](retail-evidence-v1-results.md),
  [paired question admission](release-retail-questions-v1-results.md)
- **Games and reservations:** [PufferLib integration](../puffer_lab/README.md),
  [game training results](mixed-game-training-v1-results.md)
- **Broader release data:** [Tool-choice admission](release-tool-data-v1-results.md),
  [combined corpus](release-mixture-v1-results.md),
  [balanced training results](release-balanced-v1-results.md)

## Earlier experiments

| Area | Guide | Implementation |
| --- | --- | --- |
| Software outcomes and learned inspection | [Model and results](software-outcome-model.md), [data factory](software-inspection.md) | [`inspection_lab/`](../inspection_lab/) |
| Executable evidence and verifier limitations | [Experiment](executable-evidence.md) | [`evidence_lab/`](../evidence_lab/) |
| Probability calibration and reward learning | [Calibration results](calibration-results.md), [size/data comparison](ppo-data-results.md) | [`calibration_lab/`](../calibration_lab/) |
| Early text encoder and scorer | [Architecture walkthrough](how-it-works.md), [multi-task results](multitask-experiment.md) | [`decision_model.py`](../decision_model.py), [`multitask_train.py`](../multitask_train.py) |
| Document deduplication exercises | [Learning walkthrough](how-it-works.md#learning-path) | [`pipeline.py`](../pipeline.py), [`minhash.py`](../minhash.py), [`lsh.py`](../lsh.py) |

The encoder walkthrough describes the early model; the current 9B implementation
scores allowed labels with a pretrained language model.

## Reproducing and extending experiments

Published reports link to exact inputs, checkpoints, hashes, and source snapshots.
Reproduce a historical run with its saved bundle or recorded Git revision.
Current code can evolve without rewriting old receipts or weakening their hash
checks. Prepared questions, executed branches, repeated presentations, and
optimizer updates are different units; reports retain those distinctions.

All tests now live in `tests/`. Run the suite with
`python -m unittest discover -s tests -t . -p 'test_*.py' -v`, or one module with
`python -m unittest tests.test_general_interface -v`. Old commands that name a
root-level `test_*` module need the `tests.` prefix on current checkouts.
Historical receipt tests read their recorded source revision from local Git
history; use a full checkout for those checks. They never rewrite old hashes.

Keep the root README focused on using the project. Put detailed findings in the
relevant report, update this index for major experiments, and reuse existing
training, evaluation, and accounting components before adding another variant.
