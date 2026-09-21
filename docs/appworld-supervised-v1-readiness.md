# A qualified application-data pilot

**Status update:** v1 stopped before training because its checkpoint identity
check compared two hash orderings. The [separate v2 correction](appworld-supervised-v2-correction.md)
preserves the data and learning recipe, qualifies the canonical identity locally,
and shares the original budget. The preparation evidence below remains valid;
it is not a record of consumed training data.

The next experiment keeps the original Qwen3.5-9B supervised checkpoint and asks
whether broader, executed application examples improve both decisions and
consequence forecasts. It follows the failed joint transfer gate in
[expanded-decisions-v1](expanded-decisions-v1-results.md) and the
[forecast-to-selection diagnostic](forecast-selector-v1-results.md). It does not
restart either study or expand the model's parameter count.

## What the new data represents

Ten application task programs passed our execution controls. The teaching slice
uses their observed application histories and supplies explicit alternative
continuations. It includes stopping, omitted prerequisites, wrong targets,
invalid sessions, successful retries, redundant reads, and plausible but wrong
terminal answers. Different stated costs can make stopping the best decision.

| Quantity | Count | Meaning |
| --- | ---: | --- |
| Underlying task programs | 10 | Eight training programs; two phone development programs |
| Task instances executed | 29 | Different worlds, not question or cost variants |
| Observed history points | 49 | Histories from those task instances |
| Distinct alternative branches | 210 | 132 negative, 69 successful recovery/redundancy, nine wrong-answer controls |
| Independent alternative replays | 210 | Exact independent re-execution, not new tasks |
| Reference, reference replay and collateral-control executions | 87 | Three controls per task instance |
| Total world executions in this collection | 507 | Excludes earlier substrate qualification |
| Raw question variants | 768 | Before grouping and context admission |
| Unique questions before context admission | 712 | Equal inputs grouped before decision labeling |
| Prepared questions | 641 | 484 training and 157 development |
| Task instances retained in prepared questions | 25 | Twenty training and five development instances |

Question counts include different predicates, supplied menus and costs over the
same executed worlds. They are not 641 independent tasks. Earlier substrate
screening qualified ten of fifteen inspected programs; five remain quarantined
under our stricter exception policy. All seven reserved payment-related training
programs, and the upstream development and test sets, remain unused.

## Why these labels are useful

For the same visible history, the model predicts whether a supplied continuation
will complete the task, whether it will change stored application records, or
which supplied plan is best under a stated cost. Completion is checked against
task assertions and preservation of unrelated applications. Record changes are
checked directly against separately executed stopping states.

These are predictions about **the whole supplied continuation**, not immediate
success of its first command. This version does not measure the value of another
observation under adaptive replanning. It also does not test an independent
command proposer: the supplied plans are assisted by demonstrations, with every
argument checked against public evidence at the time it becomes available.

All retained alternatives have independent exact replays. No model prediction
provides a label. Verifier exceptions are rejected rather than treated as failed
tasks. The deliberate direct-database collateral fault is caught by independent
SQLite snapshots; the upstream cached record hashes miss this out-of-interface
injection. That finding is specific to our fault-injection contract.

The initial negative alternatives all failed. We therefore qualified 69 successful
non-reference alternatives and nine same-cost wrong-answer controls before any
model call. A fixed suite of baselines that ignore the goal and history still
leaves mean regret of 0.0886 on training and 0.0816 on development, above the
prospective 0.03 floor. This excludes those simple rules, not every possible
shortcut.

Long histories use a lossless record-table encoding with an exact decode check.
Questions exceeding 8,192 tokens are rejected, never cropped. The longest
admitted question is 8,172 tokens. All ten programs retain questions after these
checks. Whole programs are kept together across splits.

## What the bounded training run will test

The frozen recipe is in [the training protocol](appworld-supervised-v1-protocol.md).
Each update presents eight new forecasts, four new decisions, four older verified
forecasts and eight general-task replay questions. There are 298 new forecast
questions, 186 new decisions, 2,280 older forecast questions and 4,096 replay
questions in the available training pools. The older forecasts preserve 380
genuinely ambiguous inputs; the new application slice adds no stochastic labels.

The parent is the original supervised step-2742 adapter, not a checkpoint selected
from a failed reinforcement-learning arm. Rank-16 adapters inside the language
network remain trainable. Foundation matrices stay frozen. Before baseline
evaluation, one longest-context backward diagnostic must preserve the original
weights exactly and pass finite-gradient and memory checks.

Training is capped at 80 updates and two hours including model loading and
evaluation. The separate H200 rental has a four-hour automatic stop and a $30
allocation from the original budget. Development checks run every ten updates;
retention failure or two non-improving checks stop the run early. The joint gate
requires a 0.03 gain in decision return and a 0.02 reduction in continued-success
expected Brier score, alongside general-task and older-forecast retention.
State-change forecasting is reported separately.

The two phone programs are development data used for selection. Passing this
pilot would justify a further transfer experiment, not establish Jev parity or
general tool-use ability. No checkpoint is automatically promoted to the demo.

The [aggregate readiness receipt](../results/appworld-supervised-v1-readiness/summary.json)
records the frozen preparation counts and hashes. Its zero training-consumption
fields describe the preparation snapshot. Actual accepted updates, repeated
backward presentations, distinct questions and input tokens must come from the
run's consumption and optimizer ledgers. Protected task material, derived
questions, traces and prediction files remain private; the public repository
contains generic code, hashes and permitted aggregates.

## Infrastructure attempt and audit

The first cloud setup failed its dependency check because the pinned base image's
`pygobject` package required missing `pycairo`. It stopped before loading the
foundation model or performing any optimizer update. All 200 archived files were
recovered and verified, and the rental was deleted. Its conservative observed
compute estimate is $0.41, excluding storage; $1 is held until billing settles.

A single infrastructure retry restores the Cairo installation from the earlier
successful environment, pinning its recorded version, 1.29.1. The frozen dataset,
model parent, training sources and learning gates are unchanged. Both attempts
share the original $30 allocation; the retry deadline is earlier, not extended.
This is a setup correction, not evidence about whether the learning recipe works.

After verified recovery, run the independent aggregate auditor with the recovered
directory and original adapter:

```sh
python -m tool_lab.supervised_pilot_audit \
  --root output/appworld-supervised-v2-cloud-v1 \
  --original output/general-supervised-complete-v1/runs/supervised-01/best/adapter_model.safetensors \
  --output output/appworld-supervised-v2-final-audit.json
```

It recalculates probability scores and checkpoint selection, checks the actual
backward/optimizer ledgers against the frozen questions, and compares saved
adapter tensor payloads. It requires a completed learning run; incomplete runs
retain their own failure receipts and cannot silently become completed results.
