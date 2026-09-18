# Learning command consequences improves forecasts, with mixed control gains

The 9B model learned immediate consequences of database commands from 2,592
verified training questions. On 864 questions from eight withheld public
histories, modal-answer accuracy increased from **81.25% to 99.65%**, and
categorical probability error fell by **90.9%**. Actual decision trajectories
improved in some settings and regressed in others. Accurate one-command
forecasts did not produce a consistently better sequential policy.

**The final audit found exact training/validation prompt overlap:** 216 of the
864 validation rows had token-identical training counterparts despite different
history identifiers. On the remaining 648 rows, accuracy rose from 83.33% to
99.54%, but that filtered result is post-hoc and does not undo overlap's influence
on checkpoint selection. This is a development pilot with a split defect, not
a clean generalization benchmark.

This is supervised training of internal language adapters on one known
mechanism. It is not a reinforcement-learning result, an independent-mechanism
generalization result, or a reconstruction of Jev's private method.

## What was trained and selected

Starting from the released supervised Qwen3.5-9B model, one H100 completed
180 optimizer updates. Each update mixed 24 exact consequence-distribution
targets with eight general replay questions. The starting model, six candidate
checks, retention gates and selection rule were fixed before inference.
Step **120** had the lowest eligible validation excess log loss. Steps 150 and
180 were retained in the evidence and were not selected.

All 496 language-adapter tensors changed, covering 43,278,336 parameters in
attention and feed-forward projections. The foundation matrices and vocabulary
projection stayed frozen. The gradient proof, parameter hashes and an independent
comparison of saved adapter tensors establish that the internal computation was
trained; this was not a fitted temperature or a new output-only classifier.

The run and its evaluations took 21.8 minutes after model-process startup.
Rental compute was approximately **$1.78**, excluding storage and invoice
adjustments. All 47 recovered files passed checksum verification before the
rental was stopped and deleted. No GPU remains rented for this experiment.

## Immediate consequence predictions and general retention

| Measure | Starting model | Selected step 120 |
| --- | ---: | ---: |
| Consequence modal-answer accuracy | 81.25% | 99.65% |
| Consequence excess log loss | 0.43859 | 0.04393 |
| Consequence categorical Brier error | 0.27859 | 0.02540 |
| General retention macro accuracy | 85.40% | 85.72% |
| General retention macro log loss | 0.35726 | 0.37618 |

Consequence metrics average the four event types. A modal answer means an option
with maximum target probability; ties count as correct. This accuracy does not
require the predicted probabilities themselves to be correct. Excess log loss
subtracts the entropy of the exact target distribution. Brier error sums squared
probability deviations across offered outcomes. Lower is better for both errors.
Retention averages task metrics over 622 validation questions. Its log loss
worsened slightly despite similar accuracy; it remained within the prospective
gate of 0.05 additional log loss and two percentage points of accuracy loss.

The 3,456 source rows contain 1,152 event questions in three presentations,
covering 32 public histories and one authored mechanism. Every presentation,
action and event for a given history stays in one split. Training and validation
still share root worlds and transaction rules. Some histories differ only in
public parameters. These are development measurements, not a clean test of
general intelligence or unfamiliar database systems. The corpus was already
inspected before this experiment.

The [post-hoc overlap audit](../results/consequence-training-v1/overlap-audit.json)
found why group isolation was insufficient. History identifiers include fee
profiles, while immediate-consequence questions do not display every fee. Two
groups can therefore render the same tokens. The training pool contains 2,268
unique prompts among 2,592 rows; two validation groups account for the 216 exact
matches. Their labels are consistent, but they must not count as unseen inputs.

| Post-hoc subset | Rows | Modal accuracy before → after | Brier error before → after |
| --- | ---: | ---: | ---: |
| Exact training-prompt match | 216 | 75.00% → 100.00% | 0.29184 → 0.00214 |
| No exact training-prompt match | 648 | 83.33% → 99.54% | 0.27417 → 0.03315 |

The selected checkpoint remains step 120; we did not reselect it using this
filter. The remaining rows still share mechanism, worlds and related contexts.
A subsequent study should group connected identical rendered prompts before
splitting, then reserve independent mechanisms for final testing.

## Executed decisions: report every presentation

The same CUDA runtime evaluated the starting and selected model. Each action was
executed in the C environment previously qualified against independent SQLite
transactions. Success and return below are weighted by the disclosed world prior;
the three presentations repeat the same physical cases and are not independent
trials. Return includes fees and penalties for incomplete partial writes.

| Cases | Presentation | Success before → after | Mean return before → after |
| --- | --- | ---: | ---: |
| Familiar, 16 cases | Original | 68.75% → 75.00% | 0.6419 → 0.5838 |
| Familiar | Reversed answers | 68.75% → 75.00% | 0.4000 → 0.4594 |
| Familiar | Reworded | 68.75% → 75.00% | 0.5194 → 0.4538 |
| Combined failures, 8 cases | Original | 12.50% → 25.00% | −0.0506 → −0.3162 |
| Combined failures | Reversed answers | 25.00% → 0.00% | −0.3181 → −0.3294 |
| Combined failures | Reworded | 12.50% → 12.50% | −0.0650 → 0.0525 |
| Fresh parameter mixtures, 12 cases | Original | 37.50% → 79.17% | 0.3200 → 0.7109 |
| Fresh parameter mixtures | Reversed answers | 50.00% → 58.33% | 0.4317 → 0.4987 |
| Fresh parameter mixtures | Reworded | 41.67% → 41.67% | 0.1390 → 0.1375 |

The original combined-failure presentation illustrates the difference between
success rate and reward. Completions increased from one to two, but partial-write
failures increased from one to four, making mean return worse. Selecting the
largest success gain alone would hide this regression.

The fresh profiles use new nonuniform priors, fees and five- or seven-turn
horizons, frozen before the candidate was trained. They still use the same six
worlds and database mechanism. Averaged over the three presentations, their
weighted success rose from 43.06% to 59.72%; the per-presentation table shows how
uneven that gain is. These diagnostics did not select the checkpoint.

All **802 recorded action transitions** (391 before, 411 after) were replayed
against the environment, including public prompts, actions, resulting states,
fees and terminal outcomes. There were 108 episodes per model. No new SQLite
qualification attempts or paid model calls were needed for reporting.

## Forecasting a complete continuation is still difficult

The separate probe asks whether an entire specified continuation succeeds,
rather than predicting state immediately after one command. It contains 39
events queried in both answer orders, with exact targets from earlier executed
branches. It was already an opened development diagnostic before this pilot.

| Probability mean squared error | Starting model | Selected model |
| --- | ---: | ---: |
| All 39 events, original order | 0.32384 | 0.22904 |
| All 39 events, reversed order | 0.28006 | 0.18014 |
| Eight uncertain events, original order | 0.03499 | 0.01394 |
| Eight uncertain events, reversed order | 0.02269 | 0.04089 |

A constant 50% forecast scores 0.20513 over all 39 events. Averaging the two
answer orders, the selected model scores 0.20459—almost the same as that simple
reference. The uncertain subset worsens under answer reversal. This does not
establish generally calibrated probabilities, despite the strong immediate
consequence result. Forecast error combines understanding of the specified
program with probability estimation.

## What this supports next

Verified consequence data can materially change the language model's predictions
without a large rental. It does not by itself teach reliable planning, recovery
from partial writes, or invariance to wording and answer order.

A useful next experiment is a controlled reward-only versus reward-plus-forecast
comparison from the same warmed checkpoint, including a forecast-only control.
That requires newly sampled trajectories, verified terminal rewards, retained
general questions and independent executable mechanisms. Extra labels and
compute must be accounted for separately. The greedy trajectories here should
not be treated as on-policy samples from their recorded probability vectors.
That reinforcement comparison has not been launched by this pilot.

The supervised demo remains the deployment default. The selected consequence
adapter is a research checkpoint, with the regressions above preserved.

## Evidence and reproduction

- [Prospective protocol](consequence-training-v1-protocol.md) and
  [pre-inference freeze](../results/consequence-training-v1/freeze.json).
- [Full computed report](../results/consequence-training-v1/report.json),
  [tensor audit](../results/consequence-training-v1/tensor-audit.json),
  [prompt-overlap audit](../results/consequence-training-v1/overlap-audit.json),
  [collection receipt](../results/consequence-training-v1/collection.json), and
  [raw predictions, training trace and trajectories](../results/consequence-training-v1/evidence/).
- [Research checkpoint and complete recovered archive](https://github.com/catoenm/first-instinct/releases/tag/consequence-learning-v1),
  including selected and latest adapters, tokenized data, runtime records and
  all 47 original files. Keep the license/attribution companion with the archive.

After verifying the release manifest and extracting the original archive into
a new directory, reproduce the report without inference or training:

```sh
python -m puffer_lab.consequence_report \
  --root /path/to/recovered-tree --output /path/to/report.json
```

The source freeze is bound to commit `171a8fe`; reporting was added separately
in `20966a1`. Checkpoint selection was prospective. The interpretation in this
page and the saved-tensor comparison were written after examining results.
