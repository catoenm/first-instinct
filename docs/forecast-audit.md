# Keeping a decision model's forecasts in practice

The previous experiment learned useful decisions while leaving weak probability
reports on states the policy rarely selected. This follow-up tests a data
intervention: keep asking for forecasts independently of which actions the
policy normally chooses.

**Continued practice reduced common-state probability error from 13.87 to
8.63 percentage points under familiar conditions**, a 38% reduction in the
three-seed mean. Giving the same extra examples early achieved 10.82 points.
Both probability error and workflow reward improved in all three seeds versus
chosen-path rewards alone. Event-label training still produced better forecasts.

| Final recipe | Probability error, points ↓ | Change after a copy, points ↓ | Workflow return ↑ |
| :--- | ---: | ---: | ---: |
| Chosen-path rewards only | 13.87 | 12.28 | 0.83907 |
| Extra practice early | 10.82 | 3.94 | 0.84003 |
| Extra practice throughout | **8.63** | **3.41** | **0.84202** |
| Event-label reference, same exercise states | 4.13 | 1.85 | 0.84849 |

These are means across seeds 11, 23 and 37 on 16,384 fresh worlds. The common
audit assigns half its weight to initial observations and half to acquired
observations, integrating over both possible additional readings.
Continued-practice probability error ranged from **8.24 to 9.31 points**;
chosen-path-only error ranged from **13.78 to 14.03**. The label reference's
range was **3.09 to 4.91**. The state counts are shared across models, not
independent examples for each recipe.

![Continued practice improves ordinary-condition forecasts, while the event-label reference remains better. Individual seeds are shown alongside means.](assets/forecast-audit/forecast-practice.png)

The first-quarter checkpoint helps interpret the schedule comparison. Early
practice largely retained its gains: mean probability error was 10.82 points
both at the quarter checkpoint and at the end. Continued practice improved
from 11.70 to 8.63; chosen-path rewards alone worsened from 11.78 to 13.87.
This supports continued practice in this setup, but not a claim that stopping
practice necessarily causes catastrophic forgetting. The early recipe had
already consumed its full exercise budget at the quarter checkpoint, while
the continued recipe had used only one quarter.

## The failures remain useful

Better average forecasts did **not** establish generally reliable probabilities.
The held-out workflow domains show both gains and a reward tradeoff:

| Domain | Chosen-path error | Early-practice error | Continued-practice error | Label-reference error | Continued minus chosen-path return |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Original conditions | 13.87 | 10.82 | 8.63 | 4.13 | +0.00294 |
| Higher inspection prices | 13.96 | 12.32 | 9.61 | 6.03 | +0.00105 |
| 80% copied sources | 14.20 | 9.59 | 8.22 | 3.77 | +0.00076 |
| Reversed additional source | 28.71 | 21.26 | 18.43 | 11.13 | **−0.00268** |

Errors are percentage points; each cell averages the same three seeds.
On reversed sources, continued practice reduced probability error in every
seed but also reduced workflow return in every seed. Return changes for seeds
11, 23 and 37 were **−0.00725, −0.00045 and −0.00034**. The first exceeds our
predeclared 0.005 meaningful-loss threshold even though the mean loss does not.
The only other return loss was −0.00125 for seed 11 at higher prices; the other
eight domain/seed comparisons improved. The reversed conditions never appeared
in this experiment's training generator.

The separate paired benchmark exposes additional inconsistencies:

| Ordinary paired cases, final model | Chosen-path rewards | Early practice | Continued practice | Event labels |
| :--- | ---: | ---: | ---: | ---: |
| Probability error, points ↓ | 14.96 | 11.49 | 8.82 | 5.38 |
| Forecast change after an exact copy, points ↓ | 13.42 | 4.57 | 3.74 | 1.68 |
| Forecast change after a price increase, points ↓ | 4.11 | 6.81 | **5.07** | 6.22 |
| Forecast change when only unseen-source metadata changes, points ↓ | 1.89 | 2.91 | 2.47 | 2.53 |
| Error in the update after fresh evidence, points ↓ | 13.92 | 11.95 | 8.08 | 3.21 |

Continued practice improves use of fresh evidence and copy consistency, but
does not fix sensitivity to prices or unobserved-source metadata. The event-label
reference also fails those invariance checks. On the reversed-source paired
cases, continued practice updates in the **wrong direction on 84.96%** of fresh
readings, versus 98.44% for chosen-path rewards, 81.71% for early practice and
36.20% for the label reference. These are three-seed mean rates over 512 updates
per seed. A lower overall error can coexist with a very poor response to new
evidence. No model was retrained after these results were opened.

The narrower finding is useful: **which forecasts receive practice, and when,
changes the reliability of a decision model's probability interface.** It does
not remove the need for broader training conditions or stronger reference
methods. The data schedule is the intervention; proper scoring rules and
Proximal Policy Optimization are established methods.

## What changed

The inference model is unchanged: a numeric network with **6,166 parameters**,
one distribution over probability reports and one probability of buying another
observation. We changed which forecasts receive practice during training.

The terminal-only learner receives a reward when it actually reports an answer.
It is not directly rewarded for the report it would have given on an initial
state where it instead buys information, or after a copy it normally declines.
The network still produces numbers for those queries, but their quality need
not match performance along its normal execution path.

An extra exercise asks for a report on both the initial state and the acquired
state of a separately generated situation. The answer receives the same
`1 - (report - outcome)^2` reward. The hidden outcome stays in the scorer; no
exact probability target is passed to the reward learner. The extra exercises
do not allow a purchase action and impose no direct gradient on the buy head.
They can still affect acquisition through shared network features.

We compare four recipes:

| Recipe | Interaction training | Extra forecast exercises |
| :--- | :--- | :--- |
| Chosen-path rewards only | Yes | None |
| Early practice | Yes | All during the first quarter |
| Continued practice | Yes | Spread throughout training |
| Event-label reference | No | Same exercise states, trained with labels |

The two practice schedules consume the same **4,000 blocks**, in the same order.
Each block contains 128 potential worlds, with an initial and acquired state
for each: **1,024,000 exercise states** per model. Early practice uses four
blocks per training rollout for the first 1,000 rollouts. Continued practice
uses one block for each of 4,000 rollouts. Both schedules allow four policy
updates per block and decay exploration entropy by block count.
All final runs completed all four updates per block; the matched exercise
budgets were also equal in actual update count.

All reward policies also receive 2,048,000 interaction episodes. Acquisitions
are fixed at 50% for the first quarter, then controlled by the learned policy.
This initial acquisition schedule is distinct from the extra forecast-exercise
schedule. We use Proximal Policy Optimization for both sampled-reward tasks.

The label reference learns from exactly the same exercise observations using
binary log loss. It has a scalar probability output and an engineered planner
that knows a copy adds no information. It is an assisted reference with different
feedback and less total training work, not a clean optimizer-only comparison.

## Reading the measurements

The report policy's 21-way distribution is over possible probability reports,
not over 21 event types. We audit its mean report as an event-probability
estimate. Probability error is root-mean-square distance from the exact event
probability on a common mixture of initial and acquired states.

Workflow return integrates over acquisition, both possible readings and the
report-sampling distribution, including squared error and inspection price.
It therefore includes the extra cost of report sampling. Improving the mean
forecast alone need not improve this return.
The frozen protocol treats a reward loss above 0.005 as a meaningful tradeoff.

Copy change is the average absolute difference between forecasts before and
after an exact copy is revealed. The exact probability does not change. This
checks output consistency on hypothetical queries; it is not a measurement
of an unobserved internal belief or a guarantee about real-world calibration.

## What was fixed before testing

The [protocol](forecast-audit-protocol.md), generators, training implementation
and benchmark were committed before the final run. Three training seeds—11,
23 and 37—use independent streams, matched across recipes.

The last training step is the preselected final model. Validation only records
progress; it never chooses a checkpoint. We also save the first-quarter model
as a prespecified diagnostic. That point marks the end of early practice, while
continued practice still has three quarters of its exercise budget left.

Only after every model is sealed do we open four fresh workflow domains and the
paired benchmark. Every seed, domain and diagnostic checkpoint is retained.
This differs from the earlier study's validation-selected checkpoints, so use
the new experiment's terminal-only learner for its controlled comparisons.

## The probability benchmark

The [benchmark guide](probability-benchmark.md) describes 3,072 numeric situations
and 6,144 text requests. It tests exact copies, fresh evidence, inspection prices,
unobserved source metadata and equivalent wording. The exact event probability
comes from a specified simulator, not another model's judgment.

A price change or duplicate observation should not change a probability forecast.
Fresh independent evidence should update it appropriately. Both checks matter:
a constant guess would pass invariance tests while failing to use information.

The numeric models see only numeric observations. The text requests are for
language-capable systems such as Jev; the numeric models get no credit for
handling paraphrases. No language encoder was trained in this experiment.

## Reproduce the work

All final and first-quarter policies, initial weights, value networks, validation
histories, compressed training traces, generator definitions, text requests and
separate answer files are under
[results/forecast-audit-v1](../results/forecast-audit-v1).

```bash
python -m pip install -r requirements-calibration.txt

# Regenerate the training streams and reconstruct every saved-model result.
python -m calibration_lab.forecast_audit_verify

# Preview a Jev request without making a network call.
python -m calibration_lab.jev_benchmark

# Train the complete experiment in a new directory.
python -m calibration_lab.forecast_audit --output output/my-forecast-audit

# Recreate per-example predictions and the complete paired dataset.
python -m calibration_lab.forecast_audit_evaluate \
  --run results/forecast-audit-v1 --output output/audit-recheck
```

Python 3.14 and the pinned dependencies were used. Install
`requirements-calibration-figures.txt` to redraw the figure with
`python -m calibration_lab.forecast_audit_analysis --figures docs/assets/forecast-audit`.

Training all twelve final models took **415 seconds** on the personal Mac's
central processor, with one computation thread per run. This excludes development,
evaluation and verification. The models consumed 18,432,000 interaction episodes
and 9,216,000 exercise-state presentations in total. Because streams are matched
across recipes, those came from 6,144,000 unique interaction worlds and 1,536,000
unique exercise worlds. Related initial/acquired states share an outcome.

All **76 offline tests passed**. The
[verification receipt](../results/forecast-audit-v1/verification.json) records
successful reconstruction of all **96 workflow evaluations and 24 paired-model
evaluations**, with a maximum metric difference of zero on the training machine.
It also checks training-stream hashes, matched exercise observations, 18 saved
interaction batches, and byte-for-byte reproduction of both benchmark files.

## Relationship to Jev

TypeSafe describes a model that returns probabilities for typed questions and
encourages asking questions the calling program may not ultimately use.
That motivates testing the quality of probability reports outside a policy's
usual chosen path. See its [primitives documentation](https://docs.typesafe.ai/primitives).

This experiment supplies an open training candidate and a controlled test of
the desired behavior. It does not identify Jev's architecture or training
algorithm. The optional runner follows the documented
[evaluation endpoint](https://docs.typesafe.ai/api); offline mock tests verify
request handling, not Jev's accuracy. Actual Jev results require configured access.

## Limits

- Proper outcome rewards do not guarantee convergence or reliable behavior on
  unfamiliar conditions. All trained models and failures are included.
- Early and continued practice match the extra examples and maximum update
  budgets. Their timing changes the model, its later actions and its collected
  interaction states. This is the intervention being tested, not a comparison
  with identical complete trajectories.
- Common-state probability error is measured against the exact conditional
  probability. It includes hypothetical reports on states the policy may avoid.
  Workflow return is measured separately with actual report sampling and costs.
- Price invariance includes a higher price outside training. Reversed sources
  are also outside training. They are deliberate tests of generalization.
- The text probes require interpreting a probabilistic mechanism and doing
  some reasoning. Their results would not be a comprehensive Jev evaluation.
- In this binary simulator, many sampled rewards reveal the underlying event
  if decoded. The reward learner is restricted to policy-gradient updates;
  we do not claim its feedback contains strictly less information than a label.
- The single development seed guided the decision to run this comparison and
  is excluded from final means. Three final seeds give a limited view of variation.
