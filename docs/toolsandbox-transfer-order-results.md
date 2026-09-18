# Answer order changes the model's forecasts

Reversing the supplied answer options changed the most likely future cost on
**69 of 96 questions**. The most likely terminal outcome stayed the same on
all **144 outcome questions**, even though some outcome probabilities moved
by as much as **27.6 percentage points**. The model, state, question and option
descriptions were unchanged; semantic answer IDs were matched across orders.

This was a **post-hoc diagnostic** prompted by the
[original supervised transfer failure](toolsandbox-transfer-supervised-results.md).
Its [protocol](toolsandbox-transfer-order-v1-protocol.md), code and all 240 inputs
were [published before these new predictions](https://github.com/catoenm/first-instinct/commit/ce4ecfb998aa94c255807c491190c462d03cf7d8).
Every nontrivial initial-state forecast menu was reversed once. The original
predictions were reused; no other permutations were tried or selected.

| Forecast | Questions | Most-likely answer changes | Mean total variation | Largest individual probability change |
| --- | ---: | ---: | ---: | ---: |
| Terminal outcome | 144 | 0 | 0.151517 | 27.65 percentage points |
| Future cost | 96 | 69 | 0.447877 | 38.23 percentage points |

Total variation is half the sum of absolute probability changes within each
question. It describes how much probability mass moved; it is not average
per-option error. Neither presentation produced an exact modal tie.

![Original and reversed forecasts, with implied decision values](../results/toolsandbox-transfer-order-v1/option-order.png)

Each scatter point matches the same semantic option across presentations. All
option components are included. The points are related within questions and
across the 48 authored roots; they are not independent observations.

## What happened to the implied decisions

Using the reversed forecasts changed the planner's initial action at just
**3 of 48 roots**: all three changed from stopping to the complete query.
It still stopped at 45 roots. Those three changes matched the exact menu
optimum, but the overall transfer weakness remained.

| Presentation | Optimal initial choices | Mean expected value | Mean expected regret |
| --- | ---: | ---: | ---: |
| Original | 17 / 48 | 0.208333 | 20.406250 |
| Reversed once | 20 / 48 | 1.838542 | 18.776042 |

Values use the existing finite-world labels and research-credit utility, with
equal weight per root and the same fixed continuation. They are implied choices
from forecasts, not newly executed adaptive-policy returns or the model's
direct action answers. The 48 singleton stop costs remain exact zero-cost
bypasses. Phone-conditioned questions were not changed or compared.

Probability error also depends on the metric. Outcome excess expected Brier
score fell from 0.692584 to 0.605979, and cost excess expected Brier fell from
0.477684 to 0.450670. Expected clipped log loss fell from 3.268207 to 2.811071
for outcomes but **rose from 3.636835 to 4.050052 for costs**. The
[complete report](../results/toolsandbox-transfer-order-v1/report.md) retains
both metrics and every probability pair. No presentation is promoted.

## Interpretation and limits

An unchanged winning answer can hide a changing probability distribution.
That matters when ordinary code uses those probabilities to compare the
consequences and costs of actions. This model's constrained output format does
not make its forecasts invariant to answer order.

One reversal did not resolve the original failure, and this experiment does
not establish that answer order caused all of it. Original and reversed calls
occurred at different times without contemporaneous original-order repeats;
temporal or numerical variation was not independently estimated. These are 48
correlated variants of one authored four-world mechanism, not broad calibration
evidence or a comparison with Jev. No reinforcement-trained checkpoint appears
in this diagnostic.

The original 720-question transfer result remains the primary reference.
These scenarios are diagnostic-only and do not select training data,
hyperparameters, checkpoints or a preferred prompt ordering.

## Reproduce

All 240 requests were attempted, received and validated once, with no retries.
Execution took 740.012 seconds on the existing supervised Qwen3.5-9B demo.
The unchanged source, adapter, tokenizer, package versions, service identity
and input hashes were checked before and after inference. This is recorded
provenance, not an attestation of resident model tensors.

[Raw journals and summary](../results/toolsandbox-transfer-order-v1/) and the
[read-only analyzer](../results/toolsandbox-transfer-order-v1/analyze.py) are
published. The public copies were independently rescored, and the resulting
figure was visually inspected. Rescoring requires no model or model-service
calls; Matplotlib is optional for the figure.

```sh
python results/toolsandbox-transfer-order-v1/analyze.py \
  --folder results/toolsandbox-transfer-order-v1 \
  --output output/reproduced-toolsandbox-order
```
