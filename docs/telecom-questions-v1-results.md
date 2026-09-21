# Short paired questions, with a stricter evaluation-only split

The verified telecom executions now support **51 distinct questions**: 15 immediate
forecasts, 18 forecasts under specified continuations, nine procedure choices and
nine value-of-observation questions. There are 54 raw variants; three immediate
forecasts are exact duplicates and were merged without changing their witnesses.
Twenty-two forecast questions retain fractional targets rather than majority labels.

Each displayed continuation is a short Python procedure with only public tool calls
and observed responses. Replaying those displayed procedures against all 144 saved
execution receipts requested exactly the original commands and arguments. Immediate
questions display only the first command and then stop. Decision and observation
questions have 90 checked joins to the matching continued forecasts. No environment
was rerun and no model supplied a label.

The longest input is **1,864 tokens**, under the prospective 4,096-token limit using
the pinned nine-billion-parameter model's tokenizer and current serving template.
Native lookup responses are preserved; there is no evidence cropping or shared
catalog encoding. This proves faithful, bounded presentation, not that a model
understands the procedure or will predict calibrated probabilities.

## The first ownership check missed an overlap

The original construction pass assigned 17 questions to each mechanism's candidate
role. A deeper review then found **two identical physical initial states and seven
identical execution trajectories shared between device and allowance mechanisms**.
The original check compared mechanism labels and rendered questions, which was
insufficient for the requirement to keep related worlds together.

We preserved that pass and rejected its proposed training partition. A separate
usage manifest now connects overlapping mechanisms and assigns each connected
group its most restrictive existing role. Device and allowance are both reserved
for transfer; roaming stays in development. Nothing previously reserved enters
training, and no world or difficult question is dropped.

| Final permitted use | Questions |
| --- | ---: |
| Training | **0** |
| Development | 17 |
| Reserved transfer | 34 |
| Actual model presentations / optimizer steps | 0 / 0 |

The original candidate-role fields remain immutable provenance, not authorization
to train. Future loaders must use the corrected manifest and reject use that
conflicts with it. This correction happened before any model call on these examples.

Seven question/loss tests, two ownership tests and eight saved-row corruption
controls passed. Targets were independently reconstructed from stored states and
actual call/billing ledgers. The existing proper categorical loss prefers the true
fractional distribution over unjustified certainty; an acceptable set containing
both answers is explicitly rejected as a training contract.

These are evaluation data from three authored mechanisms and ten physical states,
not a general training corpus. Qualifying additional independent workflows remains
necessary before another learning comparison. This stage consumed no paid compute
and changed no checkpoint or demo.

[Admission protocol](telecom-questions-v1-protocol.md) ·
[Ownership correction](telecom-questions-v1-ownership-correction.md) ·
[Aggregate evidence](../results/telecom-questions-v1/summary.json)
