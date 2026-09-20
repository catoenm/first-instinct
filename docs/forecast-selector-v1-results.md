# Forecasts are not yet reliable enough to drive the controller

The offline development diagnostic passed its integrity checks and produced a
negative capability result. Directly choosing actions from the saved consequence
forecasts did not rescue the combined models. **Forecast accuracy, not only the
connection between forecasts and actions, needs substantial improvement.**

We froze the [diagnostic protocol](forecast-selector-v1-protocol.md) before
calculating these results. It reuses the original and six already-selected
expanded checkpoints, with no temperature fitting, checkpoint reselection,
additional inference or GPU use. Calendar was not included. This is exploratory
analysis of exposed development data, not another independent transfer test.

## What was compared

For each offered command, the selector uses
`P(completed) - P(incorrect) - immediate public cost`. We score its choice against
the existing executed branch, including every actual future cost. Identical
observations are grouped before choosing; the selector cannot secretly know
which compatible hidden world is real. Exact ties use the action ID.

| Saved forecast model | One action, then stop: return | Action plus declared continuation: return | Continuation regret |
| --- | ---: | ---: | ---: |
| Original supervised | -0.0209 | 0.1582 | 0.6973 |
| Forecast / 1507 | -0.0209 | 0.2000 | 0.6555 |
| Forecast / 1609 | -0.0209 | 0.1573 | 0.6982 |
| Reward / 1507 | -0.0227 | 0.1591 | 0.6964 |
| Reward / 1609 | -0.0209 | 0.1573 | 0.6982 |
| Combined / 1507 | -0.0209 | 0.1573 | 0.6982 |
| Combined / 1609 | -0.0209 | 0.1555 | 0.7000 |
| Uniform probabilities, same selector | 0.0000 | 0.0000 | 0.8555 |
| Perfect verified probabilities, same selector | 0.3564 | 0.8409 | 0.0145 |
| Best expected full return in offered menu | 0.3564 | 0.8555 | 0.0000 |

Return is completion utility (+1), incorrect mutation penalty (-1), or unfinished
utility (0), minus costs. The table weights the 22 distinct visible contexts
equally. The [full results](../results/forecast-selector-v1/report.json) also give
case-weighted and per-regime measurements. The last two rows are verifier-based
oracle references, not learned policies. Regret is the difference from the best
expected full return after averaging compatible worlds, not an omniscient
per-world choice.

The immediate-cost rule omits future continuation costs. Even with perfect
outcome probabilities, that omission loses 0.0145 return in this small diagnostic.
The combined models lose about 0.70. Missing future costs are real, but they do
not explain most of this particular gap. This is not a general decomposition of
causes or proof that adding one loss will fix learning.

The uniform reference always stops at zero cost. Every learned selector has
negative mean return under the one-action contract. There is a positive-return
action in 8 of 22 immediate-contract contexts and 20 of 22 continuation contexts.
Thus the authored menus contain useful actions; menu coverage alone does not
explain the failures.

## A concrete failure

In visible context `5a6316ff98754aed5477c8da0c203e7247a59bc8cb13bdbe21623099876c1a21`,
a current observation identifies which region belongs in a report. Executing the
correct repair gives verified return 0.98; the wrong repair gives -1.02; stopping
leaves the report unfinished and returns zero.

The original model gives stopping a predicted continuation utility of 0.574,
above the correct repair's 0.475. Combined seed 1507 gives stopping 0.616,
above the correct repair's 0.566. The same mistake appears under the immediate
contract. The selector therefore stops despite having enough visible evidence.
These numbers are predicted utility scores, not completion probabilities.
The complete scores, selected actions and executed receipt identities are saved
in [decisions.jsonl](../results/forecast-selector-v1/decisions.jsonl).

This motivates paired examples that distinguish completed from merely error-free
or unfinished states, plus forecasts of concrete state changes. It does not
justify relabelling failures, hiding uncertain examples, or treating an operation
that returned successfully as a completed task.

## How much evidence this is

The diagnostic covers one authored report root, four world-and-goal tasks,
36 context/cost/history cases, 22 visible contexts and 504 previously executed
branches. There are 308 unique forecast inputs, of which 40 are uncertain under
the finite authored prior. Seven checkpoints supply 2,156 already-saved prediction
vectors, used for 308 model/contract/context selections. Reusing those vectors
is not new model inference or additional data diversity.

No new task executions, model calls, optimizer updates or training questions
occurred. Eight regression tests cover public-only selection, missing menus,
invalid probabilities, hidden-world information, changed artifacts and separate
context/case weighting. The runner verifies source hashes, receipt identities,
branch costs, labels, full menu coverage and identical forecast inputs.

The earlier live actor chooses under a different prompt and replans after every
observation. Its reward cannot be compared to this fixed-continuation selector
as though they were a matched experiment. No model is promoted to the demo.

The next priority is broader execution-verified supervised data: varied predicates,
failure and recovery states, immediate versus continued effects, and realistic
application workflows. Keep the existing 9B foundation for a bounded supervised
pilot before testing another reinforcement-learning recipe.
