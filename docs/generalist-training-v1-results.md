# The longer run improved forecasts and crossed the retention limit

The broader supervised continuation stopped after **384 updates**, having used
49,152 distinct questions and 19,861,872 input tokens. Outcome forecasts improved,
but general accuracy fell 1.029 percentage points, crossing the predefined
one-point retention limit. Tool accuracy also remained below its starting value.
The original supervised checkpoint stays selected for the demo. No replacement
qualified, and the training GPU has been recovered and deleted.

| Development measure | Original | Update 128 | Update 256 | Update 384 |
| --- | ---: | ---: | ---: | ---: |
| General accuracy, equal weight per task | 87.36% | 86.78% | 86.64% | 86.33% |
| Tool accuracy, equal weight per server | 86.85% | 86.01% | 85.75% | 86.22% |
| Tool log loss | 0.43250 | 0.39406 | 0.38422 | 0.37685 |
| Outcome expected Brier error | 0.34100 | 0.29971 | 0.27221 | 0.24884 |
| Outcome log loss | 0.59324 | 0.51977 | 0.47957 | 0.44712 |
| Application decisions | 16/24 | 18/24 | 18/24 | 18/24 |

Lower Brier error and log loss are better. Expected Brier error fell **27.0%**;
each of the three declared outcome groups improved on both probability scores.
This measures forecast quality, not calibration alone. The 24 application
questions provide a small diagnostic, not a broad transfer result.

The stop was caused by general accuracy, not an exhausted compute allocation or
a probability-score plateau. General log loss and all four product slices stayed
inside their limits. The accuracy drop exceeded its bound by just 0.029 percentage
points; the unchanged prospective rule still required stopping before the normal
512-update minimum. Raising that threshold after seeing these results would change
the experiment. All evaluated continuations had lower equal-server tool accuracy
than the parent, so the selected checkpoint remained update zero throughout.

Question-weighted tool accuracy increased from 85.43% to 86.20%. Seven server
groups improved, five declined, and fifteen stayed unchanged. The largest
equal-server decline came from a seven-question group that lost two correct
answers. This post-run diagnostic explains sensitivity to group weighting; it
does not replace the prespecified metric or its three-point advancement target.

## What actually trained

| Pool | Presentations | Unique questions | Input tokens |
| --- | ---: | ---: | ---: |
| General replay | 43,008 | 43,008 | 13,400,426 |
| Teacher tool choices | 4,608 | 4,608 | 5,309,564 |
| Execution-derived questions | 1,536 | 1,536 | 1,151,882 |
| **Total** | **49,152** | **49,152** | **19,861,872** |

The prepared schedule contained 400,896 presentations of 371,205 distinct
questions. Only 12.3% of those presentations were consumed. There were no repeated
questions within this run and no partial-update backward presentations. General
replay had already appeared in the original training, so these are not all new
examples. The verified subset belongs to ten split-ownership groups; those groups
are not a count of independently executed tasks or worlds. This run reused
existing evidence and executed **zero new counterfactual branches**.

Training started from the original step-2,742 adapter. All 496 internal adapter
tensors changed in the final saved checkpoint, covering 43,278,336 trainable
scalars. The foundation matrices stayed frozen. This was supervised learning from
decision and outcome targets, not online reinforcement learning.

An independent offline audit rehashed all 86 recovered files and 25 frozen inputs,
recomputed every development measurement from saved predictions, matched the
complete backward sequence to the frozen schedule and accepted-update logs, and
checked saved adapter lineage and the separate-process restart. The actual final
update-384 weights were evaluated and preserved. The audit loaded no foundation
model and scored no reserved transfer examples. Estimated GPU compute was $30.10,
excluding storage; the existing budget holds remain pending reconciliation.

## What this changes

More general replay and a lower learning rate allowed a larger dose, but did not
produce the required decision improvement. Before another run, inspect the tool
regressions and their acceptable-answer labels, and test an explicit constraint
on changes to the parent's replay probabilities. That is a proposed intervention,
not a demonstrated fix. More verified question rows also need more independent
mechanisms; the ten ownership groups reveal substantial concentration.

The [original protocol](generalist-training-v1.md) remains unchanged.
[Aggregate metrics, consumption, gates, and checkpoint hashes](../results/generalist-training-v1/summary.json)
record this completed run. Source questions and per-example predictions remain
private. Development feedback informed this continuation; it is not independent
confirmation of generality or equivalence to Jev.

## Tool-choice diagnostic after training

A subsequent offline join of all **2,217** tool questions to their original
schemas found 42 teacher disagreements corrected and 25 previously correct
answers lost; 1,869 stayed correct and 281 stayed wrong. Within the last group,
18 changed their choice while still disagreeing with the teacher. These counts reproduce both
the improved question-weighted accuracy and the lower equal-server accuracy.

The changes are concentrated around tool prerequisites. Of 85 changed choices,
**59 moved from a tool with required arguments to one with none**; none moved
in the opposite direction. Questions whose teacher tool requires no arguments
gained 34 correct answers and lost none. The remaining questions gained eight
and lost 25. Required fields describe a schema, not whether their values are
available in the request. This pattern does not establish that discovery tools
are better, or that all regressions are ambiguous.

Inspection of the two losses in the seven-question group found a plausible
ambiguity: the teacher requests content retrieval requiring an edition
identifier, while the continuation chooses a tool listing available editions.
The visible requests describe desired formats without supplying that literal
identifier. Neither route has been executed; tool descriptions or prior
knowledge might still make direct retrieval appropriate. These two cases were
selected after seeing failures and cannot estimate label-error prevalence.

This supports a specific prospective data test: keep commands fixed while
varying whether their required values are missing, current, or stale, and vary
the cost of discovery. Execute alternatives to distinguish useful inspection
from redundant inspection or sensible stopping. The
[identifier-evidence probe](identity-evidence-v1-protocol.md) uses authored
application fixtures, with none of these development requests copied into
training. The original labels, metrics, gates and selected checkpoint are
unchanged. [Diagnostic aggregates and input hashes](../results/generalist-training-v1/tool-diagnostic.json)
record zero model calls, executed alternatives or training consumption for this
diagnostic itself.
