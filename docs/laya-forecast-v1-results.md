# External forecast baseline: unmodified Qwen wins this workflow

The fixed comparison completed all **924 primary predictions**: 308 each from
Laya typed-decisions, unmodified Qwen3.5-9B, and Qwen with our original supervised
step-2742 adapter. Independent metric reconstruction passed, all artifacts were
recovered, and the GPU was deleted. This run performed no training.

| Model | Expected outcome accuracy | Expected Brier error | Log loss | Warm native latency, three input medians |
| --- | ---: | ---: | ---: | ---: |
| Laya typed-decisions | 40.91% | 0.6670 | 1.1016 | 13.5–13.9 ms |
| Unmodified Qwen3.5-9B | **57.14%** | **0.5396** | **0.9546** | 59.4–83.0 ms |
| Our supervised Qwen3.5-9B | 52.27% | 0.6403 | 1.0847 | 75.6–110.5 ms |

Lower Brier error and log loss are better. Expected outcome accuracy is the
probability that the selected outcome occurs under the executed target
distribution; it is not an agent's task-completion rate. Unmodified Qwen here is
the published Qwen checkpoint without our adapter, not a model trained from scratch.

**Our supervised adaptation makes these consequence forecasts worse than the
unmodified foundation.** This is a result about this workflow, not evidence that
the adapter is worse at every task. General retention and broad tool selection
were not measured in this comparison. The result does not qualify a replacement
checkpoint or unlock the reserved release benchmark.

## Uncertainty reveals a different weakness

The cohort contains 268 deterministic and 40 ambiguous forecast questions.

| Model | Deterministic expected Brier error | Ambiguous expected Brier error | Ambiguous expected outcome accuracy |
| --- | ---: | ---: | ---: |
| Laya typed-decisions | 0.6724 | **0.6310** | **50.0%** |
| Unmodified Qwen | **0.4495** | 1.1435 | 5.0% |
| Our supervised Qwen | 0.5411 | 1.3051 | 0.0% |

All forty ambiguous questions concern writes with their visible prerequisites
satisfied. Under the declared compatible-world distribution, successful completion
and an incorrect mutation each have probability one half; unfinished has probability
zero. Laya always chooses an outcome with nonzero support. Unmodified Qwen chooses
the zero-support outcome in 36 cases, and our adapter does so in all forty. Mean
largest probability on this subset is 42.2%, 63.4%, and 71.9%, respectively.

This shows why a single overall accuracy or confidence number is insufficient.
Laya is weaker overall but much better on this ambiguous subset. It does not prove
that Laya is generally calibrated: these are forty related questions from one
authored workflow and finite declared world distribution.

The tool descriptions are terse. Inspection of missed cases suggests another
data-quality question: does the visible contract adequately explain the executor's
failure conditions, automatic computations and terminal mutations? For example,
an inspection outage does not necessarily prevent a write in this executor.
That dependency is clearer in the implementation than in the brief tool description.
This is a **post-hoc hypothesis**, not a tested explanation or a finding that the
executed labels are wrong. The original benchmark and labels remain unchanged.

## Fairness, timing and verification

All models received the same complete state, question and option descriptions
through their existing interfaces. Every Laya input retained its full information;
every Qwen input exactly matched frozen token IDs. No truncation, temperature
fitting, checkpoint search, output relabeling or question-dependent routing occurred.
Laya used its shipped three-choice temperature of 1.7601518630981445; Qwen used one.
Native unrounded probabilities and stable log scores were retained; none of the
primary predictions required handling a native underflow.

The same L40S ran each model serially. Native latency includes tokenization,
transfers, inference and output computation, with synchronized timing. Each of
three prospectively selected questions received three warmups and five measured
calls. This is a small local timing sample, not hosted throughput or a concurrency
benchmark. Instrumented probability capture, model loading and timing calls are
accounted separately. There were 18 synthetic qualification forwards and 72 timing
forwards in addition to the 924 primary predictions. All repeated-forward
qualification checks had zero probability drift.

The [protocol](laya-forecast-v1-protocol.md) fixes checkpoint revisions and methods.
The [hardware admission correction](laya-forecast-v1-hardware-admission.md) preserves
the original frozen source and records the executed one-byte memory-check overlay.
Both remote and local independent audits reconstruct the proper scores. Recovery
verified all 42 artifact files, source identities, the overlay and checkpoint
lineage. The three-model worker finished in about 110 seconds after setup.

This cohort is the existing **exposed report development set**, with one mechanism;
its related questions are not 308 independent real-world tasks. No new worlds,
executed branches, labels or training presentations were generated. The older
training run used different hardware and batching, so compare the three models
within this run rather than interpreting tiny cross-run probability differences.

Estimated rental compute was **$0.19**, excluding storage. The conservative ledger
retains the full $6 stage hold and all older holds, leaving $146.64 unallocated
within the original cumulative $500. No GPU remains running.

## What to do next

Do not scale the existing supervised recipe on the strength of this result. First
audit the completeness of public tool contracts, then qualify a prospectively
defined contract-format comparison on exposed examples without changing labels.
That can test whether ambiguous forecasting fails because relevant mechanics are
poorly specified, before trying to teach them through more examples.

Continue the [broader mechanism plan](mechanism-expansion-v1.md): execution-verified
state changes, competing writers, object aliases and durable work, with meaningful
mechanism ownership. Keep both the unmodified foundation and trained adapter as
controls in the next learning pilot, alongside general-task replay and retention.
Scaling becomes useful after we can show that the revised data and objective
improve decisions and forecasts on unfamiliar situations.
