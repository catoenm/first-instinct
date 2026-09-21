# The selected adapter fails the joint transfer check

The original supervised adapter and the preselected update-40 adapter completed
the frozen comparison. **Do not promote the candidate or scale this training
recipe from these results.** Continued-success forecasts improved slightly and
forecast-based control improved, but direct decision return fell. Both checkpoints
also failed the presentation-format control, so the comparison does not isolate
unfamiliar-task transfer from input-interface robustness.

| Reserved payment tasks, macro across four programs | Original | Update 40 |
| --- | ---: | ---: |
| Direct decision return | 0.1852 | 0.0995 |
| Direct choice accuracy | 68.75% | 64.58% |
| Success Brier error, lower is better | 0.6232 | 0.5871 |
| Success log loss, lower is better | 1.0124 | 1.0385 |
| Application-change Brier error | 0.4330 | 0.3933 |
| Forecast minus declared-cost controller return | 0.3358 | 0.5242 |
| Direct choice with cost-dominated plans excluded | 0.5946 | 0.5099 |

The primary direct-return gain of at least 0.03 failed: return changed by
−0.0857. The Brier-improvement requirement of at least 0.02 passed in isolation:
error fell by 0.0360. Absolute success forecasting remains poor: a constant 50/50
forecast has binary Brier error 0.5 under this scoring convention. Forecast-based
control improved by 0.1883 but remained below the declared longest-completion rule
(0.6229) and the best offered-plan oracle (0.7040). It is a partial result, not a
substitute for the failed primary check.

## The presentation control changed the interpretation

We asked the same familiar phone development questions in their original input
format and in the new lossless shared-catalog presentation. Within each checkpoint:

| Phone control | Original format | Shared presentation |
| --- | ---: | ---: |
| Original adapter success Brier | 0.4725 | 0.5832 |
| Update-40 success Brier | 0.2463 | 0.5005 |
| Original adapter direct return | −0.1432 | −0.0610 |
| Update-40 direct return | −0.0003 | −0.0777 |

Both exceeded the predeclared 0.02 Brier-degradation limit. Update 40 also exceeded
the 0.03 return-degradation limit. Exact computer decoding preserved the original
records and plans; this did not establish that the model could use the new
presentation equally well.

The shared presentation also exposes a catalog built from the public candidate
plans for that history, including entries unused by an individual forecast.
Therefore these controls measure the effect of the **whole input packaging
change**—references, instructions and additional public plan context. They do not
identify reference syntax alone as the cause. No outcomes or correctness labels
were included in the catalog. A narrower format diagnosis needs its own protocol;
do not tune packaging against these exposed transfer results.

General retention passed and exactly reproduced the earlier saved metrics:
accuracy 86.9958% → 85.8811%, log loss 0.352848 → 0.356055. The original-format
phone metrics also exactly reproduced the completed training run. Both loaded
adapter identities matched their saved tensors and stayed unchanged throughout
inference. This rules out a different checkpoint as the explanation for the
observed format-control regression.

## Accounting and next action

There were 2,020 completed inference presentations: for each checkpoint, 174
transfer questions, 107 phone questions in each of two presentations, and 622
general-retention questions. The transfer set still represents only four task
instances, four programs and eight histories, using 82 earlier branch-world
executions and eight clock-evidence worlds. There were no new optimizer steps,
model-derived labels or checkpoint-selection changes.

Evaluation took 456.6 seconds after setup. The owned H200 was deleted after all
229 recovered files verified. Estimated compute cost was $1.19 excluding storage;
this is an estimate, not a settled invoice. Retain the original conservative
budget holds until provider billing settles. No rental or training remains active.

Keep the demo unchanged. Prefer shorter, explicit observations and broader
execution-verified situations before more learning. The separately qualified
[telecom substrate](telecom-local-v1-results.md) adds actual device/account
dependencies; its next useful contribution is compatible hidden causes, costly
observations and measured consequences. Its three successful repair fixtures
alone are not a diverse training corpus. A controlled input-format study should
use development data and avoid choosing another winner on this exposed payment
set. We still do not have evidence of a mini-Jev with reliable general-purpose
decisions and calibrated forecasts.

[Frozen protocol](appworld-transfer-evaluation-v2-protocol.md) ·
[Full aggregate results and per-program metrics](../results/appworld-transfer-evaluation-v2/summary.json) ·
[Verified recovery and rental closure](../results/appworld-transfer-evaluation-v2/collection.json)
