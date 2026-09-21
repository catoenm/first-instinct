# Better forecasts, but no release promotion

The coverage-balanced nine-billion-parameter continuation completed 160 updates.
All three outcome groups improved on both proper probability scores, and the
general-task retention checks passed. Tool-selection accuracy did not meet the
required three-percentage-point improvement, so the unchanged stopping rule ended
the run. The original supervised checkpoint remains selected for the demo.
The trained update-80 and update-160 adapters are preserved as experimental results.

| Development measure | Original | Update 80 | Update 160 |
|---|---:|---:|---:|
| Tool accuracy, equal weight per server | 86.87% | 86.47% | 86.68% |
| Tool log loss | 0.43247 | 0.35939 | 0.34057 |
| General accuracy, equal weight per task | 87.24% | 86.48% | 86.33% |
| General log loss | 0.34015 | 0.34885 | 0.35746 |
| Outcome expected Brier error, equal weight per group | 0.34093 | 0.26636 | 0.14586 |
| Outcome log loss | 0.59314 | 0.44280 | 0.24796 |
| Application decisions, 24 questions | 16/24 | 18/24 | 19/24 |

Lower Brier error and log loss are better. The final expected Brier score fell
**57.2%** relative to the original. This is a proper forecast-quality measure,
not a claim that calibration error alone fell by that amount. The three declared
groups improved separately:

| Outcome group | Questions | Original Brier → final | Original log loss → final |
|---|---:|---:|---:|
| Application state change | 29 | 0.001881 → 0.000535 | 0.019838 → 0.009797 |
| Application task completion | 29 | 0.380453 → 0.219096 | 0.672794 → 0.342017 |
| Report-mechanism forecasts | 308 | 0.640465 → 0.217961 | 1.086799 → 0.392074 |

Tool accuracy declined by 0.19 percentage points. General accuracy declined by
0.91 points, within the one-point bound but close to it; general log loss rose
by 0.01731, within its 0.02 bound. All four declared product slices passed. These
results support a useful forecast-learning signal, not a generally better model.
The 24 application decisions are a small diagnostic, not a broad transfer result.
The tool labels are observed teacher first actions, not independently verified
successful outcomes. Their 27 server groups have unequal support; the prespecified
equal-server metric remains binding even when larger groups improve.

A post-run diagnostic makes that distinction concrete: counting each tool
question equally, correct answers increased from 1,895 to 1,933 of 2,217
(85.48% to 87.19%). There were 58 newly correct answers and 20 regressions. Ten
server groups improved, four declined and 13 were unchanged. Menus with at least
13 options improved from 208/284 to 226/284. This is useful evidence about where
learning happened, but it neither replaces the prespecified equal-server gate
nor proves a causal effect of menu size.

The same development cohort was used throughout: 3,465 general questions across
157 task groups, 2,217 tool questions, 366 outcome forecasts and 24 application
decisions. Fresh reserved tool transfer was not scored. Earlier development
feedback informed this follow-up, so it is not independent confirmation. Sampling
coverage, verified-example weight and duration changed together; this comparison
does not isolate which change caused the forecast improvement.

## What actually trained

The run started again from the original supervised step-2742 adapter, not the
first pilot's failed continuation. It optimized the existing 43,278,336 internal
adapter scalars. These were mixed supervised decision and distribution targets,
not online reinforcement learning. The real-device context, gradient and
separate-process restart checks passed before sustained training.
An additional local weight audit found that all 496 internal adapter tensors
changed, retained their expected shapes and precision, and contained only finite
values. The final adapter file has digest
`f95c2437301b0c752023851754bd3978ba2fce58205687133917b8a7d80a0475`.

Actual consumption was **10,240 presentations of 10,032 unique tokenized
questions**, totaling **7,201,901 input tokens**: 5,120 general replay presentations,
3,840 tool choices and 1,280 verified questions. There were zero uncommitted
backward presentations. The frozen schedule prepared 15,360 presentations, but
the final 5,120 were never consumed. The 371,278-question release corpus was not
fully trained on. Its consumed rows inherited 8,945 ownership groups; that is not
a count of independently executed worlds. The run executed no new counterfactual
branches. The separately collected 190 fresh-observation retail forecasts were
not included in this frozen training schedule.

Both release attempts together consumed 15,360 presentations and 10,709,074
input tokens. Their unique-question counts cannot be added: the attempts reused
the admitted corpus and original starting checkpoint. No new main run was started
after the failed advancement gate.

An independent audit recomputed every saved evaluation from per-question
predictions and matched the complete consumed sequence to the frozen schedule.
All 64 recovered files passed their hashes before the rental was deleted. The
follow-up's estimated GPU compute cost was $5.17, or $8.53 for both release pilots,
excluding storage. Provider billing was still incomplete at collection; conservative
budget holds remain in place. No GPU rental remains active.

## Next use of this evidence

Preserve the trained forecast adapter for controlled comparisons. Do not replace
the general demo or increase the paid run merely because average probability
scores improved. The next training proposal must explain how it addresses tool
selection without spending the remaining general-retention margin. Separately,
qualify a smaller Mac inference package of the already released original model;
serving that existing model does not require promoting this continuation.

The [prospective protocol](release-balanced-v1-protocol.md) fixes the recipe and
unchanged acceptance thresholds. [Aggregate audit and final gates](../results/release-balanced-v1/)
contain no raw protected questions or per-example predictions.

Frozen run identity:
`2dac4a11244fdfd5f49829b46c672d3699499c68587354b49f41df3948d7138e`.
Original adapter:
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.
Recovered archive:
`41b76ab2e78ce6e1675277fb98d204f09a5ecbcd258e9a54521f806f9f0d1271`.
