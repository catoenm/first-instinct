# Explicit tool descriptions improve Qwen's consequence forecasts

The paired comparison supports the public-contract hypothesis for both Qwen
checkpoints on this exposed report workflow. Adding the same accurate description
of tool effects, refusals and termination substantially improves the original
supervised model's forecasts without changing its weights. Laya does not pass the
predeclared improvement checks.

| Fixed checkpoint | Expected outcome accuracy, original → contract | Expected Brier error, original → contract | Prospective checks |
| --- | ---: | ---: | --- |
| Original supervised Qwen | 52.27% → 69.48% | 0.6403 → 0.4182 | Pass |
| Unmodified Qwen | 57.14% → 70.13% | 0.5396 → 0.4539 | Pass |
| Laya typed | 40.91% → 40.91% | 0.6670 → 0.7026 | Fail |

Lower Brier error is better. Expected outcome accuracy averages the conditional
probability of the model's chosen outcome; an ambiguous 50/50 question awards at
most half a point. These numbers describe 308 related questions from one mechanism,
two physical worlds and four world/goal tasks, not 308 independent tasks.

The supervised model's overall log loss improves from 1.0847 to 0.7187. On the
40 ambiguous questions, its Brier error falls from 1.3051 to 0.7315. Its top choice
has zero support under the declared compatible worlds in 40 original prompts and
zero augmented prompts. The foundation changes from 36 such choices to zero.
The ideal expected Brier error on these 50/50 questions is 0.5, so substantial
probability error remains. This does not establish general calibration.

The improvement also appears on the 268 deterministic questions: supervised
accuracy rises from 60.07% to 72.39%, and Brier error falls from 0.5411 to 0.3715.
With the contract present, the supervised checkpoint has lower Brier error than
the foundation (0.4182 versus 0.4539), although slightly lower expected choice
accuracy (69.48% versus 70.13%). This qualifies the earlier observation that the
foundation was better under the original terse interface. It does not demonstrate
a universal benefit or harm from our supervised adaptation.

Laya's overall Brier error worsens from 0.6670 to 0.7026 and its expected accuracy
is unchanged. Its ambiguous-subset Brier error improves slightly, but not enough
to pass the prospective checks. All models and both formats retain complete input
information, and native calibration remains unchanged. The result is specific
to this model, runtime, cohort and input extension.

## What was controlled

The [protocol](report-contract-paired-v1-protocol.md) was frozen before the rental.
All 1,848 primary predictions completed on one L40S: three fixed checkpoints,
308 original questions and their paired augmented versions. Labels, option order,
checkpoint bytes, native temperatures and numerical methods stayed fixed.
Pairs alternated presentation order. The same public contract was appended to
every state; it contained no hidden-world answer or target probability.

The 18 synthetic qualification forwards and 144 timing forwards are accounted
separately. Each timing condition uses only three inputs and five measured calls;
these are descriptive local timings, not hosted throughput. There was no
temperature fit, training, checkpoint search, new branch execution, or opening
of the reserved release evaluation. The initial prepared archive remains intact;
the pre-scoring audit-tolerance correction changed no input question bytes or
model computation.

Both original-input Qwen results and Laya results reproduce the previous
baseline probabilities exactly. Independent local scalar reconstruction agrees
with the remote numerical audit, and recovery verifies the frozen source, input,
checkpoint lineage and prediction order. All 44 artifact files were recovered
and hash-verified before the owned GPU was deleted. Worker execution took about
245 seconds; rental compute including setup and recovery was approximately
$0.21, excluding storage. The full $6 hold remains
within the original cumulative authorization.

## Consequence for the data work

Tool behavior needs to be stated consistently in both training and evaluation.
The same checkpoint can look much worse when asked to forecast an underspecified
executor. This result supports improving the interface before scaling the same
data, but does not prove that missing descriptions caused earlier reinforcement
learning failures: this run measured forecasts, not learned action returns.

Keep complete public tool contracts in the next paired decision/forecast data
interface, preserve valid uncertainty, and test on broader mechanisms. The new
[competing-writer qualification](revisioned-sqlite-v1-results.md) supplies one
small executable starting point. Its command return codes and verified goal
outcomes deliberately differ. It is not yet a model training corpus.

The selected project checkpoint remains original supervised step 2742. No new
weights or release promotion resulted from this diagnostic. The
[aggregate evidence](../results/report-contract-paired-v1/summary.json) includes
all model/subset scores and the fixed improvement checks.
