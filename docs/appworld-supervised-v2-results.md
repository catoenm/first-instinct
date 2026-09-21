# Consequence learning passes development, with transfer still unproven

The bounded Qwen3.5-9B supervised pilot completed **60 updates**, stopped after two
non-improving checks, and selected **update 40** using the rule frozen before
training. The independent audit passes. All 264 archived files were recovered
and hash-verified before the H200 was deleted. No checkpoint was promoted to the
demo and no additional rental followed this result.

The selected checkpoint meets the prospective joint development gate: at least
0.03 higher direct-choice return, at least 0.02 lower continued-success expected
Brier score, and the existing general/older-forecast retention checks. Absolute
direct-choice performance remains weak, approximately the return of always
stopping. A separate controller using the model's consequence forecasts and
explicit cost arithmetic performs much better on these development cases.

| Measurement | Original adapter | Selected update 40 |
| --- | ---: | ---: |
| Direct-choice acceptable-answer accuracy | 68.18% | 72.35% |
| Direct-choice mean return | -0.14317 | -0.00030 |
| Continued-success expected Brier score; lower is better | 0.47250 | 0.24632 |
| Continued-success log loss; lower is better | 0.88844 | 0.38154 |
| Older forecast development Brier score | 0.64097 | 0.35736 |
| General replay-retention macro accuracy | 87.00% | 85.88% |
| General replay-retention log loss | 0.35285 | 0.35605 |

The two phone programs are held out from training but used for checkpoint
selection. Their 57 decision questions and 50 continued-success forecasts cover
five underlying worlds. They are exposed development data, not an independent
transfer test. The 1.11 percentage-point retention decline is within the preset
two-point allowance; it is not evidence that all broader abilities are unchanged.
State-change forecasting is reported separately and was already near ceiling.

## The useful controller result

For each offered plan, predict its probability of completing the whole task under
that exact continuation, then subtract the stated fee times its number of calls.
Choose the largest expected return. This controller uses the separately predicted
outcome of stopping too. It receives no ground-truth labels or hidden state.

On exactly the same 57 development menus, this controller improves from **0.16397
to 0.64226** after training. The best offered return using executed outcomes is
0.64252. The strongest fixed command-shape heuristic scores 0.56097. A simpler
controller that only excludes plans costing more than the maximum possible reward
scores 0.58909 with the selected model. The model's raw direct choice remains much
worse because it sometimes chooses obviously uneconomic plans.

This diagnostic was defined after baseline/update-10 aggregate development scores
were visible. It did not alter training or select update 40. The candidate comes
from the original combined development rule, independently of the diagnostic.
All option forecasts were matched to exactly the same public histories and
underlying world populations before comparing controllers. The
[diagnostic explanation](appworld-controller-diagnostic-v1.md) and
[complete checkpoint aggregates](../results/appworld-supervised-v2/controller-diagnostic.json)
include the controls and remaining probability errors.

This is promising evidence for forecasting consequences and handling known costs
in code. It is not a calibration guarantee: some failed plans still receive high
success probabilities. It is also a mixed supervised recipe, so the experiment
does not isolate forecast-only training, online reinforcement learning, or a
private Jev training method. Reference-assisted supplied plans do not establish
independent tool proposal or adaptive replanning.

## What was actually trained

The foundation is unchanged Qwen3.5-9B at its pinned revision, initialized from
the original supervised step-2742 adapter. Rank-16 adapters inside the language
network are trainable; foundation matrices remain frozen. All 496 adapter tensors
changed. The selected file contains 43,278,272 changed scalar values out of
43,278,336 trainable values. This is an actual change to the language network's
effective transformations, not a newly attached classifier.

| Consumption | Selected update-40 lineage | Entire stopped run |
| --- | ---: | ---: |
| Optimizer updates | 40 | 60 |
| Completed backward presentations | 960 | 1,440 |
| Distinct pool/question pairs presented | 835 | 1,120 |
| Input tokens | 1,999,966 | 3,000,049 |
| Presentations of genuinely uncertain older forecasts | 24 | 40 |

The whole run presented 480 new forecasts, 240 new decisions, 240 older forecasts
and 480 general replay questions. The selected checkpoint precedes the last 20
updates. One longest-context backward diagnostic, with no optimizer update, is
accounted separately. Training-process wall time, including loading and evaluations,
was about 29 minutes. Available pools contained 6,860 questions; availability is
not consumption. The application slice came from the previously completed
507-world-execution collection, not new worlds generated during training.

The selected adapter file hash is
`0e4c4888045df59c5d9f6d2adbf0dbbdc56b24ec597c8a03a203e4ad07229c5f`.
Its canonical sorted-name float32 tensor hash is
`a52a8f2c0e4d87d8269c69deb78ed667157f13156d9266601f4ad3dd119d9291`.
The saved final update-60 canonical hash independently matches the final runtime
receipt. Public [consumption and lineage](../results/appworld-supervised-v2/summary.json),
[recomputed metrics and selection](../results/appworld-supervised-v2/final-audit.json)
and [tensor changes](../results/appworld-supervised-v2/saved-tensors.json) are available;
protected question and prediction files remain private.

Observed compute is approximately $3.35 for the successful rental, or $4.39 across
all three attempts, excluding storage. The first two failures performed zero
optimizer updates. Provider billing still lags, so those are conservative runtime
estimates, not settled invoices. Every owned pod is deleted; the original budget
and recovery reserves remain in force.

## The next data bottleneck

The reserved payment-workflow collection passed execution checks on four programs,
but [question admission failed](appworld-transfer-admission-v1-results.md): public
argument provenance and the context ceiling removed useful decision coverage from
three programs. No reserved questions were sent to a model or added to training.
Repairing that evidence interface, with explicit proofs and prospective controls,
is the next justified step. Increasing model size or starting another paid run
would not repair missing or unqualified transfer evidence.
