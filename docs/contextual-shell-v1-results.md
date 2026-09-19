# Context-dependent shell decisions: results

The corrected data improved held-out shell decision accuracy from **58.27% to
84.57% and 86.26%** across two training seeds. The strongest gains were in
configuration and report tasks. Database decisions remain unreliable. General
benchmark accuracy changed little, while its probability scores became worse.
This supports further work on contextual data, without establishing a general
decision-making improvement or completing live shell reinforcement learning.

All four training arms and five evaluations completed on September 19, 2026.
The 158 archived artifacts were recovered and verified before the rented H200
was deleted. Estimated compute was **$17.77**, excluding storage.

## What was tested

The starting model was our original supervised Qwen3.5-9B checkpoint. Each arm
trained about 43.3 million parameters in internal rank-16 adapters; the foundation
matrices stayed frozen. Both seeds started from exactly the same checkpoint.

The control had 37,680 general examples. The context arm replaced 7,680 of those
with executed shell examples, retaining 30,000 shared general examples. Each
new example asked either which command would meet a stated goal or whether a
specified command would succeed. Candidate commands had already run from fresh
files, and an independent verifier checked their resulting state.

For each underlying case, we crossed two file states with two opposite goals.
The candidate commands stayed identical across all four contexts, but the
correct answers changed. Every candidate succeeded twice and failed twice.
A command-only lookup therefore scored 50%, compared with 93.66% on our
discarded first dataset. The first pilot was cancelled before any optimizer
updates. The [data account](shell-data-quality.md) explains that redesign.

The new test contains 960 questions: 192 command choices and 768 binary
forecasts. They come from 48 underlying four-context cases across six held-out
feature combinations in three authored task families. The feature combinations
were held out; the task families and command implementations were represented
in training. This measures a limited form of composition, not unseen tools or
arbitrary software tasks. See the [frozen protocol](contextual-shell-v1-protocol.md).

## Held-out results

Macro accuracy gives each of the six task types equal weight. Lower log loss
is better: it rewards assigning probability to an acceptable answer and
penalizes confident mistakes.

| Selected model | Shell macro accuracy | Shell log loss | All four contexts correct | Binary forecast Brier score |
| --- | ---: | ---: | ---: | ---: |
| Original | 58.27% | 0.8382 | 7/240 (2.92%) | 0.2557 |
| General control, seed 907 | 58.27% | 0.8382 | 7/240 (2.92%) | 0.2557 |
| Context data, seed 907 | 84.57% | 0.2671 | 149/240 (62.08%) | 0.0920 |
| General control, seed 1709 | 58.27% | 0.8382 | 7/240 (2.92%) | 0.2557 |
| Context data, seed 1709 | 86.26% | 0.2460 | 153/240 (63.75%) | 0.0790 |

The controls selected the original checkpoint under the predefined validation
rule. Their identical test results are expected; both controls did train and
change their adapter parameters, but their later checkpoints were not selected.

The four-context measure requires answering correctly for both goals and both
file states for one choice question or one fixed candidate command. The 240
decision groups share 48 underlying cases and are not independent trials.
This stricter check helps distinguish using context from always favoring the
same command. It was a post-freeze analysis and did not select checkpoints.

| Task accuracy | Original | Context seed 907 | Context seed 1709 |
| --- | ---: | ---: | ---: |
| Configuration: choose command | 76.56% | 95.31% | 96.88% |
| Configuration: forecast success | 63.67% | 97.66% | 97.27% |
| Report: choose command | 56.25% | 98.44% | 96.88% |
| Report: forecast success | 51.95% | 97.27% | 99.61% |
| Database: choose command | 50.00% | 60.94% | 64.06% |
| Database: forecast success | 51.17% | 57.81% | 62.89% |

Database performance is the main failure. Each context seed solved only 2 of
16 complete database choice groups. For database forecasts, the counts were
2/64 and 1/64. Better average accuracy does not make these decisions dependable.

The Brier score is the average squared difference between the probability of
success and the observed zero-or-one execution result. It improved on these
deterministic, authored tasks. That does not establish calibration across
unseen environments, uncertain outcomes, or long action sequences.

## General transfer and checkpoint selection

On the existing 1,128-question, 92-task general benchmark:

| Selected model | Macro accuracy | Macro log loss |
| --- | ---: | ---: |
| Original and both selected controls | 77.99% | 0.4997 |
| Context seed 907 | 78.22% | 0.5484 |
| Context seed 1709 | 78.67% | 0.5386 |

The paired accuracy changes were +0.23 and +0.68 percentage points. Descriptive
95% group-bootstrap intervals were [-0.61, +1.09] and [-0.31, +1.69] points,
respectively. Both include zero. Log loss worsened by 0.0487 and 0.0389; its
corresponding intervals were [0.0298, 0.0704] and [0.0228, 0.0574]. These intervals
describe this fixed, previously opened benchmark, not variation over training
seeds or all possible tasks. The baseline above was evaluated afresh within
this study; comparisons use that evaluation throughout.

Both context checkpoints passed the general-retention guards on validation.
The broader test nevertheless showed worse probability scores. Validation
retention did not guarantee transfer retention. The existing demo remains on the
original supervised checkpoint; this study does not justify replacing it with
either context checkpoint for general use.

All arms stopped after three scheduled checks without an eligible improvement.
The guard selected the lowest validation macro log loss, subject to general
accuracy and log-loss limits. Test results did not change training or selection.

| Arm | Updates run | Selected update | Row visits | Tokens processed | Training minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| General, seed 907 | 301 | 0 | 19,264 | 6,203,175 | 28.4 |
| Context, seed 907 | 801 | 500 | 51,248 | 19,567,680 | 74.1 |
| General, seed 1709 | 301 | 0 | 19,264 | 6,000,879 | 25.5 |
| Context, seed 1709 | 801 | 500 | 51,232 | 19,771,028 | 75.0 |

Available row counts, training limits and selection rules were matched. Actual
training work was not: the context arms continued improving for longer. The
study compares these bounded training procedures, not equal amounts of compute.
There were no failed or missing arms.

## What to try next

The useful data lesson is that execution labels alone were insufficient. The
first dataset let a lookup ignore the task. Crossing goals and states removed
that particular shortcut, and learning transferred to held-out combinations in
two of the three families. It remains possible to learn other narrow patterns;
six held-out combinations cannot establish broad reasoning.

The next experiment should expose useful observations before demanding a
decision. For a database task, offer an inspection query, return the actual
rows or calculated discrepancies, and let the selector choose the next command.
This tests evidence gathering and tool use as well as predicting the effects
of a command from a serialized database dump. Create simpler training cases
that isolate joins, empty aggregates, fees, ranking and protected-state rules;
reserve new combinations and task families for a future test.

For live reinforcement learning, collect complete tool trajectories: the visible
state, offered actions, selected action and its probability, actual command
output, state change, costs, and terminal verification. Ask for outcome forecasts
before execution and score them against subsequently observed outcomes. If a
forecast is about eventual success, specify the continuation policy and horizon;
a command being valid is a different target. Use a frozen proposer for the
first comparison so changes in its suggestions do not obscure selector learning.

A follow-up can compare reward-only learning with reward plus forecast training
on the same executable environment distribution, while checking task completion,
inspection costs, probability quality and general retention. This is proposed
work. This completed run was supervised preparation and did not train through
live Harbor trajectories or run Proximal Policy Optimization.

## Evidence and cost

The [machine-readable report](../results/contextual-shell-v1/training-report.json)
contains every task result, both paired comparisons and the selected checkpoint
identities. The [final audit](../results/contextual-shell-v1/final-audit.json)
replays validation selection from recovered predictions and verifies exact
adapter bindings. The [compute receipt](../results/contextual-shell-v1/compute.json)
records the estimate and its limits. Large checkpoints and raw predictions stay
in the local recovered artifact archive rather than the Git repository.

The corrected run used one personal H200 at $4.59/hour for an estimated 3.87
hours through provider shutdown confirmation, including setup, evaluation and recovery. Estimated compute for both
shell pilots together is $18.34; cumulative tracked compute is $151.42 against
the user's $500 authorization. Storage and any provider adjustments are not
included in those estimates. The $10 storage reserve keeps these two pilots
within their $65 allocation. Provider reconciliation confirmed zero remaining
pods. No Phantom resources were used.
