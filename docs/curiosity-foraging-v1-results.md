# Learning from curiosity without game rewards

**The agents learned useful next-observation predictors, but curiosity did not
reliably improve survival or adaptation. Evolution did not beat random exploration
at learning the world model.** The fixed experiment completed and its independent
audit passed. The predefined advancement check failed, so no larger search ran.

See the [prospective protocol](curiosity-foraging-v1.md) and
[full aggregate results and receipts](../results/curiosity-foraging-v1/summary.json).
This is a separate 40,042-parameter local experiment, not a new 9B checkpoint.

## What zero game reward means here

During this run, no game score, food bonus, death penalty, survival bonus, or
winning label entered policy training, evolutionary ranking, or checkpoint
selection. Agents received local observations and their visible energy. An
adapter discarded the game's authored reward features and explicit episode-end
signals. Resets remained visible as changes in observations.

Three world predictors learned from actual observed transitions. Internal rewards
combined observation novelty, predictor disagreement, and predictive learning
progress. Evolution selected the reward mixtures that produced the lowest error
on separate, fixed prediction probes. The value critic predicted future internal
curiosity returns. There were still learning objectives and internal rewards.

All reward mixtures and checkpoints were selected before any game performance
was measured. Continued learning in the harder world used a fixed duration and
the same curiosity objectives. Game outcomes did not alter that schedule.

## Prediction and gameplay results

Each gameplay value averages two learner seeds and 64 maps per condition.
Prediction errors average the same two learner seeds on 8,192 separate probe
transitions per condition. There are 64 distinct familiar maps and 64 distinct
scarcity maps, reused across agents and before/after measurements. Lifespans
are capped at 128 steps.

| Method | Familiar prediction error, lower is better | Familiar lifespan | Scarcity lifespan before adaptation | Scarcity lifespan after adaptation |
| --- | ---: | ---: | ---: | ---: |
| Evolution, search 1 | 0.056423 | 32.66 | 18.57 | 18.46 |
| Random search, search 1 | 0.056659 | 32.38 | 18.62 | 19.13 |
| Evolution, search 2 | 0.056459 | 30.45 | 18.97 | 18.80 |
| Random search, search 2 | 0.056246 | 31.41 | 18.52 | 18.52 |
| Observation novelty | 0.058468 | 29.20 | 15.13 | 14.41 |
| Predictor disagreement | 0.056942 | 30.48 | 19.55 | 18.79 |
| Prediction learning progress | 0.059895 | 29.45 | 20.10 | 21.34 |
| Equal curiosity mixture | 0.058922 | 29.50 | 18.63 | 18.52 |
| Random actions, learned predictors | **0.055080** | 30.07 | 18.90 | 18.90 |

The predictors learned more than simply returning an empty or unchanged view.
On the familiar probes, untrained networks had error about 0.2477, predicting
zeros gave 0.1119, and copying the current observation gave 0.0736. All trained
conditions beat those reference predictions. However, collecting data through
uniform random actions trained the best predictors in this comparison.

Evolution needed at least 5% lower familiar prediction error than both random
search and the best fixed control in each repeat, without losing to random search
on either learner seed. Instead, evolved error was **2.44% and 2.50% higher** than
the best fixed control, and the learner-seed check also failed. This gate read
only prediction errors, never game scores.

The small familiar-survival differences are not consistent evidence of improved
play. For example, search 2's evolved agent lived 34.08 steps under learner seed
137 but 26.83 under seed 241; random actions lived 32.56 and 27.58 respectively.
Two learner seeds are a pilot, not a reliable estimate of training variability.

## Continued learning after the world changed

The harder world halved food plants, doubled hazard cells, slowed food regrowth,
and increased hazard damage. Each selected/control learner then received 65,536
additional transitions. Policy weights, world predictors, both optimizers,
novelty counts, normalization, and random-generator states carried forward.

The evolved predictors adapted to these new observations: scarcity prediction
error fell from 0.06567 to 0.06354 in search 1 and from 0.06575 to 0.06358 in
search 2. Their survival did not improve. Random actions also reduced prediction
error, from 0.06447 to 0.06250, while their unchanged action rule produced exactly
the same before/after gameplay outcomes under matched evaluation randomness.

Returning to familiar maps exposed another limit: evolved survival fell from
32.66 to 30.43 steps in search 1 and from 30.45 to 29.00 in search 2. The
equal-mixture condition improved familiar survival, while the learning-progress
condition improved scarcity survival. These isolated changes did not produce a
consistent advantage across prediction, familiar play, and adaptation.

These are changes to one world's parameters, not transfer to unrelated games.
The experiment shows continued predictive learning; it does not establish
open-ended self-improvement or autonomous mastery of game objectives.

## Experience, compute, and verification

| Quantity | Actually completed |
| --- | ---: |
| Authored game mechanisms | 1 |
| Outer search seeds / learner seeds | 2 / 2 |
| Search candidate slots / distinct searched genomes | 48 / 32 |
| Initial trained states / continued states | 106 / 18 |
| Executed training transitions | 28,966,912 |
| Repeated actor optimizer sample presentations | 84,934,656 |
| Predictor bootstrap sample presentations, across three predictors | 86,900,736 |
| Actor optimizer minibatches / predictor optimizer steps | 331,776 / 56,576 |
| Separate prediction-probe transitions | 24,576 |
| Game evaluation episodes / executed game-evaluation actions | 4,608 / 112,134 |
| Unique game-evaluation maps | 128 |

Training initialized 26,431 distinct world-seed identities across base and
adaptation streams; 26,426 received an action. Probe streams initialized another
1,029 worlds. These streams are reused across candidates and controls, so summed
episode presentations are not counts of unique worlds or distinct mechanisms.
The audit verified that 32 repeated initial genome/learner combinations produced
identical final parameter states. Their repeated training still consumed the
reported experience and compute.

Evolution and random search had equal total search budgets. Fixed controls had
equal **per-learner interaction budgets**; each search used twelve times the
training budget of one fixed condition. Random actions trained predictors but
did not update the actor. None of these differences is hidden as equal wall time.

The independent audit replayed every prediction-probe observation/transition
using scalar dynamics and reset logic, and all 4,608 game episodes using separate
scalar transition/fitness code. It recomputed all candidate prediction errors and
selection, checked 124 saved states and continuation lineage, counted consumed
updates, verified split identities, and checked 1,566 animation decisions and
their probabilities against saved models. It did not independently replay gradients.

The suite passed 1,217 tests with 29 skips. The shared policy-update refactor also
reproduced the old learner's parameters and training logs exactly in a bounded
parity check. A separate-namespace end-to-end smoke run and its audit passed
before launch. Training completed in 1,068 seconds (17.8 minutes), peaked at
473 MB resident memory, and added no swap. The final audit peaked at 543 MB.
Paid compute and foundation-model calls were both zero.

Source revision: `826983775676e1cf28f95decbe84cbf48aa27979`. Exact source copies,
probe banks, all checkpoints, logs, and action traces remain in ignored
`output/curiosity-foraging-v1/`. Public receipts include the original hashes;
the earlier survival-selected experiment is unchanged. No training remains running.

## What to investigate next

Scaling this particular search is not supported by its result. A useful next
diagnostic would ask whether the curiosity signals seek **learnable information**
or merely unfamiliar/unpredictable observations. Randomized respawns and
in-sample prediction progress are plausible failure modes, not established causes.
A fixed-world respawn control and explicit random-noise distractors could test
those hypotheses without adding task rewards.

The prediction test itself also has a scope limitation: its observations come
from random exploration, so random-action data collection matches its distribution.
Future task-blind evaluation should examine diverse, predeclared observation/action
distributions. These game results are now exposed; future claims need new reserved
worlds or games rather than repeated tuning against this same bank.
