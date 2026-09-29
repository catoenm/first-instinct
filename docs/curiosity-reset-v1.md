# Curiosity: reset and learning-duration diagnostic

**Completed:** [audited results](curiosity-reset-v1-results.md). Longer learning
worsened fresh-respawn gameplay; repeated maps prevented much of that decline but
did not produce competent play. The prediction check failed. No extension or
cloud rental followed. The original prospective source/protocol copies are preserved.

Prospective follow-up to the [failed curiosity pilot](curiosity-foraging-v1-results.md).
The user authorized further diagnosis and permits RunPod if useful. This tiny
experiment is qualified locally first; no new funding allocation is assumed.
No game score, survival reward, food bonus, or terminal penalty enters training.
All game evaluations run only after every condition and checkpoint is frozen.

## Questions and controlled changes

1. Do respawns receive disproportionately large internal rewards?
2. Does reusing the current map on respawn change exploration and prediction?
3. Does four times the per-agent learning duration improve the result?
4. Does prediction progress on a training batch coincide with improvement on
   observations that were withheld from predictor training?

Use the existing 40,042-parameter learner and equal weights on observation novelty,
predictor disagreement, and in-sample prediction progress. No genetic search,
new reward formula, larger network, or game-outcome-based selection is performed.
Internal rewards and prediction objectives remain explicit optimization targets.

The four conditions are curiosity/fresh respawns, curiosity/repeated-map respawns,
uniform random actions/fresh respawns, and uniform random actions/repeated-map
respawns. Random-action controls still train world predictors. Every condition
starts from the same initial weights and action random stream for each learner seed.

Both reset treatments change all 32 parallel maps every 256 environment steps
on an external clock. Between changes, starvation or the world's 128-step horizon
causes either a fresh map or a reset of the current map's original seed, position,
food, and energy. Horizon resets count as respawns too. Reset labels never enter
policy inputs, rewards, return boundaries, or optimizers; an external observer
records them after actions. A scheduled change takes precedence over a coincident
respawn. Different map diversity and repeated experience are part of this
intervention; it is not an isolated test of sensory randomness alone.

## Fixed schedule and measurements

Namespace `curiosity-reset-v1`; learner seeds 337, 541, 809. Train each of twelve
learners for 1,024 updates of 1,024 executed transitions. Preserve checkpoints at
64, 256, and 1,024 updates without restarting environments or learning state.
This gives 12,582,912 total training transitions and 1,048,576 per learner.
The final duration is four times the previous pilot's per-agent duration.
No candidate or checkpoint is selected by game results or development errors.

Probe banks cross fresh/repeated respawns with uniform random actions and a
predeclared directional sweep (four moves per direction, phase offset by lane).
Each bank has 4,096 transitions. Four monitoring banks and four separate final
banks give 32,768 executed probe transitions. Monitor a fixed random subset of
up to 128 transitions per bank at update 1 and every 32 updates; measure their
error immediately before/after predictor updates without training on them.
Final banks are opened only after training is frozen. Report prediction error
separately for ordinary transitions, respawns, and scheduled changes when present.
The 128-step probe streams contain no 256-step scheduled changes in the main run.

Evaluate every saved checkpoint on 64 new familiar-rule maps under all three
learner seeds with matched action-sampling randomness: 2,304 episodes on **64
unique maps**, not 2,304 independent maps. Report survival, food, and completion
descriptively after freezing. No changed-world adaptation or unrelated-game transfer
claim is planned. Policy and predictor changes, sample consumption, initialization
presentations, distinct initialized/stepped seeds, and checkpoint lineage are separate.

The predeclared prediction check requires repeated-map curiosity to reduce final
**ordinary-transition** error by at least 5% versus both fresh-map curiosity and
repeated-map random actions, averaged across seeds separately for random and sweep
probe distributions; no seed may lose to either comparator. Average the two reset
probe treatments equally. Game scores never enter this check. Report early-to-late
changes regardless of direction. No automatic extension, genetic search, or rental
follows this run, even if the check passes.

## Qualification and accounting

Reuse `games_lab.curiosity.Learner.train` with optional observation stream,
read-only diagnostics, and milestone saves. Test that diagnostics/checkpointing
leave final parameters and early-checkpoint parameters exactly unchanged; verify
default behavior against the frozen previous source. Poison authored reward outputs
to test isolation. Test repeated and fresh resets and matched scheduled changes.
Run focused tests, full discovery once, then a separate-namespace end-to-end smoke
and independent label/trajectory/checkpoint audit before the main run.

Record all training actions and reset categories for auditing, not as new examples.
Expected actor minibatches: 73,728; predictor optimizer steps: 24,576. Repeated actor
sample presentations: 18,874,368; predictor bootstrap sample presentations across
three predictors: 37,748,736. There are twelve independent learners and 36 milestone
states; the final root save duplicates the last milestone, not additional learning.

Keep source copies and hashes, checkpoints, raw traces, and private launch/audit
helpers under ignored `output/` and `.local/` locations. Preserve previous receipts.
Use one local CPU process under the existing 4 GiB / 30 minute / 512 MiB added-swap
guard and an internal 1,500-second budget. Qualify throughput before launch. Do not
load the 9B foundation on this Mac. A cloud run requires the existing remaining
budget and a qualified launcher; it is unnecessary if this diagnostic fits locally.

This diagnosis is motivated by the known distinction between predictable novelty
and stochastic observations, discussed in the primary research on
[large-scale curiosity](https://pathak22.github.io/large-scale-curiosity/) and
[exploration via disagreement](https://pathak22.github.io/exploration-by-disagreement/).
Our small, feature-based learner is not a reproduction of either paper.
