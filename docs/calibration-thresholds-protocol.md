# Follow-up protocol: recover a belief by changing the cost of a mistake

This follow-up was designed after observing the first calibration experiment.
It is not a preregistered component of that first experiment. It uses new
validation and final-test seeds; the first experiment's checkpoints stay fixed.

## Hypothesis and known mathematics

For event probability `p`, a false positive costs `t` and a false negative costs
`1-t`. The optimal binary action is positive exactly when `p > t`. Therefore,
the fraction of thresholds in `[0,1]` at which the optimal policy chooses
positive equals `p`. A policy's *response across costs* can encode a belief even
when its probability of acting at one fixed cost does not.

This is an application of established links between proper probability losses
and cost-sensitive decisions, not a new theorem. See
[Reid and Williamson (2011)](https://jmlr.csail.mit.edu/papers/v12/reid11a.html).
For a forecast `q` used as a decision threshold, integrating mistake cost over
uniform `t` gives `0.5 * [p*(1-q)^2 + (1-p)*q^2]`: half the expected binary Brier
score. Our trained policy is a more general function of state and cost; finite
training does not guarantee monotonicity or exact probability recovery.

## Training and reporting

Add a fourth input to the small two-layer 32-unit network: a random threshold
`t`, uniform on `[0,1]` and drawn independently of the noisy-sensor episode.
Sample a binary action and return reward `1 - realized mistake cost`. Train
with the same detached-reward, leave-one-out policy-gradient estimator, Adam
at 0.001 and gradient clipping at 5. No labels or exact posterior enter the
policy observation. No derivative passes through the reward. This is a new
1,249-parameter model trained from scratch, without language pretraining.

Select checkpoints every 100 updates by expected policy cost on **observed
validation outcomes**, with one fixed random threshold per validation episode.
The returned forecast is the mean probability of a positive action over 129
uniform threshold midpoints. Also report the untransformed action probability
at cost threshold 0.5; this directly separates the two interpretations of the
same network. Count curves that increase with cost by more than 0.0001.
The 129 evaluations can be batched but have a real compute cost. This is not a
claim about Jev's sampler or a recommendation for an efficient production model.

Use seed 101 for development only. The planned final budget is 2,000 updates
of 1,024 episodes, with seeds 11, 23, 37, 53, 71. Freeze or document revisions
after development, before final evaluation. Do not choose a final seed by test.

Validation uses event seed 199001 and threshold seed 199101, 8,192 episodes.
Final evaluation uses seeds 199003, 199103, 199203, 199303 for the same four
domains as the first experiment, with 32,768 episodes each. Training event,
threshold, and action seeds are respectively 500000, 600000, and 700000 plus
the training seed. Preserve all checkpoints, curves, predictions, and traces.

Evaluate all fixed first-experiment checkpoints on these fresh final episodes
as reference methods. Their hyperparameters were developed earlier, so this
is an explicitly motivated follow-up, not symmetric fresh model selection.
Compare proper forecast quality, yes/no accuracy, and the same 25 fixed
act-or-inspect settings. Publish failures and distribution shifts as well as
the main result. No further tuning after this follow-up final test is opened.

## Frozen after development

The seed-101 pilot selected update 1900 and produced validation posterior error
0.0855 for the integrated forecast versus 0.2487 for the raw action probability
at threshold 0.5 (root mean squared error). Retain the planned 2,000-update
budget and 129-point integration. No further development runs were used.
