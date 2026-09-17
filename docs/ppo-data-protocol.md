# Protocol: training method, model capacity, and data coverage

This experiment follows the published calibration laboratory. It is a new
controlled experiment with fresh validation, calibration, and final-test seeds.
No claim about Jev's private method is tested. The numeric simulator is an
environment: `reset` supplies observations, `step` returns rewards and terminal
flags. Episodes contain one decision and require no hosted service.

## Questions and comparisons

1. Does Proximal Policy Optimization improve on the simple sampled policy gradient?
2. Does widening both hidden layers from 32 to 128 units help either recipe?
3. Does broader data coverage help at fixed capacity and episode budget?
4. Do action probabilities remain different from event probabilities with a stronger learner?

Both widths receive a supervised-log-loss baseline. At each width, train
21-action forecast-report policies from scratch using either the simple
REINFORCE estimator or Proximal Policy Optimization. For the correctness-reward
comparison, start both algorithms from the same selected supervised model for
that width and seed. Continue supervised training from that same checkpoint as
an additional control with the same total experience. This removes avoidable
initialization and additional-data confounds.

At width 128, additionally train supervised, REINFORCE-forecast, and
Proximal-Policy-Optimization-forecast models on the expanded data mixture.
This yields **15 recipes × five seeds = 75 trained policies**, with 25 auxiliary
value networks. The seeds are 11, 23, 37, 53, 71. Development uses seed 101 only.

Small policies have 1,217 parameters for a binary output or 1,877 for 21 reports.
Large policies have 17,153 or 19,733 respectively. Each Proximal Policy
Optimization recipe trains a separate value network of the corresponding width;
its parameters and update count are reported separately. It shares no weights
with the policy. All networks use two hyperbolic-tangent hidden layers.

## Actual Proximal Policy Optimization implementation

Use the clipped probability-ratio objective from
[Schulman et al. (2017)](https://arxiv.org/abs/1707.06347).
Each rollout stores sampled actions, their old log probabilities, rewards, and
old value estimates. Old quantities and reward advantages are detached. For
these terminal one-step episodes, the return is simply the reward, with no
bootstrap term. Normalize the fixed rollout advantages by their standard
deviation. Reuse the rollout for at most four full-batch policy updates, with
ratio clipping at 0.2 and early stopping if the estimated policy divergence
exceeds 0.03. Clipping and early stopping discourage large changes; they are
not hard guarantees of a maximum change.

Fit the separate value network to observed rewards using four squared-error
updates. Policy learning rate is 0.0003; value learning rate is 0.001. There is
no entropy bonus or reference-policy penalty. Adam and gradient norm clipping
at 5 are used throughout. The simple policy-gradient and supervised recipes use
learning rate 0.001 and one update per fresh rollout. Their reward gradient is
the existing independent-episode leave-one-out estimator.

These are **experience-matched training recipes, not compute-matched algorithms**.
Proximal Policy Optimization reuses each rollout and trains an additional network.
Preserve update counts, timing, probability-ratio clipping statistics, and value
losses. No single component such as clipping can be credited for any observed
difference between the complete recipes.

## Data intervention

The narrow generator uses priors in 0.15–0.85 and sensor reliability in 0.60–0.90.
The expanded generator allocates exactly one quarter of each batch to each of:

- The original narrow distribution.
- Priors 0.15–0.85 with weak or misleading sensors, reliability 0.10–0.60.
- Priors 0.15–0.85 with strong sensors, reliability 0.91–0.99.
- Rare or very common events: priors 0.02–0.14 or 0.86–0.98, reliability 0.60–0.90.

The number of episodes is unchanged. Expanded training sees fewer examples of
the original distribution, so any loss there is also reported. This is a data
coverage intervention, not a larger-data intervention. The sensor reliability
is announced to the model; a reversed sensor is informative in the opposite
direction, not an adversarially false reliability claim.

Final tests contain 32,768 fresh states each from the original distribution,
weaker sensors (0.51–0.59), stronger sensors (0.91–0.99), extreme priors,
reversed sensors (0.10–0.40), and a **held-out combination** of extreme priors
with reversed sensors. The expanded generator sees the two ingredients of that
last domain separately but never together. Neither training nor validation
contains that joint combination. Record the data generator and stream seeds.

## Selection and evaluation

All recipes use the same 8,192-example **narrow validation sample**, selecting
their own observed objective: log loss, sampled correctness reward, or sampled
forecast-report reward. This keeps the selection data fixed when changing the
training-data mixture. The initial checkpoint is eligible. The true conditional
probability is excluded from all training and selection. The selected supervised
model anchors its corresponding two correctness-reward continuations.

Fit one optional temperature per correctness policy on a separate 8,192-example
narrow calibration sample. This adds labeled feedback and is reported as a
separate derived variant. It does not change the main policy comparisons.

Primary measures are expected binary Brier score, root mean squared error against
the exact conditional probability, and expected cost in all 25 existing fixed
act-or-inspect settings. Also report hard-choice accuracy, sampled-action accuracy,
log loss, reliability bins, and the forecast policy's sampled-report and modal-
report scores. The reported forecast from a report policy is the mean report;
the policy distribution over report actions is not itself an event forecast.

Every width and algorithm sees matched environment streams within a data regime
and training stage. The correctness continuations use a fresh matched stream.
The training stages plan 2,000 rollouts of 1,024 episodes, checking validation
every 100 rollouts. Continuations consume an additional stage after supervised
pretraining; do not compare their total budget as if they started from scratch.

Validation seed: 7100001. Calibration seed: 7100003. Final seeds: 7200001 plus
100 times the domain index. Training seeds use bases 3100000 for narrow initial
training, 3200000 for expanded training, and 4200000 for continuations, plus the
model seed. Model checkpoints and temperatures must be fixed before generating
the final test. All seeds and all domains will be published, including failures.

## Development and stopping rule

Run development with seed 101 and validation only to check the environment,
clipping implementation, numerical behavior, and budget. Record any changes
here before committing the frozen protocol and opening the final test. Do not
retune after observing final results. A new method or data intervention after
that point needs a separately reserved evaluation.

## Frozen after the seed-101 preflight

One 400-rollout pilot exercised all 15 recipes and both model sizes. All
updates were finite, rollout reuse was recorded, and the exact clipping-gradient,
reward-detachment, data-coverage, and checkpoint-initialization tests passed.
Retain the planned 2,000-rollout budget and all hyperparameters above. The pilot
showed that increasing width did not uniformly improve forecast-policy training;
this is a reason to preserve the full matrix, not select one favored condition.
No final-test sample was generated during development.
