# Next decision-model experiment: public evidence and objective audit

Research and source audit, 2026-09-18. No additional model inference, API
inference, training, GPU rental, or external dataset download was performed for
this memo. This is a prospective experiment proposal, not a description of
TypeSafe's private method or a new positive result.

## What the public Jev material establishes

TypeSafe documents a useful behavioral target: one state plus user-defined
typed questions, distributions over the supplied alternatives, and ordinary
code composing the answers into a workflow. Its three primitives are Choice,
Score, and Noul. Questions in one request are evaluated independently; a real
dependency on an answer requires a subsequent request. This supports building
a general question-conditioned predictor and a transparent controller around
its outputs. It does not imply that a direct action-policy softmax is a set of
action-success probabilities. [Introduction](https://docs.typesafe.ai/introduction),
[question semantics](https://docs.typesafe.ai/primitives).

The company describes RLCD as training for decisions and calibrated
probabilities, and describes calibration as a property of groups of predictions.
The reviewed public primer does not specify the optimizer, loss equation,
training mixture, reward construction, architecture internals, or data scale.
We cannot infer PPO, a particular proper score, or an auxiliary outcome loss
from the name RLCD. Our proposed PPO plus empirical categorical log loss is
our own controlled recipe using established techniques.
[TypeSafe primer](https://docs.typesafe.ai/introduction/machine-learning-primer),
[original PPO paper](https://arxiv.org/abs/1707.06347).

The launch article claims a new architecture and parallel sampler. It also
states that its workflow evaluations use averaged external-model predictions
as reference probabilities. Those workflow comparisons are not measurements
against executed real-world outcomes. Its zero-hallucination discussion is
specifically justified by guaranteed schema matching; valid types do not prove
semantic correctness. Our finite-choice interface can reproduce that narrow
format property, but neither Jev's internals nor its latency and cost claims.
[Launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

TypeSafe's confidence field is a statistic of the returned distribution; the
reviewed confidence page does not disclose its exact formula. Keep our event
probabilities, distribution entropy, and chosen-action probability separately
named. Do not relabel an invented confidence formula as TypeSafe's.
[Confidence documentation](https://docs.typesafe.ai/confidence).

## Audit of the released trainer

The relevant local sources are [rl.py](../general_lab/rl.py),
[rl_mechanics.py](../general_lab/rl_mechanics.py), and
[the scoring implementation](../scale_lab/model.py). The negative result and
its four runs remain documented in
[general-reinforcement-results.md](general-reinforcement-results.md).

I found no obvious sign, detached-old-likelihood, option-mask, or return-cost
error in the existing PPO calculation. The collector sums subsequent rewards,
including evidence costs; its old values and log probabilities are detached.
The hybrid's independent forecast stream uses sampled binary outcomes, and
selection uses exact expected validation reward. These are strengths worth
preserving.

Concrete limitations that the next trainer should address:

1. **Component interaction is not measured.** `update` combines an actor with
   normalized advantages, unnormalized value MSE at weight 0.5, entropy,
   forecast log loss, and replay, then clips the combined backbone-and-critic
   gradient to norm 1. Across the four recovered runs, median pre-clipping
   norms are 9.2–11.5 and maxima are 42–54. The pure-policy audit measures only
   the first microbatch, not the complete actor gradient. Neither these norms
   nor clipping alone establishes which loss dominates; Adam also prevents a
   simple interpretation as an equal reduction in effective learning rate.
   Before optimization, record full-rollout component norms, pairwise gradient
   cosines and clipping coefficients. A critic on detached language features,
   with its own optimizer clipping group, removes an unnecessary source of
   shared-backbone interference for this small experiment. Verify that critic
   loss alone gives zero language gradient while actor and outcome losses each
   give finite nonzero gradients.

2. **The logged KL precedes the optimizer step.** The existing update measures
   approximate KL before `optimizer.step`, so it misses the last epoch's
   actual change. The first updated-policy rescore reached approximate KL
   0.0337 and a 16.8% sampled-action clip fraction. Rescore the full valid action
   distribution after every committed optimizer step, including forecast and
   replay effects. A predeclared KL guard can skip remaining epochs; record
   the committed update rather than silently rolling it back. This bounds
   local steps, not cumulative drift from the starting model. Measure retained
   general performance separately.

3. **Existing mechanics checks are necessary but incomplete.** They test 32
   sampled training episodes and fail if unchanged-policy sampled-action
   ratios leave the PPO clipping interval. They do not test post-update
   improvement, every candidate's log ratio, rare outcomes, multiple
   mechanisms, or all new prompt lengths. Extend the unchanged-weights test
   to the new shortest/longest prompts, option counts and padding boundaries.
   On a tiny CPU policy, compare the sampled-gradient expectation with an
   exactly enumerated policy gradient and verify that one sufficiently small
   actor-only step increases known expected return.

4. **Forecast quality has no enforced connection to the native actor.** The
   original action question and hidden-event question share weights, but
   action probabilities are not computed from event forecasts and utilities.
   The post-hoc forecast controller only stops immediately. Neither design
   establishes that improved forecasts improve evidence acquisition.

5. **Mechanism count is the meaningful data gap.** The prior worlds already
   cover inspection, affirm/deny and abstention. New sequential mechanisms,
   counterfactual outcome coverage and harder compositions are more valuable
   than additional domain names. The prior hybrid also receives additional
   labeled examples, so it cannot isolate reinforcement from extra outcome
   practice without an outcome-only control.

The norm/KL observations above were recomputed from the four files matching
`results/general-reinforcement-v1/runs/*/training.jsonl`; they are diagnostics,
not a causal explanation of the negative result.

## Correct per-action outcome contract

For visible history `h`, candidate action `a`, explicit continuation policy
`pi_c`, and deadline `H`, predict a categorical distribution
`P(Y | h, do(a), pi_c, H)`. The outcome categories must be mutually exclusive,
exhaustive, and sufficient for the utility calculation. Record the continuation
version, remaining horizon and relevant history. The probabilities sum to one
**within an action's outcome categories**, not across the action menu.

A binary exactly-once-success target loses important information if missed
execution and duplicate execution carry different penalties. This tiny fixture
must pass before training:

| Action | P(missed) | P(exactly one) | P(duplicate) |
| --- | ---: | ---: | ---: |
| A | 0.10 | 0.80 | 0.10 |
| B | 0.20 | 0.80 | 0.00 |

With utilities `(-1, +1, -10)`, expected utilities are A = -0.30 and B = +0.60.
With utilities `(-10, +1, -1)`, they are A = -0.30 and B = -1.20. The preferred
action reverses despite identical success probability. Keep full categories,
including distinctions involving delay or terminal state when utility uses them.

Compute utility as the sum of category probabilities times declared utilities,
minus correctly accounted action and continuation costs. If continuation cost
depends on the observed branch, use its verified expectation or include
cost-relevant distinctions in the outcome distribution. A fixed scalar charge
cannot stand in for different branch-dependent costs. If utility is nonlinear
in outcomes and costs, their joint distribution may be required.

Inspection needs a continuation that uses the result. As an exact fixture,
let an event have probability 0.5; acting earns +1 if true and -1 if false;
skipping earns zero. Perfect inspection costing 0.1 followed by acting only
when positive is worth 0.4. Inspection followed by immediate stopping is worth
-0.1. A stop-now target therefore rejects a valuable inspection. Include both
fixtures as an event-contract test, not a model benchmark.

A fixed continuation makes forecast labels stable. If deployment repeatedly
replans with a different policy, those probabilities still refer to `pi_c`,
not automatically to the deployed adaptive policy. Evaluate the actual complete
controller by executing or enumerating its own decision tree. Do not report
one-step fixed-continuation probabilities as calibrated end-to-end success
probabilities for a different controller.

## Minimal controlled next protocol

Use the same Qwen3.5-9B foundation, starting adapter, finite-choice language
interface, replay stream, reward units and frozen data definitions in all arms:

| Arm | Language-backbone training |
| --- | --- |
| Outcome-only | Empirical categorical outcome log loss + general replay |
| Reward-only | PPO actor objective + general replay; detached-feature critic |
| Hybrid | PPO actor objective + the same outcome loss and replay |

Pair the two reinforcement arms on root-world sequences and exogenous random
seeds. Pair outcome-only and hybrid on identical independent forecast worlds,
candidate actions, continuations and outcome draws. Keep forecast data separate
from policy-selected stopping states. Replay, outcome-label reuse and optimizer
steps must be counted: identical world IDs reused twice are not two independent
outcomes. Describe the hybrid's extra simulator work and tokens explicitly.

Prefer the same language learning rate and optimizer-step schedule across all
three arms. Using 1e-5 for outcome-only and 3e-6 for hybrid creates a preventable
optimization-dose confound. A separate higher-rate fitting pilot is reasonable
engineering, but its result should not be used to claim that adding PPO caused
an improvement over a matched outcome control.

Predeclare **the same forecast-to-utility controller's validation reward** as
the primary selection metric for all arms, including update zero. This answers
whether training improves the same deployable workflow. Use transparent fixed
normalization to average across mechanisms with different utility ranges, while
also reporting raw per-mechanism reward. Do not mix native actor and controller
scores into an opaque aggregate. Report the native actor at these checkpoints
as secondary; an independently predeclared actor-selected R/H comparison can
be reported separately, without choosing the selector after seeing test results.

Add a general-validation retention condition to selection. Fix its subset,
task/world weighting, tolerated log-loss regression and accuracy regression
before training. These are engineering non-regression thresholds, not a theorem
of retained ability. Reserve final outcome test worlds and mechanism
compositions until selection is fixed. Existing previously inspected general
test/prose sets remain useful regression suites, but are not newly untouched
evidence. Report categorical log loss and Brier score per mechanism/outcome,
controller reward, native reward, duplicate/missed rates, acquisition cost and
paired whole-world intervals. Two seeds are a small reproducibility check,
not a precise estimate of training-seed uncertainty.

For compute control, the arithmetic remainder of $500 after $64.27 is $435.73,
before storage and reconciliation with the actual provider invoice. A fixed
short-run allocation should be only part of that remainder. Freeze the maximum
updates/episodes, process deadline and independent provider billing stop;
reserve evaluation and recovery time. Do not change the cap or extend a run
because its latest point estimate looks promising. The proposed bounded pilot
can answer whether the new targets and mechanism transfer work before scale.

## Launch gates

1. Replayed transitions reproduce final outcome categories and all costs from
   immutable receipts. Timeouts or unresolved outcomes follow a declared
   censoring/category rule; they are not silently relabeled as observed failure.
2. Enumerated outcome mass sums to one for every action. Categorical utility
   equals direct expected rollout return, including delayed effects and
   continuation costs. The two fixtures above pass.
3. Model-visible serialization and candidate menus are identical for hidden
   worlds with the same visible history. Hidden outcome, seed, verifier facts,
   exact action values and private branch choices never enter the prompt.
4. All retries, counterfactual actions, paraphrases and related continuations of
   one latent world share a split. Changing only utility leaves outcome labels
   unchanged when the transition law and fixed continuation are unchanged.
5. The new scorer passes numerical/padding checks and component-gradient checks;
   the critic has zero gradient into language parameters. Store post-step full
   action KL and actual label/step counts.
6. The controller computes candidate utilities solely from model probabilities,
   public utility/cost rules and the declared continuation. The exact planner
   is an evaluation reference, never a hidden candidate selector.
7. Report candidate forecasts, total input tokens, wall time and environment
   calls for the controller as well as for the native actor. Batched repeated
   full-state forwards do not reproduce Jev's shared-state parallel sampler.

None of these gates or observations recovers private RLCD. They make a small
decision model's claims testable and its next experiment interpretable.
