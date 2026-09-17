# Calibration laboratory: protocol before final evaluation

This is an authored, controlled experiment about the meaning of a model's
probabilities. It is not a reconstruction of Jev's unpublished training method,
an architecture novelty claim, or a test of language understanding.

## Question

Does optimizing a decision policy's reward make its action probabilities usable
as beliefs about events? Compare accuracy rewards with rewards for probability
forecasts, ordinary supervised probability losses, and post-training temperature
scaling. Then use every forecast in the same act-or-inspect program at new costs.

## Environment

Each one-step episode has a hidden binary event, an announced event prior, and
an announced sensor reliability. A noisy sensor returns a positive or negative
reading. The three model inputs are prior, reliability, and reading. Training
never supplies the exact posterior probability; evaluation computes it with
Bayes' rule. The environment is deliberately simple enough to audit exactly.

Training and validation priors are uniform between 0.15 and 0.85; sensor
reliability is uniform between 0.60 and 0.90. Final evaluation includes the same
distribution, weaker sensors (0.51–0.59), stronger sensors (0.91–0.99), and priors
outside the training range (0.02–0.14 or 0.86–0.98). These shifts are separate
reports, not extra training data. Identical observed states can legitimately have
different outcomes because the event is uncertain.

## Models and objectives

Use a small network with two 32-unit hidden layers and hyperbolic-tangent
activations. This isolates the reward question from language pretraining and
keeps every experiment runnable on a personal computer's central processor.
It is separate from the released First Instinct text model.

1. **Supervised log loss:** one binary logit; observe the event label and minimize binary cross entropy.
2. **Supervised quadratic loss:** the same network; observe the label and minimize `(forecast - outcome)^2`.
3. **Accuracy-reward policy:** sample a binary action; receive reward 1 if correct, 0 otherwise; update from log action probability times reward advantage. Its probability of choosing action 1 is evaluated as a forecast to test whether that interpretation is valid.
4. **Forecast-reward policy:** choose a probability report from the fixed grid 0, 0.05, …, 1. Receive reward `1 - (report - outcome)^2`; update from the sampled report's log probability times reward advantage. No derivative passes through the environment's reward. The output policy has 21 actions. The returned forecast is its mean report; also report the modal report and the expected reward of its actual sampled-report policy. The probabilities over report actions are not themselves event probabilities.
5. **Temperature scaling:** fit one temperature to the accuracy policy on a separate labeled calibration sample. This is an additional labeled-data step, not reward-only training.

The first four methods see equal episode budgets and matched environment random
streams per seed. Their feedback and output heads differ, so this is a comparison
of recipes, not a claim that they receive identical information. In particular,
a correct/incorrect reward and a known binary action reveal the event label;
this simple environment gives reinforcement learning no information advantage.

Use seeds 11, 23, 37, 53, 71. Preliminary tuning uses seed 101 only and validation,
never final evaluation. Freeze the final training budget after the preflight.
Select supervised checkpoints by their corresponding observed validation loss;
select reward-trained checkpoints by their own observed validation expected
reward, including the initial checkpoint. The exact posterior is excluded from
checkpoint selection and calibration. Preserve every seed and learning curve.

## Evaluation

Primary probability measures: expected binary Brier score, root mean squared
error against the known posterior, and expected log loss. Also report hard-choice
accuracy, stochastic binary-choice accuracy, and reliability bins. A constant
0.5 forecast is included: marginal calibration alone can conceal an uninformative
predictor. Exact posterior and nearest-grid forecasts provide oracle references.

Downstream code selects positive, negative, or optional inspection. A positive
mistake costs `t`; a negative mistake costs `1-t`. Perfect inspection costs `c`
and then reveals the event before the second action. Evaluate thresholds
0.10, 0.25, 0.50, 0.75, 0.90 and inspection costs 0.02, 0.05, 0.10, 0.20.
Report expected cost and regret relative to the exact-posterior policy, plus
realized costs. The downstream program is fixed; it is not a policy learned by
multi-step reinforcement learning. Cost changes require no retraining.

Validation, temperature fitting, and final evaluation use distinct random seeds.
Use 8,192 validation episodes, 8,192 calibration episodes, and 32,768 final
episodes per domain. Freeze all checkpoints and temperatures before reading the
final test. Never select a seed by test performance. Report variability across
seeds and preserve per-example predictions for reconstruction.

## Prior work and claim boundaries

Described-label classification is established: [universal classifiers](https://arxiv.org/abs/2312.17543)
and [GLiClass](https://arxiv.org/abs/2508.07662) are close precedents; GLiClass
already studies reinforcement learning for classification. [OpenJev](https://github.com/TheoLeeCJ/openjev)
and [Jevlike](https://github.com/vinnylarouge/jevlike) explore related open decision
interfaces. These are related implementations, not evidence of Jev's internals.

The mathematics of proper probability rewards is established in
[Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
The sampled-action gradient follows established
[policy-gradient methods](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html).
[Temperature scaling](https://arxiv.org/abs/1706.04599) is also established.

TypeSafe describes reinforcement learning for calibrated decisions but does not
publish the exact reward formula in its [primer](https://docs.typesafe.ai/introduction/machine-learning-primer).
This experiment tests plausible design principles. It cannot identify their
algorithm, validate their calibration claims, or demonstrate a new learning
method. The intended contribution is an inspectable experiment and explanation.

## Preflight decisions and frozen budget

The 600-update development run confirmed the gradient checks, but the
accuracy-reward policy also learned a weaker decision boundary from scratch.
To avoid confusing poor initialization with probability semantics, add a paired
continuation: initialize both runs from each seed's selected supervised-log-loss
model. Continue one with supervised log loss and the other with binary-action
rewards. Both see the same fresh environment stream and additional episode
budget. Fit a separate temperature to each reward-trained model afterward.
This adds labeled pretraining to the continued reward model; report that budget.

The final recipe uses **2,000 updates of 1,024 episodes** per training stage,
Adam with learning rate 0.001, gradient norm clipped at 5, and validation every
100 updates. There is no entropy bonus or reference-policy penalty. The first
stage sees 2,048,000 episodes per seed and method; continued models see another
2,048,000. There are five seeds, six training recipes, and two temperature-scaled
variants. Every method's environment stream is independent of its action sampler.
The sampled-action estimator uses an independent-episode leave-one-out baseline.
An enumeration test verifies its expected gradient against the exact reward
objective; another test confirms no gradient passes through an environment reward.
