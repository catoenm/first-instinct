# Broader decision learning: local mechanics and prospective pilot choices

The live runtime qualification replays actual filesystem, calendar and reservation
actions. Before paid learning, test the exact proposed loss/update/collection path
with a small random network. This is software qualification, not an experiment on
the pretrained 9B model. Do not feed calendar prompts to this network or use
calendar model scores to choose the recipe.

Use the original supervised step2742 language checkpoint for the eventual three
arms: forecast supervision plus general replay, reward-only Proximal Policy
Optimization plus replay, and both plus replay. Preserve identical language starts,
paired seeds, schedules and general pools. The replay-only contribution is common
to all arms. The forecast target is a verified categorical distribution; reuse the
existing proper soft-target loss, including padded-option handling. General replay
continues to use acceptable-answer sets. Never turn possible outcomes into a set
of equally correct answers.

The prospective forecast coefficient is **0.2 in both forecast-bearing arms**.
This is chosen before new model evaluation. In the previous mixed comparison,
initial hybrid training gradients had actor/outcome norms 5.55/35.98 and
5.18/16.78: forecast gradients were approximately 6.5 and 3.2 times larger.
The coefficient tests a weaker competing forecast objective. It is a hypothesis,
not evidence that gradient balance caused the earlier result. New component norms
and cosine similarities will be recorded without changing the coefficient from
transfer results. Retain one optimizer pass per fresh rollout, a language learning
rate of 0.000001 and critic learning rate of 0.0001. Keep general replay weight
0.5, policy clipping 0.2, critic weight 0.5 and entropy weight 0.01.

Action sampling and Proximal Policy Optimization likelihoods use the same mixture
of 80% native model probabilities and 20% uniform exploration. Forecasts use the
native distribution. Check update divergence on **both** native action probabilities
and the behavior mixture, with mean divergence at most 0.02 and any input at most
0.10. Native forecasts must not inherit the exploration floor. Roll back parameters,
optimizer and random state after a failed guard; retain attempted/backward
presentation receipts and stop the arm. Do not retry automatically or publish a
rejected checkpoint. The detached critic cannot change language representations.

For CPU mechanics, use the five training mechanisms only. Freeze source hashes
and selected cases before execution. At most 16 live episodes, 200 offered
actions, three optimizer steps and 120 optimizer-objective example presentations
are allowed, plus at most 240 diagnostic backward presentations without an update.
Each arm starts from the same deterministic random network. Log every
started/completed objective batch, gradient diagnostics, parameter changes,
real trajectory receipts and actual execution counts. Integration tests separately
use at most 20 synthetic optimizer steps and no environment/model-service calls.

Probability reporting must separate expected observed-outcome Brier score,
excess squared distance to the target distribution and the irreducible uncertainty
floor. Average unique public inputs within each mechanism, then mechanisms
equally. Report uncertain cases separately. Preserve source multiplicities in
lineage without treating them as independent tasks. The future pilot must freeze
its actual sampling, live interaction budget, selection and joint transfer/retention
gates, runtime deadline, artifact recovery and remaining original compute budget
before hardware is rented. The local mechanics check does not itself authorize
launch or claim improved decisions.
