# First Instinct: a decision interface is the beginning

*Public-facing draft, September 18, 2026. The six-arm outcome-training study is
ongoing; this note reports completed work only.*

Letting a language model answer new questions from supplied choices is
straightforward. Making its probabilities useful for decisions across
unfamiliar tool behavior is harder. That is the distinction I have been
exploring with [First Instinct](https://github.com/catoenm/first-instinct), an
open, Jev-inspired learning project. The aim is an inspectable small decision
model. We have no claim to Jev's private architecture, training recipe or
performance.

The interface accepts a state, a question and descriptions of the permitted
answers. Those answers can change with every request. We pass the question and
options through the nine-billion-parameter Qwen3.5-9B and read its existing
next-token scores for labels such as A, B and C. Converting these permitted
label scores into probabilities gives a distribution;
ordinary code can use it to choose an action. The same interface supports
binary questions and ordered scales. Flexible output types are useful, but
their validity says little about whether the chosen answer is right.

Training changes the language network internally. Low-rank adaptation (LoRA)
adds trainable matrices throughout its language layers. We trained 43.3 million
such parameters; the original foundation matrices remained frozen. Their
combined computation changes as the adapters learn. There is no generated
paragraph followed by a new classifier reading that paragraph. This is
adaptation of a pretrained nine-billion-parameter model, rather than training
that foundation from scratch.

One supervised pass over **350,857 examples** improved accuracy from **63.3% to
78.1% on 17,277 held-out questions** under the same constrained-answer
interface. The mixture combines public instruction tasks, executable reasoning
worlds and simulated decisions. Gains were strongest on generated familiar
families. Several public sources regressed, and the small prose-pair evaluation
did not establish improvement. This is one training seed; public material held
out from our adaptation could still have appeared in foundation pretraining.
The [supervised report](general-supervised-results.md) retains those limits and
the failures.

A smaller diagnostic made the probability problem especially clear. On **48
authored configurations**, a post-hoc foundation control found that adaptation
raised mean deterministic accuracy from **82.6% to 100%**. Probability error
against exact finite distributions fell from **46.9 to 17.6 percentage points**,
measured as root mean squared error. The adapted model chose the more likely
outcome on every probability question, yet its probabilities remained wrong.
These configurations have correlated wording and evidence variants; they are
not hundreds of independent tests. See the [paired control](general-robustness-foundation-results.md).

For example, one stated random experiment makes orange occur with probability
70.8%. The model assigns orange 97.0%. Choosing orange is correct, but using
97.0% to decide whether to accept a costly failure risk can be wrong. This is
why we separately measure selected-answer accuracy, distance from known event
probabilities, and the consequences of using forecasts in a controller. A
plausible-looking distribution is insufficient evidence of calibration.

Reinforcement learning has already happened inside the language adapters.
Our first four runs used Proximal Policy Optimization, with and without extra
outcome exercises. They **did not reliably improve held-out decisions**. Three
selected the unchanged supervised start, and every latest checkpoint lost
reward under shifted conditions. Outcome exercises limited probability drift
relative to reward-only training without establishing a better policy. The
[negative result](general-reinforcement-results.md) is part of the project,
and the supervised checkpoint remains the demo default.

The most promising next work is improving what the data makes the model learn.
In a new [ToolSandbox pilot](toolsandbox-partial-v1-results.md), labels come
from actual tool execution and checks of the complete resulting database.
One query ranks fuzzy text matches, keeps five, and then applies a time filter.
Five irrelevant off-window matches can therefore hide a real reminder: the
query returns empty even though the desired row exists. A time-only query
finds it. Trusting the empty response can cause a missed update or an actual
duplicate creation.

We declare four possible database templates and their prior weights publicly.
A phone lookup supplies the current contact name; worlds with identical complete visible
histories remain indistinguishable. Executing an offered action and a fixed
continuation in each compatible world gives exact conditional outcome and cost
distributions. The collection has **48 public contexts, 1,152 distinct
world/program labels and 432 conditional forecasts**. Each program was executed
twice, with matching receipts. Private state and verifier labels stay outside
the model's inputs.

This produces a useful decision reversal. In one update scenario, a complete
query is preferable when each returned row costs one research credit. At twelve
credits per row, stopping is preferable. Costs depend on actual returned rows,
and the stated continuation can make mistakes. These probabilities describe
that continuation; a controller that repeatedly changes its plan needs its own
executed evaluation. This is verified data from one authored mechanism, not a
learned calibration result or a realistic estimate of production priors.

A [frozen transfer check](toolsandbox-transfer-supervised-results.md) on this
mechanism exposed a substantial gap. Using the current supervised model's
forecasts to choose an action selected **stop at all 48 initial states**, although
stopping was optimal at only **17**. The resulting expected regret was **20.4
research credits**, identical to always stopping. All 720 probability questions
were retained and independently rescored. Those questions share correlated
contexts; they are not 720 independent tasks. This evaluates choices implied by
forecasts under the declared continuation, not a new executed policy or the
model's direct action answers. The outcome-trained checkpoints have yet to face
this same check. A correct interface and good answers on simpler fixtures did
not establish useful transfer here.

A [post-hoc answer-order check](toolsandbox-transfer-order-results.md) exposed
another weakness. Reversing the choices changed the most likely future cost on
**69 of 96 questions**. The winning outcome answer stayed the same on all 144
outcome questions, yet individual outcome probabilities moved by as much as
**27.6 percentage points**. Only three of the 48 implied initial actions
changed; the planner still stopped in 45. The original and reversed calls were
made at different times, without contemporaneous repeats. This is a bounded
sensitivity finding, not a chosen prompt improvement or an explanation of the
whole transfer failure. It reinforces the need to evaluate probability behavior
separately from selected-answer accuracy.

There is also a concrete efficiency result. Reusing a shared state prefix for
eight independent questions took **3.50 seconds versus 12.07 seconds for an
ordinary batched complete forward**, a **3.45× ratio of medians** on our Apple
M5 Max. All answers were preserved, including the known error of selecting a
16-centimeter carton for an 18-centimeter item. The [batch control](shared-prefix-batch-control-results.md)
used three trials per condition, float32 weights and reference kernels. It is
a local implementation measurement, not a claim about optimized graphics
servers or Jev's speed.

The [ongoing six-arm study](outcome-v2-protocol.md) compares outcome learning,
reward learning and their combination across two seeds, starting from the same
supervised model. Executable retry and workshop environments provide outcomes
and future costs; we will report differing data and computation. No interim
validation numbers establish its final result.

Progress would mean better forecasts and executed decisions on held-out
mechanisms while retaining general question answering. If the hybrid cannot
improve on outcome learning alone, we should not credit its reinforcement step.
If gains disappear with new query semantics, we need broader mechanism coverage.
More rows and valid output types alone would not answer either question.
