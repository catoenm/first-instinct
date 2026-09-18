# A general decision model: the intended target

Status, September 18, 2026: the nine-billion-parameter supervised model is
trained, evaluated, released and serving the local demo. It improved held-out
accuracy from 63.3% to 78.1%. Four completed language reinforcement-learning
runs did not reliably improve decisions; the demo therefore retains the
supervised checkpoint. See the [supervised results](general-supervised-results.md)
and [reinforcement results](general-reinforcement-results.md).

A new [executable-outcome comparison](outcome-v2-protocol.md) is running from
that same supervised checkpoint. Outcome-only, reward-only and combined
training each have two seeds. Its 59,993 verified questions concern action
consequences and future costs in two sequential environments. No improvement
from this new experiment has yet been established. The earlier software model
remains a separate, narrower experiment.

The target is a small, open model that accepts arbitrary supplied state and
user-defined questions, option descriptions and ordered levels. It should answer
new questions across domains with constrained probabilities, without generating
prose. Multiple questions should be independent. TypeSafe's public interface
motivates this target; its private architecture and Reinforcement Learning for
Calibrated Decisions recipe are unknown.

## What is actually implemented

The generic Qwen scorer already accepts supplied state, question and 2–36 option
descriptions. It reads allowed label-token logits directly from the network in
one forward pass per question. It does not generate text for another classifier
to read. The general browser demo exposes four editable examples; the underlying
interface also accepts a visitor's own question and choice definitions.

The released software adapter trains 32,464,896 parameters in low-rank updates
to language attention and feed-forward projections. Original matrices and the
vocabulary output projection are frozen; internal computations change through
the learned updates. This is standard low-rank adaptation, not a new architecture
and not just a newly fitted output classifier. Merging the updates into original
matrices would change checkpoint storage, not add learning.

The software adapter received outcome-supervised training on one event family.
The separate small models received reinforcement learning for probability
reports and evidence acquisition. That older four-billion-parameter language
adapter itself did not receive reinforcement learning. This differs from the
newer nine-billion-parameter language reinforcement runs, which did update
internal language adapters from sampled rewards, but did not reliably improve
held-out decisions.

The new corpus contains 350,857 prepared training rows and 112,309,610 input
tokens. Its 43,278,336 trainable language-adapter parameters affect the existing
nine-billion-parameter foundation's attention and feed-forward computation.
The output projection retains the foundation's vocabulary weights; it is not a
new fixed-label classifier. The completed reinforcement-learning stage used
these same language adapters, with a value head used only during training. The
new outcome comparison isolates the value head from language gradients and
adds explicit forecasts of terminal outcomes and future costs.

The generic interface accepts independently specified choice, binary and
ordered questions over one supplied state. For example:

```bash
python -m general_lab.interface --input examples/general-decisions.json \
  --model qwen35-9b --run /path/to/completed/general/run
```

Omitting `--run` uses the untouched foundation. Each question has a separate
forward pass; this does not implement shared-state attention reuse. Current
limits are 1,536 input tokens per question and 2–36 offered choices.

Two supplementary checks now address this interface directly: a
[typed-question robustness audit](general-robustness.md) and an isolated
[shared-state inference prototype](shared-state-experiment.md). Neither changes
the running experiment's inputs or checkpoint selection. The
[external data review](executable-data-next.md) identifies actual third-party
tool environments to broaden the executable mechanisms beyond authored worlds.

## Revised sequence

1. Establish a general interface and untouched baselines. Allow supplied state,
   multiple independent questions, dynamic choices, binary propositions and
   ordered descriptive levels. Report constrained probabilities accurately;
   do not label every distribution as calibrated. Compare the original foundation
   and specialized adapter on the same requests. This interface alone is not a
   trained mini-Jev.
2. Train broad decision behavior in the language backbone. Combine reasoning
   over supplied rules, document evidence, semantic judgments, numerical and
   program state, tool choices and outcome forecasts. Change question wording,
   option descriptions, ordering and number, including genuinely ambiguous and
   insufficient-evidence cases. Deduplicate sources before splitting. Hold out
   entire question families, source documents and domains, not only rows.
3. Train the same language network in decision environments. Sample its actions,
   observe verified outcomes and use Proximal Policy Optimization to update its
   trainable backbone parameters. Retain a separate proper outcome-prediction
   loss and appropriate training-only value estimates. Keep event beliefs and
   action probabilities distinct. Compare supervised-only, reinforcement-only
   and combined objectives at matched budgets; do not assume that reinforcement
   learning automatically calibrates forecasts or that a combined objective wins.
4. Measure unseen-task accuracy, probability quality, workflow reward, sensitivity
   to irrelevant changes, option-order effects, and cost. Evaluate common states
   and policy-selected states. Preserve a random audit stream alongside selected
   hard cases. Include simple planners and frozen-model references.
5. Scale the model or update method after this baseline. Compare low-rank and
   broader parameter updates if adaptation capacity appears limiting. Continued
   pretraining on raw text is a separate intervention, justified by measured
   representation gaps. Starting from a pretrained foundation is compatible with
   genuine backbone learning and reinforcement learning.

## What would count as progress

A general demo must let a visitor supply a new question and new answer choices,
not just select from an authored task catalog. A useful model must also improve
on untouched task families; accepting their input format is insufficient.
Constrained output guarantees format, not factual correctness.

A language-model reinforcement result requires auditable gradients and changes
in language-backbone parameters from sampled environment rewards. Report that
separately from supervised changes and temperature fitting. If a small separate
policy makes all sequential decisions, call it a composite baseline.

We should publish the data and measured mechanisms without claiming to have
recovered Jev's architecture, achieved its performance, or invented low-rank
adaptation, constrained token scoring or proper scoring rules.

References: [TypeSafe primitives](https://docs.typesafe.ai/primitives),
[TypeSafe introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[Low-Rank Adaptation](https://arxiv.org/abs/2106.09685).
