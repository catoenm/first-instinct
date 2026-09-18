# What makes decision-model data trustworthy?

Our working thesis is that a useful decision model needs more than a large table
of questions and preferred answers. It needs a record of **what was visible,
what could have been done, how evidence was selected, and what actually happened**.
The environment determines whether those fields mean what we think they mean.

This is our research direction, not a claim about TypeSafe's private training
method or a claim of novelty for proper scoring rules or information acquisition.

## Three concrete findings to discuss

**1. A proper reward does not guarantee useful probability learning.**
We built a software environment where a policy can pay to inspect checks, then
report the chance that the complete fixed suite passes. The terminal reward is
one minus squared probability error, with inspection costs subtracted.
Several policies trained only through rewards collapsed toward constant reports.
Direct outcome supervision gave better forecasts. Starting with that predictor
and using Proximal Policy Optimization to learn inspection produced useful
behavior, but a simple empirical transition planner remained stronger.

The interesting architectural question is which parts should learn from a
verified label and which parts need a sequential reward. A probability report
has a direct differentiable loss once its event resolves. Deciding whether to
buy evidence depends on its future benefit and cost. Sharing a backbone is
possible; it does not require using the same objective for both jobs.

**2. The evaluation population changes when the model selects evidence.**
Checking forecasts only where a policy chose to stop can hide weaknesses on
other states. Our environment enumerates seven possible evidence states per
program and evaluates those same states for every policy, as well as evaluating
the paths each policy actually takes. Extra forecast practice on broader states
did not consistently beat equally much practice on selected terminal states.
That negative result is part of the experiment.

A larger system should preserve an independently sampled audit stream alongside
hard examples and policy-selected interactions. Log selection probabilities
where they are known. Weighting can help study a changed sampling distribution
when the necessary coverage and assumptions hold; it cannot recover outcomes
for regions that are never observed.

**3. Tool traces and outcome labels are different data products.**
In our first 12,963-row TOUCAN shard, all 54,100 calls had linked responses.
Yet 2,492 calls lacked an exact match in the recorded tool declarations.
Some appear to be aliases or version differences. A model-judge completeness
score is also not an independently verified outcome. Before using a trace to
teach decisions, we need the tool inventory that actually applied at that moment.
Before using it to teach success probabilities, we need an explicit success test.
An extension to 35,227 rows across three teacher subsets found 1,266 identical
questions shared across two subsets. Teacher identity alone is therefore not a
safe train/test boundary. See the [audit and its sampling limits](toucan-data-audit.md).

## The next data investment

The current [software corpus](software-inspection.md) has 7,793 program variants,
but only 369 source functions from one repository. Its seven related views per
variant become 34.6 million training input tokens after length checks. Row count
alone misses both the compute budget and the amount of independent information.

The next expansion should prioritize:

- **Independent sources:** additional repositories, dependencies, real bug fixes,
  and multiple edit generators. Reserve entire repositories and generators for
  evaluation before generating variants.
- **Independent checks:** human-written tests plus separately specified
  properties. Original-program agreement measures compatibility; it can inherit
  the original program's mistakes. Keep verifier disagreements and audit them.
- **Meaningful counterfactuals:** preserve the same hidden event while changing
  evidence availability, evidence duplication and inspection cost. Keep actions
  that change the world in a separately defined event/transition contract.
- **Failure and recovery:** retain failed calls, stale evidence, missing
  information and recoverable errors. A success-only imitation dataset cannot
  describe the uncertainty in those cases.

The [language-model pilot](software-outcome-model.md) now beats the evidence-
frequency reference on its selected held-out samples. A code-removal control
would test how much of that gain depends on the program itself versus the
visible checks. Next, train acquisition with the large forecaster, holding the
initial predictor and interaction budget fixed. Compare ordinary interaction with matched extra practice on
terminal states and on a representative audit stream. Measure forecast quality
on common states, workflow reward, inspection cost and duplicate sensitivity.

The deliverable is an inspectable data-and-environment factory, with immutable
execution receipts and measurements that can disprove the preferred story.
That is a more useful foundation for scale than multiplying paraphrases of the
same source task.
