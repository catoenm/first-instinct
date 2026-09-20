# Scale the verified curriculum before the foundation

We have a useful general constrained-answer model, but not evidence of a small
Jev with reliable general-purpose decision and consequence forecasts. Our original
supervised adaptation consumed 350,857 examples and improved held-out choices.
The later expanded environment comparison covers only 40 world-and-goal tasks
from six authored roots, and failed its joint transfer gate. The
[saved-forecast diagnostic](forecast-selector-v1-results.md) also found elementary
confusion between an unfinished task and a successful outcome.

Keep the existing nine-billion-parameter foundation for the next supervised
comparison. Increasing parameter count now would confound capacity with data,
and would not establish that the consequence labels are being learned. This is
a data and learning hypothesis, not proof that capacity never matters.

## The target capability

TypeSafe's public [primitives documentation](https://docs.typesafe.ai/primitives)
describes user-defined choices, ordered scores and binary judgments, with
independent questions about one shared state. Our constrained-choice interface
can express these answer sets. We have not demonstrated comparable breadth,
calibration, speed, shared-state computation or efficiency. The documentation
does not disclose enough to identify their private training recipe. Reproducing
the observable capability is a more defensible goal than claiming to reverse
engineer Jev.

## Broader execution first

AppWorld is the next external substrate, following the existing
[source review](executable-data-next.md). Its pinned local training inventory has
90 task instances from 30 generator programs. Seven payment-related programs
(21 instances) are reserved as a prospective whole-mechanism holdout. The other
23 programs (69 instances) are candidates, not qualified training data. Keep
upstream development/test contents untouched. Qualification begins with four
small workflows and strict no-op/reference/replay controls; see the
[local protocol](appworld-local-v1-protocol.md).

Before expanding collection, qualify at least eight distinct usable training
programs across several application operations. Use the real application
implementations, inspect each evaluator's active assertions, add wrong-target
and collateral-change controls, and propagate infrastructure errors. Reject an
entire task program when its verifier cannot cleanly distinguish these cases.
Preserve the useful existing ToolSandbox, database, filesystem and shell families;
their completed data must not be regenerated merely to increase counts.

The first collection should remain bounded: up to 32 distinct task instances,
two declared public history points per task, four proposed actions per history,
and one independent replay of each branch. These are ceilings for a prospective
pilot, not work already executed. Record finite menus, useful-action coverage,
verified terminal and intermediate predicates, and total action costs separately.
Reference solutions may validate the environment; their private arguments must
not leak into a model's menu or observed state. A later dynamic proposer gets its
own coverage measurement, independent of selector quality.

## Supervise more than the next action

From each qualified visible history and executed alternative, create matched
questions about:

- Whether the requested goal is already satisfied, remains unfinished, or was
  made impossible under the declared task rules.
- A specific immediate state change, such as whether a chosen record was updated,
  whether an unrelated record changed, or whether a prerequisite remains missing.
- Completion under one explicit, public-information continuation, and its expected
  additional action cost. Keep immediate and continued outcomes distinct.
- The best offered next action and whether a further observation is worth its
  cost under a specified information-respecting policy.

Retain both success and failure branches, already-completed goals, failed calls,
recovery paths, and worlds that genuinely remain ambiguous under the same visible
evidence. The model never supplies its own truth. Do not invent a probability from
the fraction of evaluator assertions that passed. Exact state predicates and
finite-world outcome distributions are separate supervision types.

Pair changes in goals, observations and costs while holding command wording
constant. Vary question predicates and answer descriptions, not just entity names.
Keep related histories, paraphrases, counterfactual worlds and task generators in
one partition. Inspect shortcut baselines before calling the data diverse.

If this collection qualifies, expand toward tens of thousands of distinct
verified questions by adding actual task programs and worlds. A 50,000–100,000-row
target is a planning range, not an independent-task count or a promise that the
current 90 instances suffice. Report generator programs, worlds, histories,
executed branches, questions and repeated presentations separately.

## A bounded supervised learning gate, then reinforcement learning

Start both the unchanged control and richer supervised candidate from the same
original step-2742 adapter. Keep general-task replay and the existing foundation
revision. Freeze new development/transfer ownership before model evaluation and
check question/option robustness as well as exact forecast quality. Neither the
opened calendar nor the original report diagnostic becomes a pristine new test.

A small supervised pilot must demonstrate useful learning of the new predicates,
better decisions and consequence probabilities on development programs withheld
from training, and acceptable retention. Predeclare thresholds and equal token
budgets after the qualified dataset and variance are known, before inference.
Also report the unchanged foundation through the same interface where affordable.
Do not use a larger model or more epochs to conceal a failed small pilot.

Only then expand supervised training or repeat the forecast-only, reward-only
Proximal Policy Optimization, and combined comparison from identical starts.
Forecast-driven control is a separate ablation with the same information and
continuation contract. Preserve the distinction between direct action selection
and selection from predicted consequences.

All current source qualification is local and uses no paid model services.
Any GPU pilot remains inside the original cumulative personal $500 authorization,
after reconciling billing and retaining recovery reserves. Existing permission
already covers qualified training; no new approval question is needed.
