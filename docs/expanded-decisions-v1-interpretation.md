# What the expanded comparison tests

This note was prepared while the last training arm was running, before opening
the calendar or general-transfer scores. It explains the implemented mechanism;
it neither changes the frozen protocol nor adds an advancement criterion.

The model shares language adapters across two kinds of questions. The live
actor chooses its next action and receives verified terminal utility minus
actual future command costs. The forecast question instead names one specific
action and states exactly what happens afterwards: either stop immediately or
follow an explicit public continuation. Its probability target comes from
executing that contract in every compatible authored world. Those are different
questions, even when the visible state is the same.

The combined arm adds both losses to the same language parameters. It does not
first ask the model to forecast every candidate and then choose the action with
the largest predicted utility. Consequently, a better forecast need not change
the live actor's preferred action. The controlled comparison tests whether the
shared training helps both capabilities transfer; it is not a direct test of a
forecast-based planner or of Jev's undisclosed training procedure.

The reservation forecasts also include binary questions about immediate state
after one command, whereas other families largely ask for a three-way final
outcome. Neither a command's exit status nor a generated model answer supplies
the label. Forecast probabilities need to describe uncertainty conditional on
the visible evidence, not reveal which hidden world the collector chose.

## The reinforcement update is deliberately conservative

The implementation uses the clipped Proximal Policy Optimization objective,
but accumulates one batch of gradients and makes **one optimizer update per
fresh rollout**. It does not run several optimization epochs over that rollout.
With identical rescoring, the initial likelihood ratio is one, so the first
gradient agrees with the unclipped policy-gradient objective. Clipping becomes
relevant when the rescored ratio leaves its interval; it is not what bounds the
size of this single committed update.

In the actual initial diagnostics, gradient-enabled rescoring differs slightly
from rollout inference. The sampled ratios remained inside the clipping
interval in all four completed reward-bearing arms. For the two combined arms,
the recorded ranges were 0.940–1.078 and 0.963–1.027. Every update performs an
unchanged-policy check, although only the initial detailed receipt is retained.
The separate post-update divergence checks on native and exploration
probabilities, with complete rollback on failure, are the actual commit guards.
The detached critic estimates remaining reward without training the language
representation through its own loss.

Initial combined-arm actor gradient norms were 4.036 and 4.038; unweighted
forecast norms were 15.750 and 9.536. The frozen forecast coefficient of 0.2
scales those forecast contributions to approximately 3.150 and 1.907 before
combining gradients. Their initial actor/forecast cosine similarities were
-0.027 and 0.011. These are single training-batch diagnostics, not evidence that
gradient alignment explains any eventual transfer result. No coefficient was
changed after seeing them.

## A useful next diagnostic, after closing this comparison

A small, separate evaluation could test whether the existing forecasts can
support decisions directly. For each visible history, score the same proposed
action menu, declare one continuation, and choose using predicted terminal
utility minus only the costs available to the controller. Evaluate that choice
against independently executed branches. If future costs are not predicted or
publicly known, an immediate-cost heuristic must say so; it must not use private
executed future costs to choose an action.

This diagnostic must separate three questions: whether the menu contains a
useful action, whether the forecasts describe that action under the stated
continuation, and whether the selection rule uses them well. A fixed-continuation
selector and a free-replanning actor are different controllers; their results
cannot be presented as a matched training comparison. Any analysis of the
already-opened calendar set would be exploratory, not another untouched test.

For observation-value targets, first average action outcomes across worlds
that are indistinguishable to the controller, then choose an action. After an
inspection, regroup worlds by its actual visible result before choosing again.
Averaging the best action separately in each hidden world gives the controller
information it never observed. The value of an observation is the difference
between those two executable, information-respecting policies minus its cost.

This is a direction for local qualification, not permission to silently expand
the current run. Preserve its selected checkpoints and complete joint result
first. New paid work still needs a bounded prospective comparison and a fresh
reserved mechanism, within the original remaining compute authorization.

Implementation references: [learning update](../tool_lab/expanded_learning.py),
[live actor](../tool_lab/decision_rl.py),
[training and checkpoint selection](../tool_lab/expanded_train.py), and
[saved development receipts](../results/expanded-decisions-v1-development-audit/).
