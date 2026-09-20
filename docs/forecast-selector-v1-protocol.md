# Offline forecast-to-action diagnostic

This protocol is written after closing expanded-decisions-v1 and before computing
these selection results. It is exploratory development analysis, not a new
transfer experiment or another checkpoint selection. Calendar is excluded.

## Scope and limits

Use only the saved report-development forecasts from the original model and all
six already-selected expanded checkpoints. The original is represented by
outcome-1507's audited baseline evaluation. Do not reselect checkpoints, fit a
temperature, change a threshold, or query any model. Use the existing shell
curriculum's executed report branches as truth. Maximum scope: 36 concrete
cases, 308 unique forecast inputs, seven predictors, two continuation contracts,
and the corresponding 504 forecast-labelled branches. No GPU, new task execution,
optimizer update, training question, or external service call is allowed.

First freeze hashes of this protocol, implementation, regression tests, final
audit, source qualification, source cases/executions/questions, frozen development
inputs, selected run metadata and the exact saved forecast files. The runner must
refuse changed inputs, duplicate/incomplete menus, incompatible outcome vocabulary,
nonfinite probabilities or label/receipt disagreement. Preserve failed attempts.

## Decision contract

For every unique visible decision input, collect every offered action under each
of two existing contracts separately:

- `stop_now`: execute that one action, then stop. An inspection cannot receive
  credit for a later repair that never happened.
- `evidence_then_commit`: execute that action, then follow the already published
  deterministic, observation-only continuation within the remaining horizon.

For each action, use its stored forecast to compute
`P(completed) - P(incorrect) - public immediate action cost`.
Choose the maximum; break exact ties by lexicographic action ID. This is explicitly
an **immediate-cost heuristic** for the continuation contract. It does not know
realized future costs, case identity, hidden files, truth vectors or verifier
results. Feed only the public menu/state and predicted outcome probabilities to
the selection function. Never silently substitute expected private future costs.

Evaluate the chosen action by averaging its independently verified full branch
return and outcomes across all compatible equally weighted authored worlds.
Group identical visible histories before selecting or maximizing. Require exactly
one branch per compatible case/action/contract. All model probabilities must have
the same input and outcome vocabulary as the corresponding executed forecast.

## Comparisons and accounting

Report each predictor and both contracts separately, with per-regime results:
mean realized return, completion/incorrect probabilities, full cost, regret to
the information-respecting best offered action, and fraction of optimal choices.
Average visible contexts equally; also show case-weighted values separately.
These are finite authored-prior expectations from saved branches, not newly
executed model rollouts or real-world frequencies.

Add three diagnostic references:

1. Uniform three-way probabilities under the same public-cost selector.
2. The exact executed outcome distribution under the same immediate-cost
   heuristic. This uses verifier knowledge and is an oracle diagnostic, never a
   deployable learned controller.
3. The maximum expected verified full return after averaging compatible worlds.
   This oracle also measures regret and useful-action menu coverage. It cannot
   choose a different action for an unobserved hidden world.

Compare references 2 and 3 to isolate the potential gap from missing continuation
costs even with perfect outcome probabilities. Count uncertain inputs and actions
with varying future cost, and preserve their receipt lineage. Keep underlying
roots, world-and-goal tasks, cases, visible contexts, branches, forecast inputs and
reused prediction presentations separate.

Do not treat these results as a matched comparison with the live actor: its prompt
and free replanning contract differ. A positive diagnostic only justifies a
locally qualified future experiment. A failed one does not justify more GPU time.
Any new learning comparison still requires broader qualified data, a fresh
reserved mechanism, identical starts and prospective joint improvement checks
within the original remaining compute authorization.
