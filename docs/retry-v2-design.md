# Retry outcomes with executable continuations

This is a separately versioned data adapter for a new experiment. It wraps the
audited SQLite engine in `general_lab/retry_environment.py` without editing it.
The original pilot and all released model results keep their original meaning.
No new model result is claimed by this adapter.

The scenario begins with an ambiguous submission timeout. A job may have been
lost, already committed, or still be in transit. The next action can inspect a
receipt, retry, wait, or stop. A shared unique key prevents duplicate side
effects; an unkeyed retry can create a second completion. SQLite executes the
actual insertions and supplies the final count.

## Two forecasts for each offered action

The previous pilot asked whether exactly one job would finish. That combines
two differently harmful failures: zero jobs costs 100 cents, while duplicate
jobs cost 200 cents. The new outcome question instead returns a distribution
over **zero, one, or two completions at the stated deadline**.

The second question predicts the distribution of **future action costs**, in
integer cents. It includes the offered action and all continuation actions,
but excludes costs already paid. The controller can then compute:

```
expected future utility =
    -100 × P(zero) + 100 × P(one) - 200 × P(two) - expected future action cost
```

The two questions describe the same executed rollout. Their probabilities need
not be statistically independent: linearity of expectation makes their
marginal expectations sufficient for this expected-utility calculation.
Probabilities sum to one within each question for each offered action. They do
not sum to one across different actions.

Cost options come from a public enumeration of all legal action sequences that
fit in the remaining time. This intentionally includes some costs that have
zero probability under the fixed continuation. The menu never uses hidden
transport outcomes or the verifier's conditional support. A singleton cost
menu represents a publicly determined cost and does not need a model call.

## The future policy is part of the question

Every forecast names the same transparent `receipt_then_retry` continuation:

1. Execute the offered action. Stop if the episode is terminal.
2. If a visible receipt has reported a positive completion count, or a retry
   was acknowledged, choose `abstain`.
3. Otherwise choose `lookup` if it is unused.
4. Otherwise, if the latest receipt is empty and retry is unused, choose
   `retry`.
5. Otherwise choose `abstain`. Repeat after each response.

Stopping advances to the deadline and does not cancel requests already in
flight. The continuation can therefore inspect evidence and make another
decision. It may also make mistakes: a stale empty snapshot can trigger an
unkeyed duplicate, and an acknowledgement does not reveal the final count.
The rule only sees the same visible history available to the model.

A `stop_now` continuation is available as a diagnostic. After the offered
action it permits time to pass without further paid actions. It is explicitly
different from calling the paid `abstain` action. It is not the primary
forecast target for this experiment.

A controller that chooses the highest predicted utility and forecasts again
after the next observation is a **replanning controller**. Its realized return
must be measured by executing that controller. A forecast made under the fixed
continuation does not magically become an accurate prediction of the entire
replanning policy, and it must not be scored as though it did.

## Splits, labels, and audit boundary

`mechanism_id()` hashes the transition law: deadline, key rule, receipt lag,
normalized transport probabilities, and acknowledgement probability. It
excludes all costs, random seeds, hidden draws, and action paths. Changing only
costs or multiplying probability weights by a common factor cannot move a
related world into another split.

Split ownership reserves entire combinations of deadline, key rule, and
receipt lag; `split_design()` records the exact combinations. Training sees
every individual factor and every pair of factors. Heldout conjunctions are
new for this experiment. Both heldout splits include keyed and unkeyed worlds,
so validation can expose duplicate harm.

| Split | Deadline / key rule / receipt lag |
| --- | --- |
| Train | 2 / unkeyed / 0; 2 / keyed / 1; 3 / unkeyed / 1; 3 / keyed / 0 |
| Validation | 2 / unkeyed / 1; 2 / keyed / 0 |
| Test | 3 / unkeyed / 0; 3 / keyed / 1 |

The deadline is two ticks in validation and three in test; both deadlines occur
in training. This is a specified shift, not a claim that validation and test
are identical distributions. Configuration, trajectory, random exploration, and
forecast-label draws use separate deterministic seed namespaces. This does
not make correlated states or two forecast heads into independent worlds.

`forecast_bundle()` samples a fresh transport tape conditional on the public
history, independently of the actor's hidden tape, then executes the offered
action and the declared continuation in SQLite. The observed completion count
labels the outcome question; the same rollout's actual cost labels the cost
question. It never uses exact probabilities as training targets.

Exact finite probabilities are available through `exact_forecast()` for
verification and calibration evaluation only. The separate replay object
contains the source hashes, public input hashes, draw seed, complete executed
continuation, and database receipt. Keep that object out of model inputs.
Training records contain only the public inputs and empirical category labels.

## Integration and reproduction

```python
from general_lab import retry_v2 as retry
import random

scenario = retry.make_scenario(seed=79, index=0, split="train")
tape = retry.sample_tape(scenario, random.Random(1501))
with retry.EpisodeAdapter(scenario, tape) as episode:
    observation = episode.observe()
    actor_question = episode.input()
    offered_action = episode.legal_actions()[0]
    inputs = retry.forecast_inputs(scenario, observation, offered_action)
    # Query the model only with inputs["outcome_input"] and inputs["cost_input"].
    label_pair = retry.forecast_bundle(scenario, observation, offered_action, seed=1701)
    transition = episode.step(offered_action)
```

Both forecast helpers expose `cost_values`, which maps string option identifiers
to integer cents. `forecast_bundle()` additionally returns string
`outcome_target` and `cost_target` plus the separate `replay` object.
`exact_forecast()` returns `outcome_probabilities` and `cost_probabilities`
using the same option identifiers and exact rational values.

Generate a small pilot without network access, a model, or a graphics processor:

```sh
python -m general_lab.retry_v2 --output output/retry-v2-pilot --worlds-per-split 50
python -m unittest test_retry_v2
```

Each split gets trajectories, forecast examples, and separate replay receipts.
The summary records source hashes, file hashes, root and mechanism counts,
outcomes, and forecast-pair counts. Every visited state is included under a
uniform random exploration policy, with its actual action probability logged.
One completion sample supplies two forecast labels; row count is not the count
of independent observations.

The tests cover grouped ownership, held-out factor combinations, hidden-state
isolation, conditional replay, stale receipts, duplicate penalties, costs and
termination, both continuations, deterministic reproduction, and full generated
receipt replay. A simple verified fixture has expected utility of 87 cents for
inspection followed by the adaptive continuation, versus minus 5 cents when
inspection is followed by passive stopping. Another fixture shows why equal
exactly-one probabilities can still justify different action choices when
duplicate failures are more harmful than missing jobs.
