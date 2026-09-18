# What the 9B model does in the verified reservation environment

The supervised Qwen3.5-9B model can complete some reservation workflows, but
recovery from combined faults and consistency across wordings remain weak.
Its forecasts under a specified repair procedure also need work. These are
development results from **one authored mechanism**, not a general tool-use
benchmark or evidence that model size is the cause.

No weights changed. The two fixed studies made **332 local model calls**:
254 adaptive decisions across 72 episodes, then 78 forecasts of 39 distinct
events. All calls used the already resident supervised step-2742 checkpoint.
No additional compute was rented. The native CUDA PufferLib trainer was not
run, and these were inference studies rather than reinforcement training.

## Decisions that execute

The [prospective decision protocol](reservation-language-v1-protocol.md) fixed
24 initial cases: 16 familiar cases and eight combined-fault cases, spanning
two price schedules and two turn budgets. Each ran with original choices,
reversed choice order, and rewritten choice descriptions/question. Every
chosen action executed in the unchanged C simulator previously checked against
real SQLite transactions. This run did not open new SQLite databases or Harbor
containers.

| Presentation | Familiar success | Combined-fault success | Familiar mean return | Combined mean return |
| --- | ---: | ---: | ---: | ---: |
| Original | 11/16 (68.75%) | 1/8 (12.5%) | 0.6419 | -0.0506 |
| Reversed menu | 11/16 (68.75%) | 2/8 (25%) | 0.4000 | -0.3181 |
| Rewritten choices/question | 11/16 (68.75%) | 1/8 (12.5%) | 0.5194 | -0.0650 |
| Authored inspection/repair continuation | 10/16 (62.5%) | 4/8 (50%) | 0.5644 | 0.4200 |
| Finish immediately | 0/16 | 0/8 | 0 | 0 |

Returns include action costs and penalties for leaving partial reservations.
Equal success rates therefore do not mean equivalent behavior. Reversing the
menu increased combined success by one case but produced a substantially worse
mean return. The mechanism's terminal penalties make those failures costly.

The existing 7,370-parameter numerical policies, restricted to the exact
matching previously evaluated profiles/worlds, each achieve 16/16 familiar
and 4/8 combined success. They were trained on this mechanism and receive a
different representation; this is a learnability control, **not a fair model
size or architecture comparison**.

## Wording, state, and recovery

Deduplicating initial public states leaves eight distinct starting histories.
Rewritten choices change the first action in **four of eight**; reversal changes
it in **one of eight**. The primary case-paired figures are 12/24 and 2/24,
respectively. Repeated initial inputs across hidden worlds are not independent
wording tests. The audit found 82 repeated presentations and no action changes
between identical presentations.

For matched full public histories, the original and reversed presentations
agree on 56/68 case-paired actions; original and rewritten agree on 35/50.
Diverged trajectories are excluded from these comparisons. Different actions
can both be reasonable, so disagreement alone is not proof of an error.

Post-hoc checks identify concrete recovery mistakes across the three variants:

- Five attempts to reserve a request whose header is already known to exist.
- Nine finishes with a partial reservation when scoped undo is still available.
- Three undos of an empty request.

There are also 26 finishes without a complete reservation. Some can be sensible
when recovery is impossible within the remaining budget, so that count is
descriptive rather than an automatic error count. No claim is made that these
categories capture every failure or identify globally optimal actions.

The 254 inputs contain 699–800 tokens. Median reported model time is about
2.14 seconds on the resident Mac runtime. This is not a server throughput or
Jev latency comparison.

## Explicit outcome probabilities

The separate [forecast protocol](reservation-forecast-v1-protocol.md) asks what
happens after a specified first action and the exact authored continuation.
It selects all 36 initial-history action targets plus the three noninitial
targets with uncertain success, then queries both answer orders. This is
39 distinct events, including eight uncertain ones, with exact probabilities
from the already executed qualification branches.

Mean squared error below measures deviation from the exact event probability;
lower is better. It combines task understanding with probability quality and
does not isolate calibration alone.

| Event slice | Original answer order | Reversed answer order | Always predict 50% |
| --- | ---: | ---: | ---: |
| All 39 | 0.3243 | 0.2811 | **0.2051** |
| 31 deterministic events | 0.3989 | 0.3479 | **0.2500** |
| Eight uncertain events | 0.0351 | **0.0225** | 0.0313 |

The model substantially underestimates several repairs that reliably succeed
under the stated continuation. A sequential attempt followed by the specified
repair routine succeeds in all six supported hidden worlds with six turns,
but the original presentation predicts about 23.1% success. Its better score on the small
uncertain slice does not offset these deterministic errors. Reversing the two
answer choices changes predicted success by 4.84 percentage points on average,
and at most 9.67 points.

The selection deliberately includes every uncertain target and is not a
population calibration estimate. These opened development questions must not
be reused as an untouched test set for subsequent training. No temperature
fitting, prompt changes or checkpoint selection followed these results.

## Data built from the findings

The new [command-consequence curriculum](reservation-dynamics-data-v1.md)
derives four immediate-state questions from each qualified context/action:
request status, account existence, A stock, and B stock. Labels come from the
actual SQLite state after the specified command, conditioned on the public
history and prior. They are not model-authored guesses.

The [development data pack](../results/reservation-language-v1/dynamics/) has:

- **1,152 canonical event questions**, with **194 uncertain categorical targets**.
- **3,456 serialized rows** after the three presentations.
- 288 context/action groups and 32 public histories, all within the same mechanism.
- Exact nontrivial probabilities including 1/6, 1/4, 1/3, 1/2, 2/3, 3/4 and 5/6.

This is additional supervision derived from existing experience, not thousands
of new independent tasks. Its group ID groups a public history and its labels;
it is not sufficient to create an independent train/test split across shared
worlds or mechanisms. The pack has no untouched test partition. No model has
yet been trained on it, and we have not measured its benefit.

The next training comparison should first teach and test command consequences
and partial-write recovery, then compare reward-only and reward-plus-forecast
training from the same starting checkpoint. It needs separate new evaluation
cases, stochastic on-policy rollouts, retention checks, and explicit accounting
for the treatment's extra supervision. The current greedy diagnostic actions
are not on-policy samples from the recorded softmax distributions. A longer
paid training run is premature until those data and comparison controls exist.

## Evidence and reproduction

The [evidence bundle](../results/reservation-language-v1/) preserves both
prospective freezes, exact public prompts/messages/token IDs, every response,
all action/state/reward trajectories, case metrics, the new data and its source
bindings. The offline audit replays all decisions, recomputes aggregates,
checks source and checkpoint bindings, and reproduces every saved token ID
from the cached tokenizer. Disk hashes plus service metadata do not attest
resident model tensors.

Run the offline receipt and data checks, without loading a model or contacting
a server:

```sh
python -m unittest test_reservation_language_evidence -v
python -m puffer_lab.dynamics_data verify \
  --output results/reservation-language-v1/dynamics
```

For the additional exact token audit, the pinned Qwen tokenizer must already
be cached locally; no weights are loaded. Write the regenerated report outside
the immutable evidence directory:

```sh
python -m puffer_lab.language_analysis \
  --language results/reservation-language-v1/decision \
  --forecast results/reservation-language-v1/forecast \
  --output output/reservation-language-replay.json
```
