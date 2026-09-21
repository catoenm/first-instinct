# Real tool steps with terminal rewards passed local qualification

The retail environment now exposes individual tool choices. A policy can read the
current order or account, attempt a scoped change, observe the real response, and
choose again. The menu and arguments come from the public history and requested
goal. They do not change based on hidden state.

Each attempted tool costs its declared fee, including documented refusals.
Intermediate rewards contain only that negative fee. The independent goal,
preservation and payment checks pay 20 simulated units once, when the policy stops
or reaches the six-turn limit. There is no free success signal while it acts.
Unexpected infrastructure errors abort the episode and are not failed-task labels.

Three scripted controllers—stop, inspect then act when useful, and attempt the
wrong-scope change—ran on all 20 previously qualified goal/configuration pairs.
Their actual commands, responses and final states exactly matched the existing
counterfactual witnesses. Each episode also passed independent reward accounting
and an exact replay in a fresh process.

| Runtime validation | Count |
| --- | ---: |
| Source goals / mechanisms / distinct initial states | 3 / 2 / 10 |
| Primary episodes / independent replays | 60 / 60 |
| Actual tool calls / actor turns including stop | 102 / 222 |
| Charged expected errors, including replays | 10 |
| Covered read/write cost pairs | 6 |
| New independent tasks / model calls / optimizer updates | 0 / 0 / 0 |

The inspection controller satisfies 16 of 20 goal/world pairs; immediate stopping
satisfies nine. Four goals are unreachable with these offered commands from their
particular worlds. A nominally wrong-scope attempt still ends with the goal intact
in five cases because it is a harmless no-op or refusal and the requested goal was
already satisfied. This is why actions are labelled from executed consequences,
rather than from command names. These scripted-controller counts are neither model
accuracy nor a return comparison: the qualification cycles different fees across
identities to check the runtime, and future learning must use matched cost schedules.

Six focused tests cover terminal reward exactly once, the last-turn fee, absence
of hidden success feedback, reset/observation isolation, invalid actions and
unexpected exceptions. Six saved-receipt corruption controls additionally reject
hidden feedback, premature payout, free inspection, changed menus, contaminated
resets and an inflated total reward. Every prior source freeze remains unchanged.
The six-turn cutoff test uses a stub callback; the real qualification controllers
all stopped explicitly within the bound.

This is the live stepping interface needed for later reinforcement learning, not
a completed learning run. It uses unchanged real simulator tools and authored
fixtures; it is not an official benchmark run, a Harbor run or dynamic command
generation. All related retail worlds retain one training-candidate ownership
group. No retail questions have been admitted to training yet.

Next, admit paired consequence and decision questions with displayed, executable
continuations, verify their token boundary and data ownership, and qualify the
learning integration locally. Any paid comparison must retain identical starting
weights, general-task replay, untouched transfer evaluation and predefined stop
checks. Extra hardware alone does not resolve those requirements.

[Prospective protocol](retail-live-v1-protocol.md) ·
[Aggregate checks](../results/retail-live-v1/summary.json) ·
[Source evidence](retail-evidence-v1-results.md)
