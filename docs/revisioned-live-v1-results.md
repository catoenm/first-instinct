# Live choices and immediate forecasts over conflicting database updates

The new actor interface can choose again after each observed command result.
It executes the selection in an isolated SQLite database and obtains reward from
an independent final-state check. It does not inherit the earlier examples'
fixed continuation or use a model prediction as a reward label.

Qualification exercised every terminal path of the four-decision contract:

| Quantity | Count |
| --- | ---: |
| Existing post-schedule physical states | 2 |
| Existing world/goal tasks | 4 |
| Cost profiles | 3 |
| Primary executed branches | 1,422 |
| Exact independent replay branches | 1,422 |
| Actor actions, including stop, across both collections | 10,164 |
| Actor SQL statements, excluding transaction control | 9,600 |
| Failed or refused commands | 5,064 |
| Distinct public decision contexts | 258 |
| Prepared immediate forecast questions | 3,096 |
| Forecast distributions retaining uncertainty | 108 |

There were also 2,844 initial reads and 1,932 scheduled colleague writes. These
counts include the replay collection. They are executions of one mechanism,
not thousands of independent tasks. Qualification used scripted actions to cover
paths; it measured no language model's decision performance.

For example, after a stale version makes a conditional write refuse, the actor
can inspect the new row and retry. The same retry can violate an edit authorized
only for the original revision. A command returning normally is therefore not
the same event as satisfying the goal. Read costs and failed attempts remain in
earned reward, and already satisfied goals can justify stopping immediately.

At each public history, every command was executed in every compatible world.
Repeated prefixes and replay runs add no probability mass. The declared equal
initial prior remains uncertain in 24 visible contexts; observations identify the
world in the other 234. The 3,096 questions divide equally between immediate goal
status and immediate command return code. They do not forecast an unspecified
learned continuation.

Independent goal/response reconstruction, exact replay, serialization, five
actor tests, and four corrupted-data controls passed. All 3,354 distinct actor
and forecast inputs fit the pinned Qwen tokenizer without cropping; the longest
is 736 tokens. The guarded job took about 20 seconds with approximately 828 MiB
peak process-group memory and no additional swap.

These forecasts remain candidates, not a consumed training mixture. Scripted
qualification trajectories are explicitly ineligible for on-policy training.
The [learning adapter](decision-learning-v2-results.md) now passes separate CPU
mechanics checks, but no new 9B checkpoint or GPU run resulted from this stage.
Dynamic command proposals, additional mechanisms, GPU runtime behavior, and
unfamiliar-task transfer remain separate work.

[Protocol](revisioned-live-v1-protocol.md) ·
[Aggregate receipt](../results/revisioned-live-v1/summary.json)
