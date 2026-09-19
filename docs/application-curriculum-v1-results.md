# Application decisions with real local tool effects

The next local curriculum qualification passed. It adds **one new application
mechanism** using the pinned ToolSandbox messaging and device-setting tools:
complete a simulated delivery task, avoid duplicates, satisfy connectivity
prerequisites, and preserve unrelated settings. All targets come from actual
database execution and an independent final-state verifier. Dataset collection
involved no real messages, model calls or GPU rental. None of these rows has
been used for 9B training; tiny-network mechanics tests are accounted separately.

These are our authored fixtures using upstream implementations, not the official
ToolSandbox benchmark. The [prospective protocol](application-curriculum-v1-protocol.md)
defines the finite prior, continuations, costs and collection bounds.

## What a task looks like

The goal is to ensure exactly one specified message exists for a chosen recipient.
The visible history may omit the ledger, include a current lookup, show an
explicitly old lookup, or contain an actual failed send. The same offered
commands appear across different goals and hidden worlds.

If the message already exists, stopping earns full completion credit. Sending it
again creates a second real database row and incurs the irreversible-error penalty.
If cellular service is disabled, sending raises a connection error. Low-battery
mode must be disabled before cellular can be enabled, and disabling it does not
itself re-enable service. Turning low-battery mode on can change protected Wi-Fi
and location settings, which the independent verifier treats as a mistake.

The receipts also expose a useful subtlety: a failed setting change can reveal
information about prerequisites. In some hidden contexts, trying to enable
service first has slightly higher expected return than a read-first plan.
Action labels therefore depend on **the declared continuation**, including its
later inspection costs. We do not call them unrestricted optimal actions.
Decision and observation-value rows remain diagnostics for the initial planned
comparison; consequence supervision and live reward learning stay separable.

## Qualified data, with repetitions separated

| Quantity | Accepted collection |
|---|---:|
| New authored mechanism / root fixture | 1 / 1 |
| Concrete initial database worlds | 9 |
| World-and-goal tasks | 18 |
| World/goal/context variants | 84 |
| Distinct executed action-and-continuation alternatives | 2,268 |
| Additional exact replay executions | 2,268 |
| Total production and replay branches | 4,536 |
| Setup / visible-prefix / future-action tool calls | 22,788 / 4,536 / 6,940 |
| Total top-level tool calls | 34,264 |
| Forecast / next-action / observation-value questions | 1,512 / 26 / 26 |
| Identical-input forecast groups with different real outcomes | 108 |
| Questions consumed by 9B training | **0** |

Calls count the top-level offered/setup calls, not upstream internal helpers.
Replays verify determinism; they are not additional independent labels. The
[receipt bundle](../results/application-curriculum-v1/) contains all accepted
cases, alternatives, questions, verifier distributions, source hashes and counts.

All question inputs fit the proposed 1,536-token limit: maximum 947 tokens for
a prepared question and 1,181 for an actor question over an executed history.
The audit checked 1,094 distinct actor histories without truncation. On the 18
fully observed contexts, the best possible predictor restricted to the command
menu achieves only 33.3% of the executed action labels. Removing observations
also leaves a 33.3% ceiling; removing the goal leaves 61.1%. Both goal and state
matter. These are structural shortcut checks, not trained-model scores.

The audit rebuilt requests, costs, histories, queries, protected-state checks,
rewards, menus and every saved question/label. Across Python 3.11 and 3.14,
aggregate floating-point utilities differed only at rounding scale; the audit
permits 1e-12 tolerance for derived aggregates while requiring exact questions
and categorical labels. Eight integration/contract tests cover recovery,
duplicates, hidden uncertainty, historical-world separation, changed-goal
rejection and verifier negative controls.

## Rejected work stays visible

The first collection completed the same number of alternatives but failed its
isolation-accounting gate. urllib3 attempts an IPv6 socket capability probe
during import; the interception guard blocked it. The collector compared final
counts against its pre-import probes without persisting an execution-phase
boundary. We excluded that collection and retained its exact source and receipts.

The corrected bounded replay records the blocked import probe separately and
requires zero additional socket or subprocess attempts during task execution.
It also gives the historical world a separate deterministic UUID scope so it
cannot alter current fixture identities. The first collection's **4,536
executions and 34,264 calls** remain separate rejected work, with zero training
consumption. The public rejection receipt links its artifact hashes. Two small
integration-test runs additionally used 7 and 10 branches; their ledgers are
included separately. No network access was granted to resolve the import issue.

## Training stability work

The completed [v2 results](evidence-decisions-v2-results.md) showed that an
over-limit optimizer update could still become a selected checkpoint. A new,
prospective wrapper makes checkpoint eligibility conditional on acceptance.
It snapshots trainable parameters and optimizer state, records each physical
attempt, and restores parameters, optimizer moments/counters and random state
when an update is rejected or interrupted. Rejected presentations remain counted.
There is no automatic retry or silent learning-rate change.

The guard checks mean policy divergence (0.02 limit) and maximum individual
divergence (0.10 limit) on declared policy inputs. It still needs integration
into the future 9B launcher; it does not change the completed v2 run. A local
tiny-network test with four live shell episodes and seven transitions accepted
a normal update (mean divergence about 0.000012) and rejected a deliberately
oversized update (about 0.789), with exact parameter and optimizer restoration.
Each attempt presented seven policy transitions, eight consequence questions and
four replay questions. These are mechanics tests, not model-quality results.

We also corrected future live-reward accounting for an already-satisfied goal:
pay command costs when they occur and verified terminal utility once. Subtracting
an initial success potential would incorrectly give zero reward for stopping.
The regression exercises that case. The original shell pilot did not contain
already-satisfied starting goals; its published results remain unchanged.

Four guard tests and the report/accounting/learning contracts pass. The live
mechanics test was repeated after the reward change; both 36-command runs are
retained separately in the bundle.

## Next gate before paid training

Combine this training-family candidate with qualified configuration and SQLite
families, keeping whole CSV-report and atomic-publication families for validation
and transfer. Across both qualifications there are 3,756 prepared questions,
but only five fixture roots. This is still a small curriculum. Current
contradictions are explicitly historical; conflicting current sources of
uncertain reliability remain an open data gap.

The remaining engineering is a live application worker connected to policy
rollouts, its learning-mechanics check, a frozen mixed-family data schedule,
and a controlled original-checkpoint 9B comparison with the new update guard.
The user authorizes qualified training within the remaining original budget
without another confirmation. No new machine is rented yet. The older ToolSandbox
reminder cohort stays diagnostic-only. Dynamic proposals and Harbor execution
remain a separate qualification stage.

Source and licensing details remain in the
[ToolSandbox pilot notices](toolsandbox-pilot-v1.md#separate-upstream-notices-and-license-scope).
The bundle contains our fixtures and receipts, not redistributed upstream tool
source or official scenario data.
