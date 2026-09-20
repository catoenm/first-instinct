# Five mechanisms are ready for a bounded learning comparison

The live environment adapters, probability targets, guarded learning updates and
exact cloud package passed local qualification. A new six-arm comparison has
been provisioned on one H200. **These qualification results do not establish
improved model performance.** The [frozen protocol](expanded-decisions-v1-protocol.md)
extends evidence-decisions-v2 and the completed mixed-decisions-v1 study from the
same original supervised step2742 checkpoint.

The important addition is live replanning in real filesystem and SQLite worlds.
Reservations permit partial-write repair before the episode ends; filesystem
and calendar mutations use their separately declared terminal rules. Successful
command execution is not a success label: independent predicates check the goal,
protected state, costs and the actual outcome. Sunk prefix costs are disclosed
but are not charged twice. A successful terminal outcome is rewarded once.

For example, a reservation episode starts after a sequential command has written
part of an order and then failed. A valid recovery undoes the partial order,
replenishes the missing item, executes an atomic reservation and finishes. The
integration test verifies the final database, reward 0.91 and 0.09 of new costs;
the failed prefix's already-paid 0.01 is excluded. Fresh live decisions do not
promise the fixed continuation used by a consequence-forecast question.

## Runtime evidence

| Quantity | Count |
| --- | ---: |
| Authored-reference or seeded-random primary episodes | 384 |
| Unique executed trajectories among those episodes | 370 |
| Independent native verification replays | 384 |
| Native physical episode executions, including replay | 768 |
| Additional pinned-Linux parity episodes | 96 |
| Unique public actor inputs | 470 |
| Maximum complete actor input length | 2,255 tokens |
| New training questions or pretrained-model calls in this stage | 0 |

Every replay and Linux receipt matched. The native run executed 998 filesystem/
calendar subprocess commands and 1,188 reservation actions including finish;
Linux added 124 commands and 124 reservation actions. Native offered actions,
including stop and prefix actions, totalled 2,338; Linux totalled 265. SQLite
trace callbacks counted 46,356 native and 5,127 Linux statements, including setup,
tools, observation reads and verifier queries. These categories overlap and
must not be summed into a fictitious number of independent examples.

Eight integration tests cover goal-dependent mutations, failed-command recovery,
sunk fees, already-completed goals, redundant inspections, private-state isolation,
tampered receipts and action/attempt limits. This is runtime verification with
authored and random controllers, not a trained-policy benchmark. The 96 Linux
checks include all three new adapters: filesystem, reservations and calendar.

## Learning mechanics

A separate random tiny CPU network ran one accepted update for each method:
forecast supervision, reward-only Proximal Policy Optimization and their
combination. General-task replay was included. Ten real episodes supplied the
reward-bearing arms. All three started with identical language parameters;
language weights changed, and the detached value component supplied no language
gradient. This test made **zero pretrained-model calls or 9B updates**.

The three tiny optimizer steps consumed 58 objective presentations: 20 forecast,
26 policy and 12 general-replay presentations. Another 84 component-gradient
presentations were diagnostic backward passes without optimizer steps. They are
reported separately. Four learning tests check proper uncertain targets, native
forecast probabilities, native-policy divergence bounds and exact rejection
rollback. Four curriculum/metric tests check balanced sampling, target-independent
schedules, mechanism ownership and structure-balanced transfer scoring.

The guard now checks both native action probabilities and the distribution used
for exploration. The preceding mixed study checked the latter only. Each
rejected update restores model parameters, optimizer state and random state,
retains its consumption records and stops the arm. Probability scoring reports
expected Brier score separately from excess error above irreducible uncertainty.

## Frozen pilot, not a claimed result

The training pool has five mechanisms and 268 concrete context/cost/history
cases: 36 configuration, 36 database repair, 84 ToolSandbox application, 40
filesystem-scope and 72 reservation cases. These cases are variants of a small
number of authored task structures; they are not 268 independent task families.
Filesystem adds two task structures. Calendar has three structures but remains
one complete reserved transfer mechanism. The [source registry](decision-source-registry-v1-results.md)
records underlying roots and physical worlds separately from questions.

| Training mechanism | Authored task-root groups | Concrete world-and-goal tasks | Context/cost/history cases |
| --- | ---: | ---: | ---: |
| Configuration | 1 | 4 | 36 |
| Database repair | 1 | 4 | 36 |
| Application delivery | 1 | 18 | 84 |
| Filesystem mutation scope | 2 | 8 | 40 |
| Reservations | 1 | 6 | 72 |
| Total | 6 | 40 | 268 |

The root groups use the preserved source ownership identifiers; the two
filesystem goals share an executor and are still one mechanism. World-and-goal
tasks exclude changes only to fees, observation history, horizon or wording.

The prepared training forecasts collapse 4,232 source rows to **2,280 distinct
inputs**, including 380 uncertain inputs. Report development has 36 cases and
308 forecast inputs; calendar transfer has 80 cases and 576 inputs, including
149 uncertain inputs. General replay, retention and transfer retain their
unchanged 4,096/622/1,128 questions. Preparation itself consumed none of these
questions in 9B training. Actual backward passes and executed trajectories will
be counted from each arm's ledgers, including rejected attempts and repeats.

Two seeds compare the three methods, starting from the identical original 9B
adapter. The pilot preserves general replay, reduces the forecast objective
weight prospectively using previous training-gradient evidence, and stops arms
that fail development checks. Calendar results stay unopened to the operator
until every arm's selection and stop decisions are fixed or the pilot is
irreversibly stopped. Advancement requires better decisions **and** better
probability estimates in both seeds without material general-task regression.

The extracted 252-file cloud bundle passed all 27 startup tests before rental.
The actual training host must pass shell, application and expanded-runtime
parity before pretrained inference. One H200 was quoted at $4.59/hour, with a
six-hour deadline and at most $40 allocated from the original cumulative $500
authorization. Prior provider-reported charges were $175.36. No new budget,
larger model, promotion of the demo or claim about Jev's private implementation
follows from qualification.

The [receipt archive](../results/expanded-decisions-v1-qualification/) contains
56 files covering frozen plans, native/replay/Linux receipts, tiny-network
mechanics, schedules and package checks. It excludes foundation weights and
contains no learned calendar scores. The archive is 672,994 bytes with SHA256
`b038fd833885827a5c4375ed11e4c99573df7f46d4bd50f1db8b199e3e79344c`.
