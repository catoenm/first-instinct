# Calendar decisions: qualified data, reserved transfer

The calendar mechanism passed native execution, independent verification replay,
receipt reconstruction, full token-length checks and Linux parity. **No model
has seen these prompts, and no model improvement is claimed.** The entire
mechanism remains reserved for transfer after the next learning recipe and
checkpoint selection have been frozen.

Three task structures exercise actual database and timezone behavior: booking
beside or across an existing interval, choosing the intended occurrence of a
repeated local time, and rescheduling an event while preserving its original
record if insertion fails. They are three structures within one authored
mechanism, not three independently sourced benchmarks. Four concrete database
worlds per structure vary conflicts and calendars. Missing, current and stale
evidence, a real failed foreign-key command, and expensive inspection create
different decision situations.

Commands run in separate Python processes against fresh SQLite databases.
SQLite enforces overlap, identity and foreign-key constraints. The command's
timezone converter reads a pinned 2026c file; an independent verifier uses
separately specified UTC anchors. A new read-only connection checks the exact
event rows, schema, integrity, protected files and unrelated state. A successful
exit code alone is insufficient. Actual transactions restore the old event on
failure; separately committed deletion can lose it.

For example, rescheduling with no current observation has four equally likely
worlds. Under the declared evidence-based continuation, the executed average
future returns are:

| First action | Inspection cost 0.01 | Inspection cost 1.2 |
| --- | ---: | ---: |
| Inspect, then act on current evidence | 0.72875 | -0.46125 |
| Attempt atomic rescheduling at the requested occurrence | 0.66000 | 0.66000 |
| Delete and insert with separate commits | 0.48500 | 0.48500 |
| Stop | 0 | 0 |

Thus cheap inspection is worthwhile, while expensive inspection is not.
These are verified branch averages under a specified continuation and finite
prior, not learned scores or unrestricted optimal values. A blocked move can
be unfinished after atomic rollback or incorrect after separate deletion.
Identical visible input legitimately leads to different outcomes across worlds.

## Accounting

| Quantity | Count |
| --- | ---: |
| Authored mechanism / task structures | 1 / 3 |
| Concrete initial database worlds | 12 |
| World-and-goal tasks | 16 |
| Evidence/cost cases | 80 |
| Distinct executed alternatives | 2,160 |
| Independent verification replays | 2,160 |
| Physical branch executions and database initializations | 4,320 |
| Actual tool commands, including prefixes and replays | 7,956 |
| Initialization SQL statements, excluding pragmas and trigger internals | 39,960 |
| Forecast questions | 1,440 |
| Next-action / observation-value questions | 32 / 32 |
| Total prepared questions | 1,504 |
| Distinct forecast public inputs | 576 |
| Ambiguous forecast input groups / questions in them | 149 / 596 |
| New model evaluations / optimizer steps / training consumption | 0 / 0 / 0 |

Forecast labels include 430 completed, 170 incorrect and 840 unfinished
outcomes. Every task structure contains all three. Probabilities are conditional
on the declared worlds and visible history; they are not estimates of the
frequency of real calendar failures.

Nine integration tests cover conversion anchors, endpoint adjacency, calendar
scope, actual rollback versus partial writes, uncertainty, preservation failures,
recovery, stopping, goal-dependent choices, tampering and private-field exclusion.
The offline audit reconstructs all questions from receipts and matches every
primary/replay attempt. The pinned 9B tokenizer needs at most 1,397 tokens for
questions and 1,841 for recorded actor histories; no truncation occurs. The
combined question/history audit has 1,171 distinct structured inputs, which is
a different count from the 576 forecast inputs.

A separate network-disabled Linux run executed 144 selected branches and 270
commands. All receipts matched exactly under the declared contract: public
observations, logical schema and rows, integrity, and protected bytes. Native
and Linux SQLite file pages are not claimed byte-identical. These parity runs
are additional executions, not additional training questions or independent
tasks.

The [protocol](calendar-decisions-v1-protocol.md) was frozen before collection.
The [public artifact bundle](../results/calendar-decisions-v1/) contains the
data, execution and replay receipts, source hashes, audits and parity results.
Private verifier worlds are present for reproducibility; only each question's
`input` field is permitted in a model prompt. Ground-truth qualification is
complete, while model results remain unmeasured.

This is a fixed command catalog, not arbitrary shell proposals or a Harbor
benchmark. It does not test concurrent writers, recurring calendars, nonexistent
spring-transition times or external calendar services. The next step is a
broader training mixture using the already qualified filesystem and reservation
mechanisms, followed by a bounded equal-start learning comparison. This calendar
family must remain outside that training mixture and checkpoint selection.
