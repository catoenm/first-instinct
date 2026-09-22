# A qualified competing-writer example

The local SQLite prototype passes execution, independent relational checks and
replay. It adds version-checked conflict handling to the previously exposed
transaction and stale-evidence examples. It has **not trained or evaluated a
model**, and is not an unfamiliar-mechanism test.

Two real connections act on the same database. An initial read is followed by
an optional colleague update. A third, read-only connection inspects actual
state. The visible initial history is identical with and without that update;
each schedule has probability one half. The checker verifies the requested
counter change, revision, colleague note, protected row and schema.

The same commands support two different goals: increment the latest counter,
or apply an edit approved only for the original revision. Under the latter
goal, preserving a colleague's intervening update is success.

| Evidence | Qualified count |
| --- | ---: |
| Authored fixture / mechanism | 1 / 1 |
| Initial databases / post-schedule states | 1 / 2 |
| Goals / world-and-goal tasks | 2 / 4 |
| Fee profiles | 3 |
| Primary executed branches | 96 |
| Independent replay branches | 96 |
| Actor actions, including 36 stops | 348 |
| Actor database statements, excluding transaction control | 312 |
| Prefix reads / colleague writes | 192 / 96 |
| Failed or refused actor commands | 48 |
| Distinct visible actor contexts | 60 |
| Training questions / optimizer presentations | 0 / 0 |

Replays reproduce every trace exactly and add no probability mass. Five tests
cover hidden-state indistinguishability, refusal and recovery, goal-dependent
effects, failure costs, and corrupted receipts. Collection plus read-back audit
took about 1.24 seconds under the local memory guard, with sampled peak process
memory below 27 MiB and no additional swap. Unit-test executions are separate.

Three useful distinctions emerge from the executions:

- An unconditional cached write returns success in both worlds, but satisfies
  the increment goal in only half; otherwise it overwrites the colleague's work.
- A conditional write followed by stopping returns success in half the worlds,
  yet satisfies the revision-limited goal in both. Refusal correctly leaves the
  colleague's state intact. Return-code probabilities and goal probabilities
  therefore differ.
- For the increment goal, reading before a conditional write beats the best of
  four declared one-command plans by 35 reward units with cheap reads, but loses
  by 83 units with expensive reads. Expensive writes favor stopping. These are
  comparisons among explicit witness plans, not a globally optimal value of
  information or policy label.

The [protocol](revisioned-sqlite-v1-protocol.md) and
[aggregate receipts](../results/revisioned-sqlite-v1/summary.json) preserve the
scope. The next step is a public question interface with explicit continuation
semantics, compatible-world conditioning and token-length checks. This prototype
does not yet cover ongoing contention, repeated competing writers, or crashes.
Keep its related goals and schedules in one ownership group.
