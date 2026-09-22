# Local qualification of competing database writes

This is a bounded executor qualification, not a model experiment or training
corpus. It builds on the [mechanism overlap review](mechanism-overlap-v1.md).
Keep one ownership group, provisionally a training candidate. Do not call its
goals, costs or schedules independent transfer families.

Use one initial SQLite database, two real writer connections, and a third
read-only connection for state verification. After the same initial read, the
colleague either does nothing or updates the counter, note and revision in one
transaction. Both schedules have probability one half. No concurrent event
occurs later; this first prototype tests deterministic interleaving, not
nondeterministic contention, locking, crashes or repeated writers.

Two goals use the same commands. One requests an increment relative to the
latest counter while preserving the colleague's note. The other authorizes
the original approved edit only at revision zero; after a revision change,
leaving the colleague's complete row intact satisfies that goal. Commands can
read, overwrite cached fields, perform a conditional write, increment the
current field atomically, attempt a missing-row read or stop.

A successful write terminates the episode. Reads, read failures and conditional
refusals allow recovery. Four decisions is the maximum. All attempts incur their
published costs. Completion earns 100 abstract units, damage -100, and a valid
unfinished state zero, minus costs. This is not money. The verifier compares
the final database to the goal and protected state independently of return codes.
A successful command may damage the goal; a refused command may leave an
already satisfied goal intact.

Before collection, run tests of hidden-state indistinguishability, conditional
refusal/recovery, goal-dependent effects, failed-command costs and corrupted
receipt rejection. Freeze the exact sources and witness plans before executing
the collection. Execute two goals × two schedules × three fee profiles × eight
witness plans = 96 primary branches, then one independent replay of each. Stop
on mismatch or infrastructure error. Replays add no probability mass. The
maximum is 192 database resets and 768 actor command attempts, plus one prefix
read per reset and up to one colleague write per reset. Tests are separate.

Reconstruct every event, observation, cache update, cost and terminal boundary
with separate relational audit logic. Require identical visible initial inputs
across hidden schedules and identical replay traces. Check that observation value
changes sign with its cost, that expensive writes favor stopping, and that the
same command has different outcomes under the two goals. Compute witness-plan
returns from executions; do not label them globally optimal policies.

Preserve aggregate uncertainty for immediate command-then-stop forecasts. Keep
goal-outcome probabilities separate from command return-code probabilities.
No question renderer, token-length qualification, live model adapter, learned
policy continuation, training admission or optimizer exposure is implied by this
executor stage. Those require a subsequent explicit data-interface stage.
