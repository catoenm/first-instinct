# Forecast the moment before a consequential action

The completed mixed pilot's publication forecast set had no uncertain groups and
no incorrect outcomes, despite real live policies sometimes publishing wrongly.
The new local diagnostic asks forecasts at intermediate execution histories.
**It now includes both**, without changing the old run or its labels. No model
was trained or evaluated, and this exposed publication mechanism is not a new
transfer test.

The existing actual shell executor stages bytes, validates their hash and
atomically publishes. We replay three histories before branching:

- Stage an artifact and validate it, before deciding whether to commit.
- Stage and validate an artifact, then stage a different artifact, invalidating
  the earlier validation.
- Attempt a premature commit that actually fails, before deciding how to recover.

For each visible history, execute every alternative in a fresh copy of each
compatible world, under immediate stopping and two explicit continuations. The
history's already-paid costs are excluded from future return. The total decision
horizon still applies. Ground truth comes from actual final files, protected
bytes and command costs. Identical observations with different outcomes retain
their per-world labels; no prediction supplies its own target.

The diagnostic passed all collection and reconstruction checks:

| Measurement | Count |
| --- | ---: |
| Existing mechanism roots / underlying file worlds | 1 / 2 |
| Underlying world-and-goal tasks | 4 |
| Cost/evidence cases / intermediate-history variants | 16 / 48 |
| Distinct executed alternatives | 1,008 |
| Additional independent verification replays | 1,008 |
| Actual commands, including history/menu replays | 9,652 |
| Prepared forecast / decision / observation-value questions | 672 / 30 / 30 |
| Forecast inputs with differing outcomes at equal observations | 32 |
| Questions in those uncertain groups | 64 |
| Forecast labels: completed / incorrect / unfinished | 184 / 16 / 472 |
| Optimizer updates / questions consumed in training | 0 / 0 |

There are 420 distinct forecast inputs and 30 public decision contexts. Twelve
contexts benefit from the specified inspection plan; expensive inspection does
not. Fresh current evidence resolves the hidden-world uncertainty. Asking at
the consequential state produces a more relevant learning target than simply
adding more initial-state rows.

Five integration tests checked conditional uncertainty before commit, exclusion
of sunk costs, actual failure from stale validation, rejection of invented
outcomes/histories, and preservation of diagnostic ownership. An offline audit
rebuilt all 732 questions from receipts, checked replay hashes and exact command
accounting, and verified the frozen source/fixture identities. The pinned 9B
tokenizer needs at most 1,063 tokens across questions and actor histories; no
truncation occurs.

This is a coverage improvement, **not proof that it improves a model**. Four
underlying tasks remain four tasks after 2,016 branch executions. The new
[filesystem family](filesystem-decisions-v1-results.md) contributes a different
mechanism; this diagnostic contributes more useful histories inside an existing
one. Across the two collections there are 1,848 distinct alternatives, 1,848
verification replays and 1,324 prepared questions. Counts must remain separate.

The [data, receipts and audits](../results/trajectory-decisions-v1/) include the
pre-execution freeze and all compatible-world labels. Preserve this root's
`exposed_diagnostic` ownership. The [next gates](decision-data-next-gates.md)
require broader training coverage and a newly reserved whole mechanism before
another bounded, identical-start 9B learning comparison.
