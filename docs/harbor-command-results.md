# Real command selection in Harbor

The proposer–selector–executor loop now runs real Linux commands through Harbor
0.23.0. A Codex subagent proposes commands using only the public task and observed
history. The existing supervised Qwen3.5-9B service selects one of those commands,
or finishes. A fresh, separate verifier container checks the resulting artifacts.

This qualifies an environment adapter. **It is not a reinforcement-learning
training result, a generalization benchmark, or evidence about Jev's internals.**
No weights were changed and no new paid compute or model API was used.

[Implementation and runnable example](../tool_lab/README.md) ·
[Protocol declared before execution](harbor-command-selection.md) ·
[Recorded evidence](../results/harbor-command-v1/)

## The actual loop

```text
Public task + actual command history
             ↓
Language model proposes three concrete commands
             ↓
9B selector sees exact commands and chooses one, or finishes
             ↓
Harbor executes the chosen command in the task container
             ↓
stdout, stderr and exit status become the next observation
             ↺
On completion: separate verifier checks files/database → reward
```

The selector uses the supervised checkpoint at step 2,742. Menu order is
shuffled. All live trials use greedy selection and have a six-decision limit.
The proposer is a Codex subagent in the user's existing account; the mailbox
must be serviced and is not an unattended model service by itself.

The configuration run inspected `services.json`, changed only the requested
worker count, read back the saved configuration, and finished. The database
run selected a parameterized SQLite update scoped to the requested region and
status, checked remaining pending rows and database integrity, and finished.
These commands executed in containers; their outputs were not fabricated text.

## Qualification results

Six generated fixtures cover three small authored mechanisms, with two seeds
per mechanism. All six reference solutions passed independent verification;
all six no-op controls failed, as expected. Neither control uses the selector.

| Live task | Verified result | Executed commands | Reward |
| --- | --- | ---: | ---: |
| Scoped configuration edit | Pass | 3 | 0.97 |
| Scoped database update | Pass | 2 | 0.98 |
| Filtered CSV report | Pass | 2 | 0.98 |

All **15 Harbor trials** completed without infrastructure errors: 12 controls
and three live trajectories, using ten selector predictions in total. There
were no retries. The report task created the requested JSON from filtered CSV
rows, recalculated the total from the source, checked the saved result and
finished. All attempts are retained; these are not three selected successes
from a larger search.

Reward is terminal verified success (1 or 0) minus 0.01 for each executed
command. Finish costs zero. For example, the configuration actions receive
returns 0.97, 0.98, 0.99 and 1.00. Successful command exit alone earns no success
reward. Infrastructure errors receive no training return.

The verifier checks protected file hashes, exact typed JSON values, or SQLite
integrity, schema and every row. It rejects unexpected files and symbolic links.
Agent commands run as an unprivileged user without network access. Only the
declared data directory transfers into the separate verifier environment.

Fourteen host-side unit checks cover task invariants, proposal boundaries and
reward attribution. The portable example job configuration also validates
against the installed Harbor schema without executing another trial.

## What these results do and do not establish

- Dynamic menus can select actual commands and continue from their observed
  results. Arbitrary caller-supplied commands are scored, not a fixed taxonomy.
- This is a small integration check. The strong proposer does substantial
  reasoning, and several alternatives can have equivalent effects. Success
  cannot yet be attributed to the selector over random or simple selection.
- The task generators currently vary entities and numbers within three simple
  mechanisms. They are not an infinite supply of independent problems. The
  [richer family designs](harbor-task-families.md) remain proposals.
- Choice probabilities are not probabilities of eventual task success. No
  calibrated outcome predictor was trained in this pilot.
- Greedy historical traces must not be reused as current on-policy rollouts.
  Proximal Policy Optimization needs new sampled trajectories from its current
  model, preserving exact encoded inputs, menu ordering and action likelihoods.

The first three configuration menus used public history relayed by the root
agent; one relay compacted the initial JSON's whitespace. Subsequent proposals
read exact public mailbox snapshots. No solution, verifier or expected output
was supplied to the proposer. This transport difference is retained as a
limitation, not treated as a controlled comparison of proposer performance.

### Post-run history logging correction

The final audit found that version 1.0.0 retained a mutable history list in each
receipt's `events[].request`. Later commands therefore changed that redundant
field in seven earlier events. The immutable mailbox requests, exact serialized
selector inputs, commands and rewards were unaffected. Version 1.0.1 snapshots
the history; a regression test checks both appending and editing observations.

Original receipts and all frozen source files are preserved unchanged under
the evidence directory. Explicitly named `choices-with-snapshotted-requests.json`
derivatives restore only each request from its original mailbox file. An audit
reconstructs all ten model inputs and menus from those files, validates every
selected action and return, and records the seven affected events. No trial was
rerun or excluded. The correction was unit-tested after collection, not presented
as the version used for the live runs. Run the offline evidence audit with:

```sh
python3 results/harbor-command-v1/verify.py
```

## The next data experiment

First make the choices consequential: layered configuration overrides,
multi-table updates, malformed inputs, command failures and dependent steps.
Check reference solutions and deliberately wrong repairs before collection.
Reserve whole mechanisms or combinations for evaluation before expanding seeds.

Measure random/simple selection alongside the current model under a fixed
proposer setup. To identify useful alternatives, fork the same observed state
and execute different proposed actions under a fixed continuation policy. Count
that branching as additional work and retain failures. This produces evidence
about the choices the selector could have made, not just the path it took.

Then train the command-selection policy on verified terminal reward. A separate
outcome head can forecast eventual success for a proposed action under a stated
continuation policy. Repeated controlled executions and a proper scoring loss
can train those forecasts; a single successful trace is not a calibrated
probability target. Evaluate both decision utility and forecast quality on the
reserved mechanisms. This would test our calibrated-outcome idea without
claiming to reproduce an undisclosed training recipe.
