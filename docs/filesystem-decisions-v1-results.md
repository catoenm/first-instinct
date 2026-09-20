# File identity changes the correct decision

The local filesystem family passed native execution, independent replay,
receipt reconstruction and a 72-branch comparison against the pinned Linux
runtime. **No model was trained or evaluated.** This implements the filesystem
mutation-scope direction in the earlier data design and adds one mechanism,
not hundreds of independent tasks.

The two user requests require different behavior:

- Change just `a.txt`, preserving the other paths' contents.
- Change the physical file named `a.txt`, including its existing hard-link aliases,
  while preserving all sharing relationships.

Four actual temporary filesystem worlds have the same initial contents but
different sharing: no aliases, `a` shared with `b`, `a` shared with `c`, or `b`
shared with `c`. Writing in place can change another path. Atomic replacement
can prevent that side effect, or incorrectly break the sharing the user wanted.
The same command can therefore succeed or fail under identical observations.
The parent verifier checks actual bytes, permissions, file sets and relationships;
a successful process exit is insufficient.

A concrete executed example uses the first goal and an equal prior over those
four worlds. An in-place write costs 0.02, replacement costs 0.12, and inspection
costs 0.015. Under the declared continuation:

| First action | Mean verified future return |
| --- | ---: |
| Inspect, then choose using observed sharing | 0.915 |
| Replace immediately | 0.880 |
| Write in place immediately | -0.020 |

When inspection costs 1.2, replacing immediately is best. When all mutations
cost at least 1.2, stopping is best. These are averages of executed branches,
not model predictions or learned performance. Current evidence removes the
uncertainty; identical contents alone do not. A menu-only or constant-content
lookup can achieve at most 75% on the eight fresh decision contexts in this
small slice. This is a diagnostic ceiling, not a learned-model baseline.

## Counts and verification

Two task structures, four concrete initial worlds, eight world-and-goal tasks
and 40 cost/evidence cases produced **840 distinct alternatives**. Every
alternative was separately replayed: 1,680 branch executions and 2,048 tool
commands, plus 15,960 initialization file operations. They produced 560 forecast
questions, 16 next-action questions and 16 observation-value questions: **592
prepared questions, zero consumed in training**.

There are 24 ambiguous forecast input groups covering 96 questions. Forecast
labels comprise 186 completed, 190 incorrect and 184 unfinished outcomes.
These are genuine distinct execution outcomes under equal visible input, not
confidence labels invented by a model. Decision targets average branch values
across compatible worlds, and all acceptable tied choices are retained.

Six integration tests cover real alias side effects, goal reversals, byte-correct
but relationship-wrong mutations, protected data and permissions, replay,
invented observations/labels, and cost-sensitive behavior. The offline audit
reconstructs every question and label, checks replay hashes and conservation of
command counts, and excludes private labels from model inputs. The pinned 9B
tokenizer requires at most 992 tokens across questions and actor histories;
no truncation occurs.

A separate network-disabled Docker run executed 72 representative branches and
112 commands. Every public observation and byte/link receipt matched native
execution exactly. Raw inode numbers are never serialized; link relationships
come from actual device/inode equality. This is a portability check, not a cloud
training run or a Harbor benchmark.

The [data, execution receipts and audits](../results/filesystem-decisions-v1/)
retain the pre-execution source and fixture hashes. The original native
qualification's `training_ready: false` is deliberate: this corpus still needs
a broader learning mixture, a new reserved transfer mechanism and a frozen pilot
recipe. Its entire mechanism is `train_candidate`; variants must not be split
row by row into later evaluation.

The first slice has no symlinks, concurrent writers, crashes or new application
tools. The separate intermediate-publication diagnostic covers failed commands
and recovery. Installed ToolSandbox remains the source of actual application
workflows. The [next local gates](decision-data-next-gates.md) reserve calendar
semantics for transfer before further paid training.
