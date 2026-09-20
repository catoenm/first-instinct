# Calendar decisions: reserved mechanism, local qualification only

Before collection or model evaluation, reserve this entire calendar mechanism
as `reserved_transfer`. No calendar row, history, alternative or wording variant
may enter training, replay, development model prompts or checkpoint selection.
Ground truth may be inspected to qualify the environment; model scores stay
unopened until a new learning protocol is frozen. This is an authored diagnostic,
not a claim that the foundation never encountered calendars in pretraining.

Implement three task structures using actual SQLite: booking beside or across
an existing interval, choosing the requested occurrence of a repeated local time,
and atomically rescheduling an existing event. Related first/second-occurrence
goals share the fold task root. Use four concrete worlds per structure, 16 total
world-and-goal tasks, five evidence/cost regimes and at most 80 cases. Nine offered
actions and three continuations give at most 2,160 distinct branches, each
independently replayed once (4,320 physical branch executions), with a 25,000-tool-
command cap. Initialization and prefix commands are counted separately. No GPU,
model or external service is used; stop and preserve rejection records on failure.

The timezone asset is the exact 3,552-byte America/New_York TZif file from the
installed 2026c database, with a recorded SHA-256 and provenance. Load that file
explicitly, without a system database fallback. First/second occurrence uses
Python's `fold` semantics; duration means elapsed minutes, so the end is computed
in UTC after resolving the start. Independent verifier anchors are separately
authored UTC timestamps, not outputs of the conversion function used by commands.
See the primary [ZoneInfo documentation](https://docs.python.org/3/library/zoneinfo.html).

Events use half-open intervals: touching endpoints do not conflict. A database
trigger rejects genuine overlap within the same calendar, a foreign key rejects
unknown calendars, and event identities are unique. Atomic rescheduling explicitly
begins a transaction, deletes the old row, inserts the new row and commits; any
constraint error rolls it back. The sequential alternative autocommits deletion
before attempting insertion, so failure can lose the original event. Consult
[SQLite transactions](https://www.sqlite.org/lang_transaction.html) and
[trigger behavior](https://www.sqlite.org/lang_createtrigger.html).

The authored command catalog offers current inspection, booking either occurrence,
atomic or sequential rescheduling at either occurrence, a wrong-calendar booking,
and stopping. Program bodies and user requests are fixed data, never model-generated
shell code. Each branch starts in a new temporary database. Commands execute in
separate Python processes; a separate read-only connection verifies actual schema,
rows, integrity, unrelated data and protected files. Logical database state is the
portability contract; raw SQLite file bytes are not claimed identical across
engine versions. No verifier target is passed to the command worker.

Regimes include hidden evidence, current inspection, a stale cached report,
a real failed foreign-key operation, and expensive inspection. Cache generation
is explicitly historical and independent of current truth. Current inspection
also performs the actual timezone conversion and returns current calendar rows.
Inspected histories are states before a committing command; this does not expose
an open transaction across separate commands. The fold worlds have four equally
likely combinations of blocked first/second occurrences. Other structures have
free, adjacent, conflicting, or other-calendar occupancy. Keep all compatible
worlds and contrary labels at identical visible histories.

Reward is verified completion +1, a wrong committed change or lost original -1,
otherwise 0, minus future costs. A failed atomic operation with unchanged state
is recoverable; an incorrect committed change ends the episode. Prefix costs are
sunk. Forecasts concern either the named action followed by immediate stopping,
or the declared four-decision continuation. That continuation uses only visible
history: inspect while current rows are unknown unless inspection costs at least
one; with current rows, stop if the requested slot conflicts, otherwise book or
use the cheaper correct rescheduling operation. The no-observation continuation
uses the correct booking/atomic command without more inspection when state is
unknown. Both stop after an attempted command at the requested occurrence on the
requested calendar reports a conflict. These are named policies, not unrestricted
optimality.

Pair forecasts, next-action choices and observation-value questions from each
visible history. Derive labels from executed receipts across compatible worlds;
no prediction or language-model judgment can supply truth. Require all three
outcomes, uncertain groups, evidence resolving uncertainty, useful and unnecessary
inspection, cost-sensitive choices, actual recoverable failures and wrong
irreversible outcomes. Negative controls must reject no-op, wrong occurrence,
wrong calendar, wrong duration, collateral changes and partial deletion. Verify
adjacency acceptance and atomic rollback independently. Reconstruct histories,
labels, costs, menu coverage and physical counts after serialization; replay every
branch, check token lengths and perform bounded pinned-Linux parity before any
training comparison is considered. This family remains entirely reserved.
