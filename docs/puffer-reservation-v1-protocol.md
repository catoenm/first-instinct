# Transactional reservation: qualify the environment before scaling the learner

This is a new, bounded experiment following the Harbor command pilot. The target
language model remains the existing 9B checkpoint. This phase introduces one
transactional mechanism, not a new language model or an unlimited dataset.
Frozen previous experiments and their execution budgets are not reused.

## Task and information boundary

Create request 17 for account 7, allocating one unit each of A and B. Preserve
other accounts, requests, inventory and database schema. It is permitted to
create account 7 when absent and replenish A or B by one unit when that item's
quantity is zero. Actions have publicly specified research costs, not measured
database latency. The initial world is drawn from a disclosed finite prior:

| World | A | B | Account exists |
| --- | ---: | ---: | --- |
| ready | 1 | 1 | yes |
| A shortage | 0 | 2 | yes |
| B shortage | 2 | 0 | yes |
| missing account | 1 | 1 | no |
| A shortage + missing account | 0 | 2 | no |
| B shortage + missing account | 2 | 0 | no |

The model receives this prior, costs, remaining actions, query results and
committed-write receipts. It never receives the sampled world ID, unobserved
stock, or a verifier answer. Action availability never depends on hidden state.
Under a prior with zero probability on combined faults, a stock shortage implies
account presence; this is a declared prior implication, not extra observation.

## Nine actions

Inspect stock; inspect account; reserve atomically; reserve sequentially;
replenish A; replenish B; create account; undo this request; finish.

Both reservation programs insert the request (checking its account foreign
key), decrement A, insert A's allocation, decrement B, then insert B's allocation.
The atomic program explicitly rolls back the entire operation on error. The
sequential program commits each statement: a stock failure can leave a request
header or an allocation. Errors identify the failing stage and committed effects.
Undo returns quantities recorded in this request's allocations and deletes only
this request and its allocations, atomically. It does not restore an entire
database, erase unrelated work, or undo legitimate replenishment.

Finish and horizon exhaustion both independently verify the state. The mutually
exclusive outcomes are success, incomplete, partial request, and forbidden
change. Success earns 100 research credits; partial/forbidden outcomes earn
minus 100; incomplete earns zero. Every action also pays its public fee. The
implementation uses quarter-credit integers, scaling total reward by 400 for
the learning interface. Completion on the last allowed action counts.

## Qualification limits

Use four public profiles combining two inspection prices (0.25 or 2 credits),
two atomic prices (1 or 4), horizons 3 or 6, and either a four-world uniform prior
or a six-world uniform prior. Sequential reservation costs 1; replenishment and
account creation cost 3; undo costs 2; finish is free. The exact four profiles
are recorded in the implementation and frozen manifest before collection.

For every profile and all six worlds, evaluate three prefixes (empty, stock
inspection, sequential reservation). From each prefix, execute each of nine
first actions followed by a fixed public continuation. Replay each branch once:
**1,296 SQLite branches**, with at most **128 additional guard/debug attempts**,
and an absolute **1,424 database-attempt ceiling**. Every database construction
is logged before creation; failed attempts consume the budget. Count SQL
statements and simulated transitions separately. Stop at 30 minutes or 100 MB.
Some enumerated worlds have zero prior probability; retain them as implementation
controls, excluding them from that profile's probability targets.

Compare the C simulator and independent SQLite execution after every action:
actual quantities, request/allocation rows, supplied stock, public observations,
terminal classes and rewards. Replay must match exactly. Verify no-op, partial
write, rollback, scoped undo and protected-row corruption controls. Group
branches by identical public history, then weight supported worlds by the prior
and actual observation likelihood. Verify outcome/cost *joint* distributions;
do not confuse an action-selection probability with a success probability.

The continuation finishes a completed request; undoes a known partial request;
repairs an observed absent account or observed zero stock; otherwise inspects
unknown stock, then unknown account; then reserves atomically. It uses only
public observations. Forecasts mean first action followed by this specific
continuation within the remaining horizon, not optimal success probability.

## Runtime and later gates

The native adapter targets PufferLib 5.0 commit
`6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`. Headless compilation will exercise its
actual pinned environment header and lifecycle on the local CPU. This is not
a claim that the CUDA trainer has run. Third-party files remain outside the
model environment. No rented compute or model inference is needed to qualify
the execution labels.

After qualification, a separate, prospectively bounded local small-policy
learning check may use the same environment with a named training algorithm.
Record whether training uses native PuffeRL or a separate trainer. Reserve
combined faults from training and expose them only in a final diagnostic.
Any new training settings, transition limits, seeds and selection rule must be
written before that run; do not add search after seeing final results. Native
PufferLib GPU training and a 27B model comparison remain later stages, warranted
by useful environment and learning results rather than by row count alone.

References: [PufferLib current documentation](https://puffer.ai/docs.html) and
[pinned environment interface](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/pufferenv.h).
