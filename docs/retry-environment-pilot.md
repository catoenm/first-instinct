# Retry environment: data-quality pilot

This is a CPU-only prototype for inspecting executable decision data before
any new expensive training. **No model was trained or evaluated.** It is
separate from the frozen general-model experiment and is not a frozen benchmark.

The world starts after a job submission returned an ambiguous timeout. That
request may have committed, been lost, or still be in flight. A retry can
change how many jobs actually complete. Reusing an idempotency key prevents
duplicate side effects; retrying without one can process the job twice.
Waiting changes which in-flight requests have arrived, and a receipt lookup
can reveal a timestamped, possibly stale view of completed work.

## Run and inspect

Only Python's standard library is required, including its SQLite module:

```sh
.venv/bin/python -B -m unittest test_retry_environment -v
.venv/bin/python -B -m general_lab.retry_environment \
  --output output/retry-environment-pilot-reviewed-v1 \
  --worlds 2000 --audit-worlds 200 --seed 41
```

The generator refuses an existing destination. To repeat locally after the
recorded run, choose a fresh output directory. The repository ignores `output/`;
these generated files do not modify existing training or evaluation data.
There are no cloud, GPU, model, external-data or paid-service calls.

The [release](https://github.com/catoenm/first-instinct/releases/tag/general-decisions-v1)
also provides `first-instinct-retry-pilot-v1.tar.gz` and its checksum manifest.
Extract it into `output/` to inspect the exact reviewed data without regenerating
it. The [archive receipt](../results/retry-environment-pilot-v1/archive.json)
records every included file's hash. The synthetic pilot is separate from the
model archive and was not part of the completed model's training.

| File | Purpose |
| --- | --- |
| `exploration.jsonl` | Complete random trajectories, visible inputs, legal-action masks, chosen-action probabilities, costs, rewards and separate truth receipts |
| `forecast_examples.jsonl` | Binary examples with `input` and executed `target`; no exact probability labels |
| `forecast_replay.jsonl` | Verifier records linking each forecast to its configuration, visible history, independently drawn hidden tape and executed database outcome |
| `counterfactual_audit.jsonl` | Separate evaluation-only worlds, binary outcomes and exact enumerated probabilities outside `input` |
| `summary.json` | Counts, seed, SQLite version, generator source hash and hashes of the four JSONL files |

Only a row's `input` belongs in a model prompt. The surrounding target,
provenance, replay tape, database rows and verifier fields are never model
inputs. The serializer takes public configuration and public observations;
it does not accept an episode object or hidden tape.

## Executable mechanism

Each world uses its own in-memory SQLite database. A committed row represents
one completed job. Request insertion and its side effect occur in one SQLite
transaction. A non-null `UNIQUE` key with conflict handling deduplicates a
retry; SQLite permits multiple null keys, allowing an unkeyed retry to create
a second completion. This tests actual uniqueness behavior, not a label that
simply declares a retry safe.

The initial timeout is observed at tick zero. The operator has three active
controls—retry, one receipt lookup, and wait—plus an explicit abstain/stop
action. At most one retry and one lookup are permitted; repeated attempts are
rejected. Retry, lookup and wait each consume one tick. The deadline is tick
two or three. Including the initial submission, a recorded trajectory contains
two to four actions. Stop ends operator decisions but lets in-flight work
continue to the deadline.

The initial request is lost, already committed at tick zero, or delivered at
tick two. A retry is lost or arrives one or two ticks after it is issued.
An on-time retry acknowledgement may itself be lost. Deliveries at the
deadline count; later ones do not. Receipt snapshots report completions as of
their stated timestamp, with zero or one tick of lag. An empty stale receipt
does not cancel pending work or prove that no completion exists now.

Public receipt tokens identify logical requests. Keyed initial submissions and
retries share one token; an unkeyed retry has the same token whether the initial
request committed or not. Internal SQLite row numbers remain verifier-only.
An earlier draft hashed row numbers into tokens, which still let the exact
verifier decode completion counts. The reviewed implementation removes that
shortcut and tests that identical retry acknowledgements retain a 50% success
probability in an appropriately ambiguous scenario.

Rewards use executed truth at the deadline: +100 cents for exactly one
completion, −100 for none, and −200 for two, minus charged action costs.
Exploration selects uniformly among currently valid actions and records the
chosen probability `1 / number_of_legal_actions`. These choices are exploration
data, not supervised labels for optimal actions. No oracle action-value scores
enter prompts.

## What a forecast means

Every forecast names a currently valid offered action and a fixed deadline:

> Take this action now, then make no further requests or lookups. Will exactly
> one job have completed when the stated deadline arrives?

This continuation rule matters. A retry issued one tick before a deadline may
arrive too late, while the same retry with another tick available can succeed.
The target counts **all** job completions by the deadline, including the original
request and any duplicate caused by the retry. It does not mean eventual
success, receipt availability, or success under an unspecified later policy.

Each offered action defines its own yes/no event. Its success probability is
scored separately; probabilities across actions are not normalized. Several
actions can each leave exactly one job complete. Under this passive continuation,
lookup, wait and stop often share the same outcome probability. Lookup's value
for a later adaptive retry belongs to a different continuation-policy target.

For each explored world, one visited nonterminal state is selected. The
generator finds hidden tapes compatible with that state's visible history,
then draws a **new independent conditional tape for each offered action**.
It replays the history and action in SQLite and uses the actual binary outcome
as the label. These are sampled outcomes, not thresholded oracle probabilities;
different draws at the same visible state can legitimately disagree. Full
replay receipts preserve each draw.

The audit uses separate configuration, prefix, hidden-outcome and sampling
streams, with no exploration-world overlap. It reserves all its configurations
to test ownership. Exact probabilities are computed by finite enumeration of
at most 18 hidden tapes, conditioned only on visible history, using rational
weights. Audit outcomes are independent executed draws as well. Exact values
and fractions stay in verifier fields outside `input`.

## Grouping and measured prototype checks

A content hash of the public world configuration owns its split: approximately
80% training, 10% validation and 10% test. Every hidden draw, trajectory and
counterfactual for the same configuration stays in that group. Ownership does
not depend on outcome, generation order or requested split. These are random
**configuration holdouts**, not held-out mechanics: all splits use the same
retry, timing and SQLite rules.
Configurations that differ only in costs can share transition dynamics across
splits. A future generalization study should group those related cases and
reserve complete mechanism combinations before collecting model results.

The reviewed 2,000-world run above recorded:

| Measurement | Count |
| --- | ---: |
| Exploration worlds: training / validation / test | 1,630 / 190 / 180 |
| Final jobs: zero / exactly one / two | 491 / 1,325 / 184 |
| Trajectories with two / three / four recorded actions | 492 / 977 / 531 |
| Forecast examples: yes / no | 4,908 / 2,439 |
| Independent audit worlds / action rows | 200 / 748 |
| Audit binary outcomes: yes / no | 525 / 223 |

All 184 duplicates came from unkeyed worlds. Keyed worlds still included 244
missed jobs; idempotency prevents duplicates but does not guarantee delivery.
In 194 audit worlds, the per-action probabilities summed above one, as expected
for separate counterfactual events. These are simulator and sampling checks,
not model performance or calibration results.

The final generator hash recorded by this run is
`b8c18307479c88ee99d730948ff2daa71337f15fc50758eaceeff2e3cb5f5de6`.
The earlier `output/retry-environment-pilot-v1` is a superseded, ignored draft
from before the logical-token correction. Use the **reviewed** directory and
its summary hashes for inspection.

All 16 tests pass. They cover lost-acknowledgement duplicates, keyed
deduplication, delayed commits, stale snapshots, forbidden repeated actions,
terminal stepping, cost accounting, deadline-specific forecasts, conditional
probabilities, replay and seed determinism, split separation, independent
outcomes, and hidden-state/token leakage.

The [generation receipt](../results/retry-environment-pilot-v1/summary.json) and
[full replay check](../results/retry-environment-pilot-v1/verification.json) are
tracked in the repository. The latter re-executed all 2,000 trajectories,
7,347 forecast outcomes, and 748 audit outcomes, and recomputed every exact
audit probability. It checks reproduction and integrity; the tests separately
check semantic invariants. After generating the reviewed directory above,
reproduce that check from the repository root:

```sh
PYTHONPATH=. .venv/bin/python results/retry-environment-pilot-v1/verify.py
```

## Simplifications

This is one job, at most two submissions, a short discrete clock, and a finite
transport tape. Jobs complete atomically when requests arrive. There is no
partial business transaction, real network, concurrency, key expiration,
malicious caller, worker crash, cancellation or unrelated traffic. Receipts
perfectly report the stored count at their stated timestamp. Distribution
parameters are supplied exactly in public configuration; their estimation is
not tested. The SQLite transaction treats the completion row as the side effect,
so it does not claim that a database key makes arbitrary external side effects
atomic. These assumptions make the prototype inspectable; real service behavior
would require broader mechanisms and separate validation before training.
