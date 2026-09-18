# Small executable CLI task families

These are task-authoring proposals for the Harbor/Docker prototype, not executed
tasks. The proposer receives the public goal, workspace observations and prior
command results. The decision model selects among the proposer's current
commands; Harbor executes one and returns stdout, stderr and exit status. New
observations lead to new proposals. The examples below are for dataset authors,
not a solution sheet supplied to the proposer.

Keep verifier programs, reference outputs, fixture-generation metadata and
protected-state snapshots outside the agent-accessible container. Every success
condition must follow from the public goal; no concealed target string or secret
file hash defines the requested answer. A zero command exit code is insufficient.

## 1. Repair layered application configuration

**Public goal.** “The effective service configuration must bind to
`127.0.0.1:8081`, use two workers and disable debug mode. The application loads
`service.ini`, then `overrides/local.ini`, with later values overriding earlier
ones. Change only `overrides/local.ini`; preserve all effective settings not
named in this request. Do not modify the application or other files.”

The workspace includes both INI files and an immutable
`bin/show_config.py --json` that exposes the application's real parsing and
precedence behavior. Initial settings may be syntactically valid but shadowed by
the overlay. The task is semantic configuration repair, not reproduction of a
particular formatting style.

**Realistic proposals at different steps:**

- Inspect effective settings: `python bin/show_config.py --json`.
- Inspect inputs: `cat service.ini overrides/local.ini`.
- Patch the overlay using Python's `configparser`, preserving unrelated values.
- Apply a textual edit such as
  `sed -i 's/debug = true/debug = false/' overrides/local.ini`.
- Edit the base file instead, or run validation again without changing anything.

The last alternatives can be ineffective or violate the stated boundary. Do not
hardcode their labels: a text edit may be correct for one fixture and incomplete
for another. Candidates should arise from visible observations, and complete
shell arguments must be shown to the selector.

**Executable checks.** Parse using the unchanged application loader; assert exact
typed effective host, port, worker count and boolean debug values. Compare every
other effective key with the initial snapshot. Require both files to remain
parseable; compare all protected file bytes, modes and paths against the initial
workspace. Only the overlay may change, with formatting/comments unrestricted
unless explicitly included in a variant's public goal.

**Meaningful variation and groups.** Vary overlay precedence, missing versus
overridden keys, duplicate-key parse failures, and typed boolean/numeric values.
Reserve whole repair structures or combinations for evaluation. Related key
names, whitespace, values and domain wording remain in the same group; renaming
the service does not create a new mechanism.

## 2. Correct selected SQLite invoice totals

**Public goal.** “In `billing.sqlite`, repair invoices whose status is `draft` and
whose issue date is before `2026-01-01`. Set `subtotal_cents` to the sum of
`quantity * unit_cents` over their invoice lines, treating no lines as zero.
Set `total_cents = subtotal_cents + shipping_cents - discount_cents`. Preserve
invoice IDs and every other value. Leave sent/paid invoices and all other tables
unchanged. Preserve the database schema, indexes and triggers.”

Provide ordinary SQLite tables `invoices`, `invoice_lines` and `customers`, plus
public schema documentation. All money is signed integer cents, not floating
currency. Include eligible and ineligible rows, an empty invoice and multiple
lines per invoice. The predicate and empty-group rule are public, not hidden
exceptions discovered only by failing a verifier.

**Realistic proposals:**

- `sqlite3 billing.sqlite '.schema'` followed by a limited `SELECT` showing
  relevant invoices and their line totals.
- A transaction containing correlated aggregate updates with the exact
  status/date predicate and `COALESCE(SUM(...),0)`.
- Two updates in one transaction: subtotal first, then total, followed by a
  query and commit.
- An unfiltered update, an aggregate that leaves empty invoices `NULL`, or a
  one-statement update that incorrectly assumes a sibling assignment supplies
  the new subtotal when computing total.
- A read-only discrepancy query, useful for inspection but insufficient to
  complete the repair by itself.

**Executable checks.** Independently recompute expected values from the initial
line rows using integer arithmetic. Compare selected invoice fields exactly;
compare unselected invoices, all remaining columns, other tables and schema
objects with canonical initial snapshots. Run SQLite integrity and foreign-key
checks. Verify no unexpected files outside the declared database output set.
Database byte equality is inappropriate after legitimate writes: use logical
table/schema equality. If WAL mode is supported, declare its sidecars explicitly
and inspect a consistent database snapshot.

**Meaningful variation and groups.** Group by predicate/aggregation/update
structure: filtered versus unconditional targets, empty aggregate handling,
joins with multiple lines, and dependent computed columns. Hold out combinations
before generating rows. All amounts, IDs, row-order variants and repeated
versions of a database stay with their structural group. This tests database
correction and transactional scope, not another ambiguous-acknowledgement retry.

## 3. Produce a CSV report without damaging inputs

**Public goal.** “Read `sales.csv` and write `reports/by_region.csv` with columns
`region,orders,net_cents`. Include only rows with `status=settled` and dates from
`2026-01-01` through `2026-01-31`, inclusive. Count included records per region
and sum `quantity * unit_cents - refund_cents`. Emit one row per included region,
sorted by region's case-sensitive Unicode order. Preserve source files and
scripts. Use normal CSV quoting; do not omit zero or negative net totals.”

Public input columns include an otherwise irrelevant description field with
quoted commas and newlines. IDs are unique in this initial family. If later
variants introduce duplicate IDs or multiple input shards, their deduplication
and conflict rules must appear in the public goal rather than in private tests.

**Realistic proposals:**

- Inspect the header and a few records with Python's `csv.DictReader`.
- Inspect with `head`, which is useful but can split a quoted multiline record.
- Use a Python `csv`/integer-arithmetic aggregation that applies the public
  filter, sorts regions and writes the requested header.
- Use `awk -F,` or `cut -d,`, which can misparse quoted delimiters; aggregate all
  statuses; subtract one refund per region instead of per record; or use
  `sort | uniq`, which may silently discard legitimate rows in later variants.
- Write a report to stdout without creating the requested output file.

**Executable checks.** A separate reference implementation parses the initial
CSV and calculates the public transformation. Parse the produced CSV; require
the exact header, ordered region set, integer counts and totals, with no extra
rows or columns. Accept equivalent valid CSV quoting and line endings. Hash all
protected inputs/scripts and reject unexpected output paths. Never compare only
the report filename, command status or a substring of stdout.

**Meaningful variation and groups.** Vary CSV lexical structure, filtering,
aggregation and explicit duplicate/shard semantics. Separate plain rows, quoted
delimiters, multiline records and cross-shard conflicts by structural groups or
predeclared composition holdouts. Region/customer renames and reordered records
are variants of the same group, not new task families.

## Collection and evaluation boundary

Start with a few fixtures per family, each small enough to inspect from ordinary
CLI output. Freeze goals, allowed write paths, initial fixture hashes, verifier
version and split owner before collecting trajectories. All proposer/model
seeds, alternative menus and repeated attempts from a fixture remain together.
The three families are useful distinct surfaces, not proof of unlimited task
coverage or independent real-world domains.

Before accepting a task, independently check: the untouched workspace fails;
at least one ordinary command sequence succeeds; a no-op, overbroad change,
plausibly incomplete repair and protected-file mutation fail for the intended
reason. Keep those author-only gate receipts away from both models. Malformed
commands, tool failures and exhausted budgets stay in the dataset. Distinguish
task failure from container/verifier failure.

Record the public goal, visible observations, complete offered command strings,
chosen ID, execution results, changed paths, terminal checks and elapsed/tool
costs. A proposer may generate no successful option; report that candidate-set
limitation separately from selection error. Selected-option probabilities are
not automatically probabilities of eventual task success. Evaluation should
execute complete proposer–selector trajectories under the same command/time
budget, including a genuine starting-model baseline, rather than grade only a
single option match or an unexecuted proposed command.
