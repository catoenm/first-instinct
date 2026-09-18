# Outcome-v3: qualify better data before training

**Proposal only: no collection, inference or rental has started.** First qualify
a small SQLite transaction dataset on local CPU. Then test schema coverage and
mechanism breadth separately, before adding another reinforcement-learning arm.
This is a finite executable-data experiment, not a claim about Jev's private
training recipe or arbitrary-question calibration.

## What the results support

The [fixed transfer comparison](../results/toolsandbox-transfer-outcome-execution-v1/report.md)
reports that all six eligible adapter identities still choose stop at all 48
initial states. Initial expected regret remains 20.40625 research credits.
Probability scores change without a primary decision improvement. Ten original
roles alias those six identities; hybrid-77's two unfinalized roles remain
excluded. These are 48 variants of one authored mechanism, not broad transfer
evidence. The independent raw-response audit passed.

Two observations motivate controls, rather than establish causes:

- The [order audit](toolsandbox-transfer-order-results.md) found large cost
  probability drift and 69/96 cost modal flips after one reversal. Original and
  reversed calls occurred at different times without original-order repeats;
  ordering and temporal variation were not fully separated.
- The [local divergence analysis](outcome-v2-divergence-results.md) found that
  terminal-value substitutions removed much more local ranking error than cost
  substitutions. Those calculations were on existing retry/workflow states,
  not executed policy improvements or a diagnosis of ToolSandbox transfer.

Read-only counts from the frozen training rows and saved tokenizer audits show
additional distribution shifts. Singleton costs are excluded below.

| Forecast corpus | Outcome options | Cost options / numeric range | Outcome tokens | Cost tokens |
|---|---:|---|---:|---:|
| Retry training | 3 | 2–8 / 2–37 cents | 610–721 | 600–713 |
| Workflow training | 3 | 2–12 / 1–104 cents | 716–809 | 731–933 |
| Transfer initial states | 8 | 9, 12, 14, 22 / 5–227 research credits | 1,078–1,088 | 1,110–1,333 |
| Transfer phone states | 8 | 9, 12, 14, 22 / 2–205 research credits | 1,195–1,217 | 1,224–1,458 |

Table inputs: `output/outcome-data-v2/train.jsonl` and
`output/outcome-prepared-v2/train.jsonl`, pinned by the published
[raw manifest](../results/outcome-v2/raw-manifest.json) and
[prepared manifest](../results/outcome-v2/prepared-manifest.json); transfer
[public questions](../results/toolsandbox-partial-v1/public-questions.jsonl) and
[saved token audit](../results/toolsandbox-partial-v1/prompt-audit-final.json).
Lengths use saved token IDs/audit counts; this inspection made no tokenizer or
model call.

Thus all transfer prompts exceed the training forecast length range, while
remaining within the 1,536-token limit. Menu size, class meanings, numeric range,
units, prompt structure and mechanism all changed together. Broader mechanisms
alone are **not** established as a remedy. Insufficient optimization, schema
sensitivity, candidate-menu limitations and inadequate terminal reasoning remain
untested explanations; more GPU steps are not yet justified by these results.

## The data contract

Retain the existing public `state / question / options[{id,description}]`
interface and separate per-action outcome and future-cost questions. A label
means: execute offered action **a**, then versioned continuation **π**, through
the declared horizon. Publish that continuation, tool semantics, costs,
terminal utility table and finite prior in the input. It must permit useful
information gathering; a stop-now continuation cannot value a later write.

For each visible history, enumerate actual hidden database/filesystem worlds.
Group worlds by the **complete identical public history**, preserving result
order and stable visible IDs. Compute the posterior from the prior and the
likelihood of the actual observations, then execute every offered action and
continuation in every supported world. For deterministic tools the likelihood
is an indicator; any modeled randomness requires its own enumerated tape and
declared probabilities. Never swap hidden state after an observation or create
uncertainty by flipping labels.

Store exact rational outcome/cost marginals, their joint distribution, terminal
state/frame checks and execution receipts. Preserve empirical outcome/cost pairs
sampled from the same executed branch for compatibility with outcome-v2's
categorical training loss. Exact probabilities remain audit targets in the
first training comparison; full-distribution training is a separate hypothesis.
Separate marginals suffice only for additive terminal utility minus cost.
Action-choice probabilities are not action-success probabilities.

Vary supported outcome partitions, public cost supports, menu order and wording
on training roots. Use meaningful exhaustive partitions of actual outcomes,
not arbitrary distractor padding; retain a mapping to common coarse outcomes
when refining categories. Preserve utilities when projecting those categories.
Permutations and wording variants share one root/group and count as repeated
presentations, never independent labels. Counterbalance semantic IDs across
answer positions; include declared unit rescalings that preserve every utility
ordering. Aim for varied short and long inputs within 1,536 tokens and 36
choices. Reject overlength examples before model use; do not truncate priors or
continuations. Include contemporaneous identical-input repeats in future order
diagnostics to measure numerical/time variation separately.

## Fresh mechanisms

| Family | Actual execution and new failure structure | Verifier |
|---|---|---|
| Relational reservation | SQLite multi-row writes, CHECK/foreign-key/unique constraints, transaction rollback versus partial autocommit writes | Exact inventory, account and reservation tables; requested atomic effect and protected rows |
| Filesystem mutation scope | Local temporary-tree `lstat`/`readlink` and atomic replacement; aliases can refer to different resources | Declared file contents, types, modes and links; all protected paths unchanged; no access outside the temporary tree |
| Calendar scheduling | Pinned local `zoneinfo` plus SQLite calendar rows; ambiguous local times and interval conflicts | Explicit UTC instants, half-open interval rules and exact unrelated-event preservation |

These reuse real engine operations but have authored goals, priors and costs.
They are not three independent real-world datasets. Reserve the entire calendar
family for a new diagnostic; train initially on reservation and filesystem
families plus fresh old-mechanism controls. Filesystem and calendar adapters
must first pass their own isolation, deterministic replay and platform-version
checks. Neither is implemented by this proposal.

The [external-source review](executable-data-next.md) still identifies AppWorld
as a richer subsequent integration, subject to setup, verifier and protected
artifact requirements. It is unnecessary for the first CPU pilot. Do not spend
the previous ToolSandbox branch reserve or reuse its exposed four-world corpus.

## Smallest independently useful CPU qualification

Implement **only the SQLite family first**, in new files and a fresh temporary
database. No ToolSandbox calls, downloads, model inference or rental. It tests
the execution/posterior pipeline, not learning or broad generalization.

A representative request reserves one unit each of A and B. Four valid initial
worlds share the same observed total on-hand stock: `(1,1)`, `(2,0)`, `(0,2)`,
and `(1,1)` with the required account absent. Actual SQL enforces nonnegative
stock and the reservation's account foreign key. The first three distinguish
aggregate from item-level availability; the fourth can fail after stock writes
and exposes rollback. No initial database violates its own constraints.

Offer at most four programs: stop, reserve with separate autocommit statements,
reserve atomically, and inspect per-item stock. After inspection, the declared
continuation checks account eligibility and reserves atomically only when all
requirements are met. Programs can use only the visible request/results. The
first and fourth worlds share the per-item observation, giving a nontrivial
posterior after a real query. Account, quota and pre-existing-key variants can
exercise different native constraints; no ambiguous acknowledgement or retry
transport is introduced.

Freeze six request/constraint program groups, four prior/cost variants each:
**24 root configurations**. Assign whole groups before collection: four train
groups (16 roots), one validation group (4), one sealed diagnostic group (4).
Keep entity, menu and wording variants with their group. This is a combination
holdout within one engine family, not unseen-primitive transfer.

At most four hidden worlds, four programs and two recorded decision prefixes
per world yield **768 distinct execution branches**. Replay each once for
**1,536 total attempts**, plus at most **256 guard/failure attempts**: a hard
**1,792-attempt cap** recorded before every attempt across tests/processes.
Limit each branch to eight public tool calls; record SQL statements separately.
At most five distinct public contexts per root produce **960 marginal question
records** before singleton bypasses. Two initial presentation variants cap the
rendered pilot at **1,920 rows** without multiplying evidence. Stop at 30 local
CPU minutes or 200 MB of output. Failures consume the cap; no automatic reruns.

Required gates, before expanding the dataset or using a model:

- Every reset/replay reproduces state and receipt hashes; no-op, wrong-target,
  partial-write, rollback and protected-row corruption controls have the
  specified mutually exclusive labels. Infrastructure errors never become
  task-success labels.
- Histories within each posterior group are byte-identical; independent
  recomputation from execution receipts reproduces every rational marginal,
  joint distribution and expected utility. Candidate arguments and cost menus
  derive only from public data and declared bounds.
- The predeclared prior/cost grid contains genuine query value and at least one
  action-order reversal as query cost changes. If it does not, reject the
  proposed family design; do not manufacture labels or claim information value.
- Split groups do not overlap; all labels/failures are retained; deterministic
  and uncertain targets, class support and action coverage are counted honestly.
  Cached-tokenizer audit fits every presentation without truncation. Independent
  review approves the verifier and public-input boundary.

Any failed gate produces a partial qualification report and **no GPU stage**.
Passing only establishes usable executable data; this tiny pilot is not a
sufficient training corpus. An expansion needs its own frozen root/execution
budget, not an implicit continuation past these limits.

## Later matched comparison, only after qualification

Use the identical released supervised adapter as the common start. Before any
updates, save a **genuine starting-adapter test evaluation** on the newly frozen
test roots, including complete controller traces and independent forecast
scores. Preserve it regardless of later checkpoint selection; update-zero
validation is not a test baseline. Seal those test outputs until training and
validation selection are complete. Evaluate native actor and forecast controller
separately, with the same decision/inference budget for every arm.

First compare outcome-only training on three predeclared data treatments:
**A:** fresh retry/workflow roots with the original schema;
**B:** the same roots/labels with varied schema and presentation;
**C:** fresh old plus reservation/filesystem roots with B's presentation coverage.
A–B tests the schema/presentation package; B–C tests added mechanism coverage.
Neither isolates every factor inside its package. Use the same two declared
seeds, optimization schedule, general replay and retention gate. Do not introduce
PPO, soft-distribution loss or a new learning rate simultaneously.

Match labeled branch pairs, presentation reuse and cumulative training tokens
prospectively; stratify menu sizes and lengths without consulting predictions.
Apply a common cap on executed data-collection transitions and record actual
reset/tool work, exceptions, labels, forwards and optimizer steps. Where engine
costs prevent exact matching, report that mismatch instead of calling the arms
equal-work. Duplicate renderings and exact enumeration are not free new labels.
A later reward-only/hybrid comparison must separately match supervision and
compute, and disclose hybrid's extra labels/work.

Use training-only fixtures for debugging and validation only for a common
checkpoint rule. The exposed v2 and ToolSandbox roots, their labels, and the
order-audit prompts are excluded from training, selection and new tuning.
Freeze fresh test generators and family ownership before collecting results.
Require validation gains in both proper probability scores and controller value
across families, without retention failure, before a larger run. No eligible
gain, collapse to one action, or dependence on one ordering ends the pilot.
Final test failure is reported; it does not trigger another search on those
roots. Exact stopping thresholds and an evaluation-inclusive compute ceiling
must be frozen before any paid model stage. This document starts no such stage.
