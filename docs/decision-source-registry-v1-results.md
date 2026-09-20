# Five training mechanisms, with uncertain outcomes preserved

The prospective data index passed source reconstruction, probability-mass,
serialization, token-length and split checks. It reuses existing execution
receipts instead of running a duplicate collection. **It has not trained or
evaluated a model.** The next comparison still requires qualified live environment
adapters and a frozen learning recipe.

| Mechanism | Role | Source forecast rows | Distinct rendered/token inputs | Uncertain inputs |
| --- | --- | ---: | ---: | ---: |
| Configuration changes | Training candidate | 504 | 308 | 40 |
| Database repair | Training candidate | 504 | 308 | 40 |
| ToolSandbox application delivery | Training candidate | 1,512 | 468 | 108 |
| Filesystem mutation scope | Training candidate | 560 | 224 | 24 |
| Reservation transactions | Training candidate | 1,152 | 972 | 168 |
| Report generation | Development | 504 | 308 | 40 |
| Calendar | Reserved transfer | 1,440 | 576 | 149 |
| Publication, including intermediate histories | Exposed diagnostic | 1,176 | 728 | 32 |

The training candidates total **2,280 distinct forecast inputs across five
mechanisms**, of which 380 have multiple possible outcomes. They derive from
4,232 source forecast rows. Across all roles the index contains 3,892 inputs from
7,352 selected source rows. These are input counts, not independent tasks, newly
executed branches, optimizer presentations or evidence of model improvement.
The original corpora retain paired next-action and observation-value questions;
this index selects only consequence forecasts. Entire mechanisms retain one role.

The configuration and database-repair sources each have one authored task root.
Application delivery has one root and nine initial database worlds. Filesystem
scope has two task structures and four sharing configurations. Reservations have
one existing mechanism and six initial stock/account worlds. Repeated goals,
fees, horizons and presentations do not multiply those underlying mechanisms.
The source qualification reports and the separate branch-reference accounting
are included in the [artifact bundle](../results/decision-source-registry-v1/).

## What collapsing rows means

An identical public question can occur in several compatible hidden worlds.
If three executed worlds complete and one fails, the combined target is
`[0.75, 0.25]`, with all four source identities and receipts retained. It does
not become a hard success label, nor a set saying either answer deserves full
credit. Conditional distributions already computed from reservation priors are
preserved exactly. The source auditors reconstruct their labels from actual
database states; no model prediction enters a target.

The full reservation source has 3,456 rows: 1,152 canonical event identities
presented three ways. Its original presentation contains only **972 distinct
rendered inputs**; all three presentations contain 2,916. The index uses the
original presentation once and retains the duplicates' lineage. Duplicate
reservation inputs have identical target distributions. This corrects the
interpretation of 1,152 as unique model inputs. The old reservation validation
was already exposed and had exact training-input overlap, so none of it becomes
a pristine test in this study.

The earlier reservation code already supplies a padded soft-target cross-entropy
loss in `puffer_lab.consequence_train.soft_loss`; reuse it rather than silently
passing these targets through the hard-label trainer. Its seven existing tests
passed. Six new index tests cover uncertainty, lineage, hidden option-identifier
renaming, rejected ownership collisions, invalid targets, private input fields,
and equivalence of soft cross entropy with repeated outcome supervision,
including gradients.

Future reporting must distinguish **expected Brier score**
`sum(p*p) - 2*sum(p*q) + 1` from **excess Brier error** `sum((p-q)**2)`, where
`q` is the verified outcome distribution. The latter removes irreducible
uncertainty. The old reservation report's `brier` field used excess error;
mixed-decisions-v1 used individual observed outcomes. These values cannot be
compared as if they were the same metric. The next recipe must also state how
mechanisms, unique inputs and source multiplicities are weighted.

## Checks and remaining work

Every source was reconstructed by its existing offline auditor, including
executed-label provenance and frozen source hashes. The accepted application
replay collection is used. The new filesystem and calendar Linux parity
receipts passed. All index inputs fit the pinned 9B tokenizer without truncation;
the largest forecast is 1,101 tokens. This forecast-only length is different from
the larger decision and actor-history lengths reported by source audits.

There are no token-identical inputs across mechanism roles, or between this
index and the unchanged general replay (4,096 questions), retention (622) and
general transfer (1,128) pools inherited from evidence-decisions-v2. Exact-token
checks do not prove semantic independence; whole-mechanism ownership supplies
the stronger separation. Private metadata remains outside model messages.

The first local indexing attempt stopped because the two parity reports use
different field names. Its rejection, original source and freeze are preserved.
The corrected attempt passed. This performed zero new database/environment
executions, model calls, optimizer steps or training presentations.

Next, qualify live filesystem/reservation adapters and the calendar evaluation
adapter without opening model scores. Reuse the actual SQLite reservation
engine, not just its faster C mirror. Actor prompts must promise live replanning;
fixed continuations remain explicit only in forecasts. Audit action costs,
terminal rewards, failed-command recovery, arbitrary in-catalog histories and
the full public/private boundary. Reuse guarded updates and the existing proper
forecast loss, preserve general replay, and freeze the equal-start 9B comparison
before any rental. Calendar stays out of training and checkpoint selection;
publication stays an exposed diagnostic. No new budget is implied.
