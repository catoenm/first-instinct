# Paired questions and balanced answer positions

The local candidate index is qualified: **4,721 canonical questions** reuse
already verified executions across seven existing task families. They include
**3,979 consequence forecasts** and **742 decision or inspection-value questions**.
No model has consumed this package yet. It is not a new trained checkpoint or
evidence of better task completion.

| Existing family | Forecast questions | Decision and inspection questions | Canonical questions | Cyclic presentations |
|---|---:|---:|---:|---:|
| Configuration changes | 308 | 44 | 352 | 1,122 |
| Database repair | 308 | 44 | 352 | 1,122 |
| Application delivery | 468 | 52 | 520 | 1,690 |
| Filesystem mutation | 224 | 32 | 256 | 816 |
| Reservations | 972 | 0 | 972 | 2,187 |
| Retail procedures | 31 | 36 | 67 | 206 |
| Competing database writers | 1,668 | 534 | 2,202 | 7,170 |
| **Total** | **3,979** | **742** | **4,721** | **14,313** |

The 14,313 presentations include each canonical order once. The remaining 9,592
are rotations, not independent questions or new tasks. Every semantic option
occupies every answer position once, and the targets move with their options.
This addresses a specific exposure gap in the
[completed learning experiment](oracle-capacity-completion-v1-results.md): its
starting decisions always used one menu order, and the combined model changed
its action in 16.6% of full-panel comparisons when that order was reversed.
The data change has not yet been tested for a learning benefit.

## What the labels mean

The index keeps three different kinds of supervision separate:

- 3,979 outcome distributions, including 525 legitimately uncertain forecasts.
- 706 sets of acceptable decisions, preserving all tied choices.
- 36 retail decision-preference distributions, which are not outcome confidence.

Each forecast retains its original immediate or continuation contract. A
question about a fixed procedure is not a forecast of whatever the model might
choose to do later. Decision and inspection questions join forecasts by public
evidence within a source; their complete prompts still retain costs, goals and
the declared procedures. Reservations remain forecast-only because this source
does not provide independently justified paired decision labels.

The competing-writer index includes both fixed and optimal-public-continuation
questions. Its 264 contract-specific evidence groups share underlying worlds;
they are not 264 independent situations. The optimal-continuation source keeps
its original training-mechanism diagnostic role. This candidate index does not
silently admit those questions into a new training loss.

## Existing execution evidence, not newly generated worlds

This stage adds **zero underlying tasks, execution branches, model calls or
optimizer updates**. The source collections have different accounting units:

- Configuration and database repair each contribute four world-and-goal tasks
  and 756 primary alternative/continuation branches from the
  [qualified shell curriculum](decision-curriculum-v3-results.md).
- [Application delivery](application-curriculum-v1-results.md) has 18
  world-and-goal tasks, 2,268 primary alternatives and 2,268 separate replays.
- [Filesystem mutation](filesystem-decisions-v1-results.md) has eight
  world-and-goal tasks, 840 primary alternatives and 840 separate replays.
- [Retail procedures](release-retail-questions-v1-results.md) reuse three goals,
  two physical mechanisms, ten initial states, 120 primary alternatives and
  120 separate replays.
- [Competing writers](revisioned-questions-v1-results.md) have four world-and-goal
  tasks and 96 fixed-procedure primary branches. The later
  [public-continuation oracle](revisioned-oracle-v1-results.md) reuses 1,422
  terminal paths and reconstructs 1,692 world/history/action edges in that same
  mechanism. Paths, edges and prior branches must not be summed as independent
  tasks.
- The reservation forecasts reference 486 distinct primary branches in the
  [existing branch-reference census](../results/decision-source-registry-v1/branch-reference-accounting.json).
  This is selected-source coverage, not the size of the whole collection.

Repeated executions verify outcomes; they do not add probability mass. Some of
these sources were used by earlier experiments. New consumption for this package
is zero; these counts do not claim that every source is previously unseen.

## Qualification and remaining work

Seven focused tests passed. The actual preparation checked accepted source
hashes, original roles, target semantics, serialization, cyclic position
coverage and token identities. The longest presentation is 1,910 tokens, with
zero truncations and zero exact token overlap against the inherited general
replay, retention and transfer pools. Preparation took 11.5 seconds, peaked at
759 MB of process memory and added no swap. No foundation model ran on the Mac.

The first preparation correctly rejected forecasts that joined more than one
original ownership group: 180 reservation forecasts retain two group aliases.
The corrected index preserves every alias and binds each entire mechanism to
one ownership component. The rejected attempt is preserved privately. Targets
and underlying executions were not changed. A new component identifier cannot
erase relationships to earlier data.

Three concrete steps remain before a bounded training pilot:

1. Add and verify public tool-effect descriptions where they are incomplete,
   especially configuration and database repairs. Saying “apply the requested
   update” does not explain what a command actually changes. Keep this as a
   separate input overlay, preserving the original prompts and executed labels.
2. Admit the complete mixture against current ownership and consumption records,
   resolving old group aliases. Exact token non-overlap alone does not establish
   independent task families or transfer.
3. Connect and qualify the distinct target types in the learning consumer, with
   balanced menu positions, adequate early-decision exposure and general replay.
   Freeze bounded training, final evaluation and recovery windows before renting.

The previous supervised checkpoint remains selected. Release still requires
better executed decisions and forecasts without material general-task damage;
fitting these exercised families would not demonstrate unfamiliar-task transfer.

See the [protocol](paired-curriculum-v1-protocol.md),
[index implementation](../tool_lab/paired_curriculum.py) and
[aggregate counts](../results/paired-curriculum-v1/summary.json).
