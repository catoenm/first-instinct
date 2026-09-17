# Executable evidence: pilot collection protocol

This is a small experiment in data collection for probability learning. It uses
24 authored Python contracts, not imported SWE-smith tasks or production records.
No Jev calls or external paid model calls are required. There is no reinforcement
learning in this experiment; direct-label learning first tests the data pipeline.

## Data and visibility

The target is whether a candidate passes a fixed private suite of 32 inputs.
The suite mixes explicit contract fixtures and fresh generated inputs. It is
not a proof of correctness on every possible input. Inputs satisfy each contract.
Candidate execution and reference outputs use separate code paths. Both were
authored in this project; this is not independent human validation.

Generate up to 24 proposals per task from two authored implementations plus
literal, arithmetic, predicate, call and composed transformations. Do not assign
labels by transformation intent. Keep stable per-input exceptions as failing
checks; quarantine syntax/worker errors and nondeterminism. Keep rejected proposals.
Each screen runs five public inputs twice in fresh processes. Four text views
reveal two checks, those checks plus an explicit copy, a third check, or the
same two checks at a higher independently assigned inspection price. The two
remaining screened checks are logged but not used in model inputs.

The model sees only the contract, code, revealed inputs and outputs, and price.
Case identifiers, split, mutation mechanism, source lineage, private inputs and
private verdict are never encoder inputs. The price cannot affect correctness.
Distinct tests need not be statistically independent. The question is fixed.

Ten tasks train the collector; five other tasks provide development validation;
five further tasks test transfer to new tasks. Four tasks in the predicate family
are entirely held out. A further final set uses held-out boundary-comparison and
input-slicing transformations on final tasks. Other training transformations can
also cause boundary bugs: the held-out unit is the transformation mechanism,
not a guarantee that a semantic class of bugs has never appeared.

All siblings from an underlying task stay in one task split. Task definitions,
fixtures and proposal mechanics are authored and sanity-checked upfront. Final
candidate pools and their execution verdicts are opened only after all final
model weights are sealed. These
are small authored holdouts with shared programming patterns, not a broad or
pretraining-uncontaminated software benchmark.

## Collection comparison

Three collection seeds (11, 23, 37) each compare random, coverage and adaptive
selection. All recipes share a seed-dependent bootstrap of two proposals per
training task (20 total). Each then acquires five batches of 16 new candidates,
giving exactly 100 private-suite queries and 400 labeled text views.

Random selection is uniform without replacement. Coverage selects uniformly
among least-used task-by-transformation strata, then uniformly within a stratum.
Adaptive selection combines forecast uncertainty and changes under copying or
price variation, computed before querying outcomes. Each draw is a mixture of
25% uniform sampling and 75% uniform sampling from the highest-scoring 20% of
remaining cases. Scores stay fixed within a batch; the model refits between
batches. Conditional selection probabilities and reasons are logged. These are
not marginal inclusion probabilities and are not used for importance weighting.

Every acquisition runs the same private suite twice: 64 test executions, 6,400
logical private test executions per recipe/seed. All recipes pay the same common
screening and feature-extraction overhead. This matches verification-query and
test-execution budgets, not wall-clock time or money. Actual measured execution
times are reported. A content-addressed cache shares physical executions between
recipes; receipts distinguish logical cost, standalone measured time, and new
physical work. Selectors cannot access an unqueried label through the cache.

Validation labels have a separate common budget. Final evaluation has a separate
budget and cannot inform training. Selection uses no private verdict, and no
inferred success/failure label based merely on a proposed edit.

## Fixed learner and references

The frozen Microsoft DeBERTa-v3-small encoder uses revision
`a36c739020e01763fe789b4b85e2df55d6180012`. Average non-padding token outputs,
then normalize the vector. Never truncate an input; reject the run if a context
exceeds 512 tokens. Center features using the unlabeled training pool only.
Do not train the encoder. It is a general language encoder, not demonstrated
here to be a competent code reasoner.

Fit a scalar logistic head from zero after the bootstrap and each collection
batch. Use full-batch Adam, learning rate 0.02, 200 steps, and penalty
`0.05 * sum(weight**2)`; do not penalize the bias. Train on all four views of each
selected candidate, with equal weight. Keep the last collection round. Validation
is diagnostic, never a checkpoint selector. The three seeds measure variation in
collection, not stochastic optimizer variation.

Fit the same type of head to an engineered visible-check reference: whether any
check failed, fraction failed, and number of distinct revealed checks. It ignores
copy repetition and price by construction. Also retain a smoothed training-label
prior. These references expose whether the language model adds information
beyond reading test verdicts or guessing the collection success rate.

## Measurements and interpretation

Report Brier score (mean squared probability error against binary outcomes),
log loss, threshold accuracy, calibration bins, and error versus decision coverage.
Report automatic pass counts and failures separately so always rejecting programs
cannot masquerade as useful autonomous approval. Thresholds 0.7, 0.8 and 0.9 are
fixed descriptive points, not optimized deployment guarantees.

Report each view and domain, all seeds, copy/price changes and the Brier change
after a fresh check. Group related views and mutations by task when describing
sample size. Unlike the earlier simulator, exact conditional probabilities are
unavailable. Brier score combines calibration and discrimination; do not label
every improvement as a pure calibration gain.

The primary comparison is initial-view Brier score on new tasks across the three
collection recipes, with the two references and all shifted domains retained.
Inspect the verifier's blind spots by counting candidates that pass visible checks
yet fail the private suite. The proposal population is synthetic; measured pass
rates do not estimate production failure rates. No human audit or rare-failure
certification is claimed.

One development collection seed (101) can guide implementation or budget changes
before freezing this protocol. Record any such changes and its results separately.
The development run used seed 101 and the stated settings without learner or
acquisition retuning. Its validation initial-view Brier scores were 0.12129
(random), 0.11413 (coverage), and 0.12649 (adaptive), on 76 candidates across
five validation tasks. It selected 20, 29 and 29 private-suite passes out of
100 queries, respectively. Development implementation fixes covered stable
mutation ordering, compact evidence rendering, cache/text integrity checks,
bootstrap probability receipts, source sealing and offline replay. The maximum
development context was 343 tokens. No final-candidate outcome metrics guided
these changes.

Freeze source and protocol before the three-seed final comparison. Retain the
cache, receipts, request text, embeddings, every collection-round head and complete
selection logs for reconstruction. Full-encoder fine-tuning, reinforcement learning,
real-repository transfer, and a human audit are follow-ups, not completed claims.
