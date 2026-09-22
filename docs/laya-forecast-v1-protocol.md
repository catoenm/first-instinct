# Bounded external forecast baseline

This is an inference comparison on the 308 already exposed report-development
forecasts from the frozen live-tools pilot. It introduces no new task, label,
training, temperature fit, checkpoint selection, or reserved release evaluation.
The cohort contains 268 deterministic and 40 ambiguous questions, in one workflow.
These dependent questions do not support a broad generalization claim.

Compare these fixed checkpoints, in order, serially on the same GPU:

1. `convaiinnovations/laya-typed-decisions` at
   `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`, using reviewed upstream runtime
   `573e5b62696ba441230cd6be71d593331b5d23af` and its shipped calibration.
2. The unchanged Qwen3.5-9B foundation at
   `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
3. That foundation with the original supervised step-2742 adapter, whose weight
   file SHA-256 is
   `882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.

All receive the same complete state, question and option descriptions through
their existing native interfaces. No shared compact-action transformation is used.
Laya native formatting must retain every component; Qwen encoding must exactly
reproduce the already frozen input tokens. Source identifiers, outcome labels and
probability targets remain outside both prompts. Labels come from prior executed
branches; none are supplied by a model prediction.

Report expected outcome Brier error, excess Brier error, stable log loss, excess
log loss and expected choice accuracy, separately for ambiguous and deterministic
questions. Preserve native unrounded float32 probabilities. Compute log loss with
float64 log-softmax of the actual float32 calibrated logits, and verify probability
parity within 0.000002. This avoids clipping losses when a float32 tail underflows.
Record every underflow. Never normalize the rounded native public probabilities.
Entropy-based confidence and the independent act/abstain head are not success
probabilities. No question-dependent model routing or temperature adjustment.

Before scoring, verify checkpoint files and offline CUDA placement and run three
synthetic menu checks twice per checkpoint. They test formatting, native output
capture and repeated-forward agreement (maximum probability difference 0.00001),
not semantic correctness. Abort on failure. Batch size is one throughout; there
is no padded batching claim. The Qwen foundation is loaded once, scored, then
the original adapter is attached for the third fixed comparison.

For each model, record loading time, the first instrumented synthetic call, and
instrumented cohort time separately. Native timing uses the same three questions
chosen prospectively by shortest, median and longest Qwen token length (ties by
question identity). After three warmups per question, measure five serial calls
each with CUDA synchronization. Include tokenization, transfers, forward pass and
the native output computation, excluding file writes and compatibility hooks.
Record each duration and resident/peak GPU allocations. This is a small local
timing sample, not a hosted concurrency or production-throughput benchmark.

Qualify locally using tokenizers and tiny numeric fixtures only. No foundation
model on the Mac. Then allow one inference rental within the original remaining
authorization: at most $6 total stage hold, one 48-GB-or-larger GPU at at most
$1.60/hour, 90 minutes including setup and recovery. Install an independent remote
stop guard before setup. Allow at most 30 minutes of setup and 30 minutes of model
work, bounded further by the rental deadline with recovery time reserved. Archive
partial failures, recover and verify artifacts, then delete only the owned pod.
No automatic retry, extension, recurring hosting or training follows this pilot.

Completion requires all 924 primary predictions, complete information for every
input, intact lineage, numerical checks and independent metric reconstruction.
An incomplete cohort is reported as incomplete, without cross-model claims from
unequal subsets. A good result motivates a new-mechanism test, not release promotion.
