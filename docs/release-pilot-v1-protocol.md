# Bounded 9B release continuation pilot

This continues the admitted `release-mixture-v1` pack. It starts from the original
supervised step-2742 rank-16 adapter, SHA-256
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`, on
Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
It is supervised tool/decision/outcome learning, not online reinforcement learning.

The prospective maximum is 160 optimizer updates, each with 32 general replay,
28 new tool choices and four verified questions. Questions have equal weight
within an update. General and tool pools are shuffled without replacement;
verified families rotate equally, with at most two presentations per question.
Verified examples include both application decisions and outcome distributions.
Seed 20260922 fixes the complete schedule before scores. No target-dependent
sampling is permitted. AdamW uses learning rate 0.00001, weight decay 0.01 and
gradient norm clipping at 1. Microbatches contain two questions, sorted by length
within each update; effective batch size is 64. Maximum full context is 4,096
tokens, without truncation. Only internal adapters change.

The pilot evaluates all 2,217 tool development questions, averaged equally by
their 27 server groups; questions spanning multiple development servers contribute
to each involved group. General retention uses up to 32 deterministically sampled
questions per original task (157 tasks), with equal task weight. Declared product
slices are dialogue intent, supplied rules, evidence judgments and incident
priority. Their accuracy is a question average with support reported. These are
limited proxies: they do not establish broad ordered-score quality. No numeric
ordinal error can be inferred from arbitrary option positions, so that capability
remains a separate release requirement.

The outcome suite contains all 366 admitted development distributions, reported
separately for the existing report mechanism, application task completion and
application state change. Both categorical log loss and expected Brier error
must be no worse than the original checkpoint in every group, allowing only
0.000001 numerical tolerance. The 24 development application decisions are
reported as a diagnostic, not used to override the tool/general/outcome gates.
No final/reserved-transfer, known-test or calibration rows enter this pilot.

Evaluate the original weights, then every 40 updates. An eligible candidate must
retain general macro accuracy within 0.01 and log loss within 0.02; no declared
product slice may lose more than 0.03 accuracy. Stop immediately if a retention
or product-slice gate fails, on nonfinite loss/gradient, or after two checks without
an eligible tool improvement. Among eligible checkpoints, maximize tool macro
accuracy, using lower tool log loss as a tie-breaker. The original checkpoint is
always available. Advancing toward a main run requires at least 0.03 improvement
in tool macro accuracy and all gates. Do not relax thresholds after seeing scores.

Before optimizer updates, compare individually scored shortest/longest scheduled
inputs with their left-padded batch (same decisions, maximum probability deviation
0.025), run a backward pass at the longest context, verify finite nonzero adapter
gradients and unchanged weights, and report GPU memory. These diagnostic
presentations are separate from training. After update one, save the adapter,
optimizer, random-number states and exact schedule cursor. Exit and resume in a
new process, verifying the saved file digest and canonical trainable tensor hash.
The local CPU fixture additionally compares a resumed stochastic optimizer update
with an uninterrupted update exactly. The GPU restart does not claim bitwise
equivalence across different CUDA kernels or hardware.

The training process has a three-hour wall limit including its evaluations and
restart. One H200 rental is bounded to four hours at at most $5.40 per hour,
within the existing $25 pilot allocation ($21.60 compute ceiling and $3.40
storage/recovery reserve). Fresh provider prices and billing are checked before
creation. A provider-authenticated stop guard runs on the rented machine. Artifacts
and hashes are archived on persistent workspace storage before the deadline;
collection verifies local copies before deleting the rental. Laptop connectivity
does not determine whether the run completes or its compute stops.

Record unique prepared questions, scheduled presentations, successfully completed
optimizer updates, backward presentations (including any uncommitted interrupted
update), tokens, per-pool counts and checkpoint lineage separately. The full
371,278-question release pack is prepared data, not a claim that this pilot visits
it all. Leave the public demo on its existing checkpoint until independent release
evaluation and serving qualification pass.
