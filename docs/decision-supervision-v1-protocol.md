# One decision-focused supervised comparison

The September 23 reset pauses new environments, data expansion, larger models,
and reinforcement learning. Compare the original supervised Qwen3.5-9B
step-2742 checkpoint with one candidate trained directly on existing verified
decision and inspection questions. Better forecasts alone cannot earn success.

This reuses the paired-capacity trainer and admitted data. The training change
is **removing forecast supervision**: keep the existing teacher schedule,
general-task replay, learning rate, internal adapters, and transactional update
guards. No new preservation loss, optimizer, proposer, or serving architecture
is introduced. This is a prospective comparison with the original model, not
an isolated causal comparison with the unrecovered paired run.

## Fixed data and dose

Use the existing 742 canonical decision/inspection questions across six authored
families: configuration repair, database repair, application delivery, filesystem
scope, retail procedures, and concurrent database changes. Their targets come
from already verified executions or the qualified finite-state oracle. Preserve
the distinction between actions under a displayed continuation and freely
replanned actions; do not relabel the former as unrestricted optimal actions.

Each update presents six teacher questions per family, the twelve existing
starting database decision/inspection questions, and 32 general replay questions.
Cycle option positions using the existing schedule. At 128 accepted updates this
is **6,144 teacher presentations and 4,096 general replay presentations**. The
teacher pool contains 742 canonical questions; position variants and repeated
visits add no independent tasks. Forecast training presentations are zero.
The newly prepared 160 identifier-evidence questions are excluded from this run.

Start from the original adapter hash
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.
Keep the existing learning rate 0.0000008 and equal teacher/replay objective
weights. Stop at 128 accepted updates, a failed safety/update check, or the
runtime bound. A deadline before 128 means incomplete dose, not a negative
conclusion about learnability. Save actual final weights and evaluate them.

## One fixed evaluation

At the start and every 32 accepted updates, run exactly the same original and
candidate evaluation contracts:

| Executed mechanism | Episodes | Prior exposure |
| --- | ---: | --- |
| Concurrent database changes | 12 | Training-owned capacity cohort |
| Report workflows | 36 | Previously used development cohort |
| Calendar workflows | 80 | Previously scored transfer cohort; now a known regression set |

The primary measure is **verified completed-task rate**, giving the three
mechanisms equal weight and retaining their existing within-mechanism weighting.
Also report raw completions, incorrect changes, and reward after action costs.
All 128 episodes must finish before a measurement is valid. These are previously
exposed mechanisms, not fresh evidence of generality. Do not open the reserved
release tool or telecom model scores during this experiment.

Retain both 2,064-question database panels, 308 report forecasts, and the existing
622-question retention panel as secondary diagnostics. Add the same frozen
3,465-question general development panel used by the completed generalist run,
including its four product slices. No model score selects or changes these cases.
Forecast targets remain available for evaluation only. No new trajectories are
fed back into training.

A candidate passes this experiment only if completed-task rate rises by at least
five percentage points, at least two mechanisms improve, none loses completion
rate or increases incorrect changes, and average cost-adjusted reward does not
decline. Keep the original database/report safety bounds: neither return may
fall by more than 0.02, neither measured forecast Brier error may rise by more
than 0.02, and general accuracy/log-loss limits remain one percentage point and
0.02. Apply those general limits to both retention panels; no product slice may
lose more than three accuracy points. Calendar reward and incorrect-rate safety
limits are 0.02. Failed safety checks stop learning immediately.

Select only qualifying checkpoints, ranked by completed-task rate. Keep the
original when none qualifies. A final checkpoint already fully evaluated at the
same accepted update reuses that evaluation, with an explicit receipt, rather
than repeating it. This new prospective experimental gate does not rewrite old
results or relax the existing release requirements. One seed and these exposed
cohorts cannot establish a general mini-Jev release.

## Boundaries and execution

No new GPU is allocated by preparing this protocol. Qualify the updated schedule,
loss separation, evaluator, final-checkpoint behavior and exact launch bundle
locally first. Use the existing isolated runtimes and independent verifiers.
Local tests use small test networks, never the 9B model on the Mac.

The proposed ceiling is **$60 within the remaining existing $195
training/evaluation/recovery allocation**, not a new authorization. Reconcile
provider billing and ownership before launch. A single GPU may cost at most
$4.60/hour and run at most twelve hours, leaving at least $4.80 for storage and
recovery. Reserve 30 minutes for setup, at most 10 hours 15 minutes for training
and intermediate evaluation, one hour for final evaluation, and fifteen minutes
for recovery. Preserve an independent provider stop guard and collect artifacts
incrementally so final results do not depend on the laptop staying online.

Keep the stopped older paired pod and its volume; do not restart or duplicate
that recipe. Do not automatically extend this experiment or rent a replacement
if it fails. A useful result earns a subsequent fresh transfer check; another
failure earns a diagnosis, not an automatic larger run.
