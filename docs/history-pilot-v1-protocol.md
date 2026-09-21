# Tool histories with general-behavior preservation: prospective pilot

Keep Qwen3.5-9B and the original supervised step-2,742 internal adapter. This is
a supervised recipe revision, not the main release run or online reinforcement
learning. It follows two completed release pilots that improved forecasts but
failed tool advancement, with the latest general accuracy close to its limit.

## Data and evaluation

Use the previously admitted release mixture, independently admitted later-tool
training questions and the 190 independently verified retail history forecasts.
Later-tool targets imitate teachers; retail labels describe executing one named
command and then immediately stopping. Do not confuse action probability,
outcome probability, success under a continuation or dataset judge scores.

Every update schedules 32 general replay questions, 16 first-tool questions,
eight later-tool questions and eight verified questions. Rotate general tasks,
tool server groups and verified family/question-kind pairs. General and tool
questions appear once at most; verified questions twice at most. Cap later-tool
sampling at two questions per normalized underlying request. Limit combined
first/later sampling to three presentations per request. These caps concern this
pilot, not cumulative exposure across earlier training. Preserve source lineage.

Maximum 320 accepted updates; development checks every 80. Keep the exact original
6,072-question development cohort. Add a development-only later-tool cohort from
the original tool-development ownership, using the same audited source parser
and nonredundant rendering. Select by public input hash with seed 20260923,
at most one question per normalized request and 16 questions per server. Reject
conflicting token targets. Require at least 100 questions across ten servers.
This is development, not unopened transfer. Reserved tool and mechanism data
remain untouched. No model score affects cohort construction.

The historical release requirements remain binding: first-tool equal-server
macro accuracy must improve by at least three percentage points, general macro
accuracy may fall at most one point, general log loss may rise at most 0.02,
no declared product slice loses more than three accuracy points, and neither
proper probability score may worsen in any outcome group. Additionally require
later-tool macro accuracy within one point of its matched baseline and log loss
within 0.02. Original weights remain always eligible. Two full checks without
eligible improvement stop the run. Stop numerical or retention failures.

## Update and preservation controls

Use learning rate 0.000005, AdamW weight decay 0.01, gradient clipping at 1,
microbatches of two and evaluation batches of four, with the existing 4,096-token
scorer and rank-16 internal adapters. This changes data, learning rate and
regularization together: it is a practical recipe revision, not a causal ablation.

Before updates, cache the unchanged parent's distributions on the scheduled
general **training** questions only. The supervised target remains the original
dataset annotation. Add 0.5 times the mean full-distribution Kullback–Leibler
divergence from those fixed reference predictions across the update's general
rows. This auxiliary preservation penalty is not outcome ground truth, does not
replace annotations, and consumes no development examples. Record its forward
calls separately from training presentations. Save reference hashes for restart.

For each update, select deterministic public probes from the scheduled rows,
including each pool. Compare their full distributions before and after the step.
Reject and stop if mean divergence exceeds 0.02 or any individual exceeds 0.10,
restoring both weights and optimizer state. Rejected attempts and presentations
remain in the ledger and cannot become eligible checkpoints. This local-step
guard does not replace full development retention checks.

Before sustained training, qualify real GPU short/long-context forward and
backward behavior, matched padding, finite adapter gradients, the auxiliary-loss
gradient and unchanged starting hashes. Save update one and restart in a separate
process; verify weights, optimizer, random state, schedule cursor and reference
cache identity. The parent adapter is
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.

## Bounds and progression

Allocate at most $25 from the existing $217.64 conservative remaining allowance;
this is part of the same original $500. No Mac model runs, new budget, paid-model
API or recurring hosting. Reconcile posted charges and existing holds first.
Rent at most one H200 at no more than $5.40/hour, with a three-hour provider-side
deadline. Maximum quoted GPU compute is $16.20; the rest covers storage and
recovery. Limit setup to 24 minutes and the trainer to 8,280 seconds, with a remote
pipeline deadline that reserves recovery time. Freeze exact inputs, configuration,
code, tests, rates and accounting before renting. Preserve failures and recover
hash-verified artifacts before deleting the pod.

Passing permits a separately reviewed larger continuation; failure does not
trigger an automatic retry or main run. The later identical-start forecast-only,
reward-only and combined real-environment comparison remains a distinct stage.
Report optimizer presentations, unique questions, tokens, source tasks, reference
forwards and rejected work separately. Publish no raw protected examples.
