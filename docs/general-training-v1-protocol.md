# General decisions: training protocol

Status: data and code prepared; measured model results are reported separately.
This protocol was fixed before examining the new model's held-out predictions.

## Claim being tested

Can one language backbone answer new user-defined questions and choices across
domains, and can outcome-grounded reinforcement learning improve its decisions
without making its forecasts or general language judgments worse?

The foundation is Qwen3.5-9B at the revision pinned in `scale_lab/common.py`.
One forward pass reads probabilities over supplied choice labels. Low-rank
updates train language attention, linear-attention, and feed-forward projections.
Original foundation matrices remain frozen; effective internal transformations
change. This is standard backbone adaptation, not a separate classifier reading
generated text. It is not pretraining a nine-billion-parameter model from scratch.

## Data fixed before training

The raw mixture contains 350,935 training rows from 256,906 document/world groups:

| Component | Training rows | Label source |
| --- | ---: | --- |
| Public instructions | 170,935 | Original annotations across 123 tasks and 40 source components |
| Executable worlds | 160,000 | Recomputed answers across 16 families, two questions per world |
| Evidence-and-action environments | 20,000 | Exact-planner actions or sampled event outcomes |

The public source audit is described in [general-data-sources.md](general-data-sources.md).
Two entire public skills and four generated composition families are reserved.
Known source ancestry is grouped before partitioning. Generated worlds have
content-based split ownership. The environment varies event prevalence, evidence
quality, evidence costs, and asymmetric error costs; challenge parameters and
domain wording are reserved. Its eight domain wordings are variations of one
environment mechanism, not eight independent kinds of reasoning.

After rejecting overlength examples without truncating them, the prepared data
has 350,857 training rows and 112,309,610 input tokens. Validation has 14,157
rows; test 17,277; challenge 7,048. Eighty-three rows across all splits exceed
the 1,536-token limit. Answer choices are shuffled per row before tokenization.
The final local raw/prepared directories are `output/general-mixture-v3` and
`output/general-qwen35-9b-v2`; preceding local drafts are unused.

The prepared manifest SHA-256 is
`59cf10b64d67ad81ea1ad7f61a7e0e006432563b7a42e59be1678b8eb7ef0c42`.
It records hashes for raw inputs, upstream attribution, prepared splits and
generator code. A separate prose probe set is evaluation-only, authored and
independently reviewed by agents before seeing model outputs. It does not count
as programmatically verified ground truth.

## Supervised run

Run a short mechanical validation first, then start the full run from the
untouched foundation. Use rank-16 low-rank adaptation, AdamW, a peak learning
rate of 0.00008, three-percent warmup and cosine decay to ten percent of peak.
The primary run visits the whole mixture once, with a declared maximum of
100,000 updates and six-hour execution deadline. Four data-parallel workers
initially use 16 examples per device and two accumulation steps: effective
batch 128. A smaller microbatch may be used if measured memory requires it;
preserve the effective batch and record the actual command.

The hardware check retained gradient checkpointing: at the longest training
inputs, a 16-row forward/backward used 38.8 GB and a 32-row batch used 58.5 GB.
Disabling checkpointing exhausted device memory. The full run keeps the
original 16-row microbatch and effective batch of 128. Inputs are left-padded
to multiples of 64 to limit compiled shape variants; a training-only numerical
check compares single and mixed batches before the full run. This runtime
choice does not use held-out predictions. The pilot is discarded and the full
run starts from the foundation with seed 41.

The first BF16 scoring check exceeded the predeclared 0.02 absolute-probability
tolerance (0.02089, with all choices unchanged). The vocabulary projection now
uses its existing frozen weights in float32; converting logits after BF16
rounding was insufficient. The diagnostic maximum fell to 0.01032. The language
trunk remains BF16 on CUDA, and the same scoring precision is used for the
untouched baseline, training, reinforcement learning and evaluation. This is a
numerical change, not a newly trained output classifier. The original failed
check remains part of the experiment record.

Length bucketing changes padding, not row weights. Tail copies have zero loss
weight; every real row contributes once. Save initial, selected and latest
adapters. Select the lowest mean per-task validation log loss, on up to twelve
seeded examples per task, including the untouched foundation as step zero.
Checkpoints are evaluated every 500 updates and at completion. Do not select
from held-out test, challenge, or prose probes. Report actual visits and tokens
if a time bound ends the run early.

## Reinforcement-learning comparison

Start from the selected supervised adapter. The same language network answers
action questions and independent binary outcome questions. Sample actions,
observe rewards, and update its language adapters with clipped Proximal Policy
Optimization. A training-only value head reads final language features.
Buying evidence changes the next visible state and costs reward; hidden outcomes
and unrevealed sensor readings never enter model inputs.

Compare reward-only training with the same method plus a proper outcome log
loss. Forecast labels are observed events, not model-written confidence or an
oracle posterior. Use paired seeds, the same initial checkpoint and the same
held-out worlds. Retain a modest shared supervised replay term in both arms;
“reward-only” refers to the absence of an explicit forecast loss, not the
absence of replay or the critic. Record every actual setting and completed
episode/update count. Audit a pure-policy gradient separately from auxiliary
gradients, and record changed language parameters.

Selection uses exact expected validation reward. Report both the selected and
latest checkpoints, including honest step-zero selections. Held-out evaluation
integrates the finite action/evidence tree and compares an exact planner, a
no-inspection oracle, and the supervised starting model. Independently measure
forecast error against observed outcomes and the exact Bayesian posterior on
common states, new domain wording, and shifted sensor/cost conditions.

The method uses known techniques. It is our transparent experiment in learning
decisions and calibrated forecasts, not a recovered implementation of Jev's
private Reinforcement Learning for Calibrated Decisions recipe.

## Compute and interpretation

The user authorized up to $500 for this phase. The initial rental is four H200
GPUs at $18.36 per hour total, with an independent eight-hour provider stop
deadline on both the host and the rental. Its maximum scheduled GPU charge is
$146.88; storage and any later experiments remain within the overall ceiling.
No automatic balance top-up is configured by this project. Setup, smoke runs,
failed attempts and evaluation all count toward spending.

Save and verify artifacts, then stop and delete the rental. Training process
deadlines alone do not stop billing. A smaller model, additional epochs or
continued pretraining require a measured reason; they are not assumed to help.

Public data may overlap foundation pretraining. Several rows share a world.
Use paired world-group comparisons, state limitations, and distinguish accepting
arbitrary questions from performing well on them. Constrained outputs guarantee
the output format, not correctness or calibration on every new question.
