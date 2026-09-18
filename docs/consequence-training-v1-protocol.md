# Executed consequence learning on the 9B model

This prospective pilot trains the internal language adapters of the released
Qwen3.5-9B supervised checkpoint on exact, execution-derived outcome probabilities.
It tests whether learning immediate command consequences improves forecasts and
subsequent decisions. This stage is supervised learning, not Proximal Policy
Optimization and not evidence about Jev's unpublished training procedure.

The frozen `puffer_lab/consequence_train.py` configuration is authoritative.
The foundation and tokenizer revision are
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
The starting adapter file hash must be
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.
The previously published source, evidence, and live demo checkpoint stay intact.

## Data and limits of the comparison

The existing 3,456 verified consequence rows describe one reservation mechanism,
32 public histories, 288 history/action groups and 1,152 event questions in three
presentations. Probabilities come from actual SQLite branch states weighted by
the disclosed prior, not model judgments. This is not 3,456 independent worlds.

Hash public history identifiers with the fixed `consequence-pilot-109` salt.
The first eight histories form optimization validation; the other 24 form
training. All events, actions and presentations of a history stay together.
This gives 2,592 training and 864 validation rows. Root worlds and mechanism are
shared, and the development corpus has already been inspected. Therefore this
split cannot establish independent task or mechanism generalization.

Retain 16 deterministic reservoir samples per task from the old general training
split for replay and four per task from its validation split for retention.
No general test/challenge rows are used. Freeze row identities, input token IDs,
model, adapter, source files, settings and output data hashes before inference.
The public freeze contains hashes; it does not require redistributing the entire
old general training set. Targets and private branch states never enter prompts.

## Optimization and selection

One seed (109), one 80 GB H100, at most 180 AdamW steps. Each step contains 24
consequence examples and eight general replay examples, with microbatches of four.
The objective is 75% categorical soft-target cross entropy and 25% the original
acceptable-answer-set log loss. The latter does not force equal probabilities
across multiple acceptable answers. Mask padding without multiplying zero by
negative infinity. Reorder soft targets by opaque option identity.

Learning rate peaks at 0.00002 after a six-step warmup and follows cosine decay;
weight decay 0.01, gradient norm clipped to one. Train the existing rank-16
adapters throughout language attention and feed-forward layers. The foundation
matrices and vocabulary projection remain frozen. Record nonzero gradient names
and initial, first-update, final and selected trainable-parameter hashes.

Evaluate every 30 steps, on the same hardware and padding configuration as the
starting baseline. A checkpoint is eligible only if retained general macro
accuracy falls by no more than two percentage points and macro log loss rises
by no more than 0.05. Select the lowest mean per-event validation excess log loss,
requiring improvement of at least 0.0001. Starting step zero remains eligible.
Stop after three consecutive non-improving eligible-selection checks, nonfinite
loss/gradient, budget exhaustion or mechanical failure. Preserve failed evidence.

## Diagnostics after selection

Run the original 72 decision episodes and 78 fixed-continuation forecasts before
training and on the selected checkpoint, using the same CUDA runtime. Their
previous local results were already opened; they remain development diagnostics.
Additionally, freeze 36 decision episodes from two new disclosed nonuniform
priors, fee schedules and horizons (five and seven turns), all six existing
worlds and all three presentations. These test new parameter combinations in the
same mechanism, not a new mechanism. These scores do not choose checkpoints,
learning rates, stopping times or follow-up arms within this pilot.

Every policy action is actually executed by the C environment previously checked
against SQLite. Record public prompt, tokens, probabilities, action, resulting
state and reward. Greedy evaluation trajectories are not on-policy training data.
Do not claim reinforcement learning has occurred in this pilot.

## Compute and recovery

One personal Runpod allocation. Maximum quoted GPU rate $4/hour, six-hour provider
stop deadline, at most $24 GPU compute plus a small storage allowance, capped at
$30 additional rental cost for this pilot. Prior cumulative compute estimate is
$97.48, excluding storage, against the user's existing $500 authorization.
No automatic replacement rental or external model API spending.

Use the existing pinned CUDA image, Torch 2.13.0, Transformers 5.17.0, PEFT 0.21.0,
flash-linear-attention 0.5.2 and causal-conv1d 1.7.0. Authenticate and install an
independent provider stop guard before downloading or training. Training has a
three-hour optimization limit and four-hour total process limit. An external
launcher enforces the total process deadline, leaving recovery time before the
provider deadline. Stop early after checksum-verified artifact recovery; only
then delete this pilot's rental. The running local demo is not promoted by this
experiment. Publish actual results, including failure or selection of step zero.
