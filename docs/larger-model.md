# A larger First Instinct

This is a new experiment in adapting a pretrained model with billions of
parameters to the described-option interface. The released v0.2.0 model remains
the 141-million-parameter DeBERTa experiment. There is no claim of Jev parity or
knowledge of its private training method.

The newer [software outcome experiment](software-outcome-model.md) adapts the
same foundation to execution-verified event forecasts and includes a local
inspection demo. The general-task mixture described below is a separate pilot;
its adapter was not used to initialize the software model.

See the [architecture roadmap](architecture-roadmap.md) for the distinction
between the running supervised model and proposed outcome/value predictors.

## What changes

The default starting point is `Qwen/Qwen3.5-4B`, pinned to revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The supported full checkpoint includes
an unused vision component. The local adapter experiment loads 4,571,730,432
parameters including 32,464,896 new trainable adapter parameters. The language
model is nominally four billion parameters.

All options enter a single context. Each receives an arbitrary one-token label.
The network processes that context once; we select the final-position logits
for those labels and normalize only across the offered options. There is no
generated explanation or tool execution in this inference path. This differs
from v0.2.0, which runs a shared encoder separately for each option. The new
model can compare option contents jointly, but may become sensitive to their
order. Preparation shuffles labels independently per example; permutation
robustness still needs measurement.

Low-rank adaptation updates small matrices throughout the language network.
Vision weights, original language weights and vocabulary weights stay fixed.
The model sees only state, question and option descriptions. Source identifiers,
targets and execution receipts are excluded by an explicit input allowlist.

Multiple acceptable actions are supported. Training maximizes their combined
probability rather than declaring all but one incorrect. These probabilities
describe relative choices. They are **not calibrated probabilities that an
action will succeed**.

## Data, with provenance

The pilot build contains 13,939 training questions; the larger build contains
153,031, representing 84,503 source records and 44,672 connected source groups.
All fit the 2,048-token limit; the larger set contains 26,716,480 input tokens.
Those are question counts, not independent source counts. Both retain source identifiers, grouping, conversion provenance,
source licenses and file hashes.

| Task | Pilot questions | Larger questions | Feedback |
| --- | ---: | ---: | --- |
| Choose a tool or continue a conversation | 5,000 | 40,000 | Glaive synthetic first actions |
| First tool among described candidates | 239 | 239 | ToolACE synthetic reference calls |
| Sentence relationship, categorical | 1,800 | 30,000 | Public human annotations |
| Sentence relationship, binary | 3,600 | 60,000 | Derived from those same annotations |
| Emotion, categorical | 600 | 4,264 | Single-label public human annotations |
| Emotion, binary | 1,200 | 8,528 | Derived from those same annotations |
| Operation selection | 1,500 | 10,000 | All candidate functions executed |

This mixture is a starting point for a training pilot, **not an assertion that
153,031 examples have passed a human quality review**. It is heavily weighted
toward sentence relationships. Glaive follow-up questions use a question-mark
heuristic; a source response can be suboptimal. Its largest connected component
contains 58,210 candidate records, leaving only 41 validation and 55 test
questions in that family. This limits what a validation improvement can tell us.
The converter quarantines 1,888 unclassified non-call responses, including
declarative requests for information. Its retained non-question responses
express inability explicitly. Further human tool-label auditing must precede
claims about a broadly capable demo.

Public sources and versions are pinned. ToolACE reserves earlier evaluation
and introductory inspection groups. Glaive groups connected function names and
exact normalized requests before splitting. Human datasets preserve official
splits and reserve all final and validation source groups before selecting
training data. Exact source/state separation is checked independently.
Semantic near-duplicate leakage and base-model pretraining contamination remain
possible. Public final sets are regression checks, not newly blind benchmarks.

The new ToolACE split is not a clean evaluation of the previously released small
model: 24 validation rows and 26 test rows occur in that older model's training
IDs. This does not overlap the new larger model's own training partition, but
it invalidates an unqualified small-versus-large comparison on those rows. The
pilot's small-model reference excludes all 17 such rows it could process, leaving
120 questions from six tasks; three other tool questions exceeded its input
limit. The primary comparison for the larger experiment is its own pretrained
checkpoint versus the selected adapter on identical held-out examples.

The execution task in this original general-task mixture runs only twelve known
local arithmetic functions. Incorrect actions, coincidentally
correct alternatives and no-suitable-action cases are retained. Three operations
are held out entirely as a separate challenge. This is a narrow authored
environment, not evidence of general tool execution or repository-level coding.

## Run it

```bash
python -m pip install -r requirements-scale-cuda.txt
python -m scale_lab.data --output output/scale-pilot
python -m scale_lab.prepare --data output/scale-pilot \
  --output output/scale-pilot-qwen35 --model qwen35-4b --max-tokens 2048
python -m scale_lab.train --data output/scale-pilot-qwen35 \
  --output output/scale-run --device cuda --max-steps 100 --max-hours 0.5
python -m scale_lab.infer --run output/scale-run \
  --input examples/weather.json --device cuda
```

These commands target the selected Runpod image and retain its PyTorch 2.13.0
build. For a Mac, install `requirements-scale.txt` and use `--device mps`; the
local check uses PyTorch 2.14.0. Actual library and graphics-runtime versions
are recorded with each run.

Use `scale_lab.data --full` for the larger source quotas. Preparation records
token counts and rejects overlong inputs without silently truncating them.
Dataset and prepared-file hashes are checked before training. A model alias
selects a pinned checkpoint; each model needs its own prepared dataset.

The original pretrained checkpoint is evaluated before any update. Checkpoint
selection uses validation set likelihood; the original remains eligible if
training does not help. Training records validation predictions, learning rate,
gradient norm, example visits, processed input tokens, package versions,
code hashes, time and peak graphics memory. `best` and `latest` contain adapters,
which require the pinned original weights for inference.

For larger batches, `--length-bucket-size 2048` groups nearby input lengths
inside shuffled pools, then shuffles whole batches. Every row is visited once
per epoch, including a possible partial batch; target labels do not determine
the order. The bucket size must be a multiple of the effective batch size.
On the 153,031-question build, 64-example batches process 29,486,235 padded token
positions with this grouping versus 50,611,409 with ordinary random ordering.
Both contain the same 26,716,480 real input tokens. This is a padding count,
not a measured speedup. Runs record real and padded token counts separately.

`scale_lab.bundle` packages an explicit allowlist of code, dependencies and
prepared data for transfer. Credentials and arbitrary workspace contents are
excluded. The bundle's companion manifest lists every file hash.

After checkpoint selection, `scale_lab.order_audit --raw <raw-data-directory>
--data <prepared-directory> --run <run-directory> --output <new-directory>`
checks a deterministic sample from the test split with the options reversed.
It reproduces original tokens before comparing choice agreement, accuracy,
probability changes, and switches between correct and incorrect answers.
Switching between two acceptable actions is counted separately. This is one
order perturbation of the same cases, not an independent new test set or a
guarantee of order invariance.

## Rental and interpretation

The [Runpod walkthrough](runpod.md) records the image, installation fixes and
storage layout used by the first rental.

The first cloud target is one H100 with 80 gigabytes of graphics memory.
Measure actual throughput and memory in a short pilot before scheduling a full
run. The training deadline limits the Python process; **it does not stop cloud
billing**. A separate provider stop must be scheduled and verified, and artifacts
copied before deleting storage. Account keys and rental receipts stay local.

Local verification has exercised the real four-billion-parameter model's
forward pass, two adapter optimizer updates and checkpoint save. That small
smoke test establishes compatibility, not a performance improvement.

The [first cloud pilot](../results/scale-pilot-v1/README.md) subsequently completed
100 updates over 3,200 training questions. The selected checkpoint improved
140-question validation accuracy from 78.57% to 94.29% and log loss from 0.5378
to 0.2000. Selection used log loss; a different checkpoint reached 95% accuracy.
The final test split remains unevaluated. The larger data run is paused while
the project revisits environment-backed data and reinforcement learning.

This new text pipeline is supervised learning. Later reinforcement learning
should compare action/inspection policies under observed outcomes and costs
against a supervised learner receiving the same information. Calibrated success
forecasts need their own outcome targets and evaluation. Neither more parameters
nor reinforcement learning alone establishes calibration.
