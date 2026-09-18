# Prospective shared-prefix mechanics and timing check

This protocol is supplementary to the frozen training run and to the demo.
It uses one authored shipment state and eight independently answerable questions:
dispatch route, restocking, urgency, temperature safety, carton selection,
guaranteed evidence of signed delivery, warranty coverage, and carrier selection.
The questions include choice, binary, and ordered outputs. Their answers follow
executable rules from visible facts. Unknown delivery evidence has no invented
probability; the question asks whether the fact is guaranteed in every possible
completion. The eight questions are correlated observations of one authored
state, not eight independent worlds or a broad capability benchmark.

No real model result has been obtained merely by implementing or testing this
runner. Tiny fixture callbacks in its tests check mechanics only. Nothing in
this protocol selects a training checkpoint or automatically changes the demo.

## Conditions and workload

Use the first 1, 3, and 8 questions in their authored order. The three-question
subset already includes each output type. For each subset compare:

1. The existing options-first layout, with independent complete forwards.
2. State first, with independent complete forwards.
3. The identical state-first token streams, with a shared prefix and deeply
   copied independent cache branches.

Condition 1 versus 2 measures a prompt-order change. Condition 2 versus 3
measures caching with identical inputs. The implementation comes from
`general_lab/shared_prefix.py`; both recurrent/convolution states and attention
states are copied. All conditions use the same resident model in evaluation
mode, with all parameters frozen. No optimizer is created.

Each condition receives one separate eight-question warmup. Then all 27 measured
trials (three subsets, three conditions, three repetitions) execute in a fixed
seed-shuffled, interleaved order. The exact order is frozen before inference.
Finally, the eight cached questions run in reverse order, followed by each
question alone. These check question-order and branching independence against
the eight-question cached result assigned repetition index zero before shuffling.

The concrete upper bound is **148 forwarded questions and 156 model calls**:

| Work | Forwarded questions |
|---|---:|
| Separate warmups | 24 |
| Three measured repetitions of all conditions/subsets | 108 |
| Reversal and each question alone | 16 |

The model-call count also includes prefix forwards. The requested measured
matrix alone exceeds 100 questions, so this explicit bound replaces a
100-question budget. There are no retries, parameter sweeps, additional model
loads, or automatic rentals. Parent orchestration must approve when and where
to launch. Preparation uses only a cached tokenizer and does no model inference.

## Freeze before inference

The `prepare` command refuses to overwrite its output folder. It writes the
authored public fixture and private executable targets, exact legacy and
state-first token streams, per-condition full prompt/prefix/suffix token counts,
the complete execution order and total logical token work, and a freeze receipt.
Every complete prompt must fit the declared maximum of at most 1,536 tokens;
overlong inputs fail rather than truncate.

A cached-tokenizer-only check of this fixture produced these complete-prompt
counts. It performed no language-model inference:

| Question | Existing layout | State first |
|---|---:|---:|
| Dispatch route | 528 | 531 |
| Restock | 525 | 525 |
| Urgency | 535 | 538 |
| Temperature safe | 524 | 524 |
| Carton | 529 | 532 |
| Signed delivery guaranteed | 528 | 528 |
| Warranty coverage | 532 | 535 |
| Carrier | 532 | 535 |

The state-first prefix is 467 tokens for both multi-question subsets. Three
questions require 1,594 independent input tokens or 660 cached forwarded tokens;
eight require 4,248 or 979 respectively. The entire prescribed experiment
forwards 59,363 logical input tokens, including warmups and independence checks.
The freeze records exact streams and independently recomputes these counts;
they are not speed measurements.

The receipt pins every source and this protocol, the fixture and token files,
package versions, the Qwen3.5-9B foundation model revision/specification hash,
the saved run receipt, and every regular file in its selected adapter directory.
The foundation revision is its identity; this runner does not reread and hash
all foundation tensor shards. Adapter files are content hashed. Model loading
is offline from cached artifacts, with no download or credential fallback.

The launch verifies those hashes and package versions, reproduces the token
streams with the loaded tokenizer, then verifies the freeze again before the
first forward. A second results folder prevents rerunning over previous results.
Completed observations survive later failures. The files are reverified after
the experiment, and model parameter version counters must remain unchanged.
Version counters are an inference mutation check, not a second full tensor hash.
Run in a fresh CLI process as shown below. If a caller previously imported the
Hub library with online mode enabled, setting environment flags afterward is
insufficient; the runner detects that cached state and fails before constructing
the model loader.

## Measurements and interpretation

Wall time synchronizes the accelerator before and after each call and includes
cache copying plus prediction setup/validation. Warmups and independence checks
do not enter timing summaries. Report all three measured observations, mean,
median, minimum and maximum per condition/subset; do not infer significance
from three repetitions. No concurrent requests or server throughput are tested.

Report actual forwarded logical input tokens and model calls. Token savings do
not directly measure saved attention work or floating-point operations. On CUDA,
report allocator memory before and after each call and its absolute peak,
including resident weights. On Apple graphics processors, report current and
driver allocations before/after; a peak is unavailable through this runner.
CPU memory peaks are not measured. Do not compare these different memory
statistics as equivalent.

Report all per-option probability changes and answer agreement for prompt order,
same-layout caching, reversed question order, and questions run alone. The
predeclared cache check requires probability differences at most 0.02 and exact
selected-answer agreement for all same-layout comparisons. A stricter positive
tolerance may be frozen before inference; values above 0.02 are rejected. This tolerance is a
mechanics gate, not a calibration guarantee. A failure preserves the report,
exits unsuccessfully, and does not trigger an automatic adjustment or rerun.

Correctness is scored against the authored executable targets. The principal
quality display uses the preassigned repetition-zero eight-question trial of each condition;
other repetitions and nested subsets are not extra independent test examples.
Because trials are shuffled, repetition zero need not run first chronologically.
A prompt-order quality difference on this one fixture is a reason to study the
layout further, not evidence of improved general reasoning. No result here
identifies Jev's private architecture or training recipe.

## Commands

Use a separate output folder and an already downloaded completed 9B run:

```sh
python -m general_lab.prefix_benchmark prepare \
  --folder output/shared-prefix-v1 \
  --adapter-run output/general-supervised-complete-v1/runs/supervised-01 \
  --device mps
```

Review and preserve the freeze before explicitly launching:

```sh
python -m general_lab.prefix_benchmark run --folder output/shared-prefix-v1
```

`cpu`, `mps`, and `cuda` are explicit frozen choices. The launcher uses its own
model instance; do not run it alongside a resident 9B demo on a memory-limited
Mac. Stop the owned demo first and restart it afterward under parent
orchestration. There is no automatic server stop, deployment, or cloud rental.
