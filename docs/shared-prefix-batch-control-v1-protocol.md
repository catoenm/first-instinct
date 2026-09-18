# Post-hoc ordinary-batch control for shared-prefix timing

The published shared-prefix experiment measured a 3.47× median speedup over
serial complete forwards for eight questions on one state. It did not compare
ordinary batching. This supplementary experiment closes that performance
control; it is designed **after seeing the original result**. It provides no
new independent quality set and no evidence about Jev's private implementation.

Use the exact bytes of `results/shared-prefix-v1/fixture.json` and
`encoded-inputs.json`, pinned in the new runner along with the original freeze
and summary. Do not author new prompts or retokenize during preparation. The
state-first lengths remain 531, 525, 538, 524, 532, 528, 535 and 535 tokens.
Every full prompt fits 1,536 tokens; reject longer inputs, never truncate.
The existing options-first token streams remain in the copied receipt for
provenance but are not forwarded in this experiment.

## Conditions and fixed work

Use the exact same completed supervised Qwen3.5-9B adapter and foundation
revision as the original experiment, on one model instance in evaluation mode,
with every parameter frozen. The selected adapter files and saved run receipt
must match the published content hashes. Reuse the published package versions
and existing kernels; do not install optional kernels or change frozen sources.

Compare the first 1, 3 and 8 state-first questions with:

1. **Serial complete:** existing unpadded independent complete forwards.
2. **Batched complete:** one ordinary left-padded complete forward, using the
   existing `scale_lab.model.batch` and `score` functions. Padding multiple is
   one, the last position is a real token, and each row has its own option mask.
   Targets are absent from model inputs and no loss is computed.
3. **Serial shared prefix:** the existing shared-prefix code and deep-copied
   independent cache branches, including recurrent and convolution state.

No prompt-order change, batched cache, decoding, training or concurrent requests
are part of this control. All conditions use the identical real token streams.
Padding and attention masks are the ordinary batch machinery being tested.

Each condition gets a separate eight-question warmup in seed-shuffled order.
Require both optimized conditions to agree with serial complete before starting
the measured matrix. Then execute three repetitions of all nine condition/subset
pairs in the exact seed-shuffled interleaving frozen in `plan.json`. Finally run
eight questions in reverse order through both batching and caching, and all
eight individually through the size-one batch path. These checks compare with
the preassigned repetition-zero eight-question trials; repetition zero need
not be first chronologically. Full serial inherently evaluates every question
alone, and the additional singleton batch checks attention-mask equivalence.

| Work | Forwarded questions |
|---|---:|
| Separate warmups | 24 |
| Three measured repetitions of the complete matrix | 108 |
| Two reversals and eight singleton batches | 24 |
| **Maximum total** | **156** |

This requires **123 model calls**, including shared-prefix forwards. Total
submitted real tokens after cache reuse are **63,698**; including batch padding
there are **64,038 input positions**. Eight-question batching submits 4,248 real
tokens plus 56 padding tokens, versus 979 real tokens for serial cache reuse.
The three-question batch submits 1,594 real tokens plus 20 padding tokens.
These are input-position counts, not floating-point operations or speed claims.
There are no retries, sweeps, extra fixtures or automatic replacement runs.

## Equivalence and measurements

Tiny randomly initialized real Qwen3.5 hybrid models test variable sequence
lengths, two/three-option masking, 1/3/8 batches, reversal and individual queries,
for both text-only and conditional-generation wrappers on CPU. They must match
serial complete and cached probabilities within 0.000002 and preserve answers
and all tensor values. These checks load no pretrained weights and do not prove
9B equivalence or quality.

The real experiment freezes an absolute probability tolerance of **0.0001** by
default, with exact chosen-answer agreement. An explicitly selected positive
tolerance may not exceed 0.001. Apply the gate to warmups, every measured
serial/batched and serial/cached pair, reversals, and singleton comparisons.
Warmup failure stops before measured inference. Later failure preserves every
observation and exits unsuccessfully; timings do not support an equivalence-based
performance claim. Never loosen the tolerance after viewing the results.

Synchronize the accelerator before and after every measured invocation. Outer
wall time includes tensor construction, padding, transfer, option scoring,
probability conversion and cache copying, as applicable. Post-return validation,
memory querying and receipt serialization are outside that elapsed interval.
Warmups and independence checks stay out of timing summaries. Preserve all
three observations and report mean, median, minimum and maximum, plus ratios of
medians for serial/batch, serial/cache and batch/cache. Do not select the fastest
trial or pool subsets as independent samples. Three repetitions on one instance
are descriptive; there is no server concurrency or throughput claim.

Report allocated memory before and after each call. CUDA additionally records
absolute allocator peak including resident weights. MPS records current and
driver allocations, with no peak; CPU peak is not measured. These quantities
are not interchangeable. Runtime records package/Python versions, platform,
device, thread count and actual weight/output-head dtypes. The existing loader
uses float32 on MPS/CPU and bfloat16 on CUDA with a float32 output head. The
planned Mac comparison uses float32 MPS as before; another device is a separate
hardware result, not a direct comparison with the published 3.47×.

Retain per-option probabilities/deltas and tiny-fixture accuracy, even if all
conditions make the same wrong answer. Accuracy on the already seen authored
state is a mechanics diagnostic only. Parameter version counters must remain
unchanged; they are not a second full tensor hash. No result deploys a new layout
or cache path automatically.

## Preparation and later explicit launch

Preparation only reads local published files, hashes adapter artifacts and
runtime versions, and writes a new non-overwritable folder. It imports no model
loader or tokenizer and performs no inference. Its freeze pins the exact
published identities, copied input bytes, supplementary plan, old and new
source files, tests, protocol, model revision and selected adapter content.
The foundation is identified by its pinned revision; foundation shards are not
rehash-read. Run verifies the freeze before loading, reproduces state-first
tokens with the loaded tokenizer, verifies again before forwarding, then checks
again afterward. Failed observations remain available. A fresh CLI process is
required so cached offline flags are active; no network fallback is permitted.

```sh
.venv/bin/python -m general_lab.prefix_batch_control prepare \
  --folder /Users/mitchell/Documents/workspace/ml-data/output/shared-prefix-batch-control-v1 \
  --published-folder /Users/mitchell/Documents/workspace/ml-data/results/shared-prefix-v1 \
  --adapter-run /Users/mitchell/Documents/workspace/ml-data/output/general-supervised-complete-v1/runs/supervised-01 \
  --device mps
```

Only after parent orchestration decides to allocate local resources:

```sh
.venv/bin/python -m general_lab.prefix_batch_control run \
  --folder /Users/mitchell/Documents/workspace/ml-data/output/shared-prefix-batch-control-v1
```

The driver does not stop or restore services, deploy a demo or rent a graphics
processor. Parent orchestration owns model/service lifecycle so another 9B load
does not compete with the resident demo. Preparation and tiny tests are not
authorization to launch the real-model command.
