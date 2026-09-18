# Optional foundation control: post-hoc addendum

This control was proposed **after viewing the completed supervised audit**:
all 336 deterministic questions and all 120 finite-forecast modes were correct,
while error against the stated forecast probabilities was about 17.6 percentage
points root mean squared error. The comparison is a post-hoc extension, not a
prespecified component of the original audit. It remains optional until a
separate freeze is prepared and an explicit run is launched. No foundation
predictions have been made as part of preparing this implementation.

Run the untouched `Qwen/Qwen3.5-9B` foundation at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with `Predictor(run=None)` and no
adapter. Use exactly the original frozen 48 roots / 456 questions, supplied
option order, serializer, label-token scoring, floating-point output head,
and evaluator. The scorer keeps deterministic accuracy/log loss/Brier separate
from exact expected finite-forecast log loss/Brier and probability error. It
averages within roots before averaging roots. No new data, prompt changes,
temperature fitting, candidate selection, or outcome-dependent retries.

Every question receives one single-question forward in original corpus order.
There are **456 forwards maximum**, no warmups, shared prefix, response cache,
batching, or generation. Opaque-ID variants retain their original duplicate
prompts; they test serialization/mapping, not a learned semantic capability.
Use the original 1,536-token limit without truncation. Freeze all token streams
and check the original prompt accounting (99,354 tokens, maximum 272,
360 unique encoded prompts). The callback receives public inputs only.

Preparation uses only a cached tokenizer and local artifact reads. Before
inference, the separate freeze records the original corpus and completed
supervised result hashes, observed supervised summary, original source hashes,
this addendum/runner/tests, dependency versions, exact foundation revision and
cached runtime snapshot artifact hashes. Locate that snapshot through its pinned,
cached `config.json`; repository documentation such as README and LICENSE need
not be cached. Independently require tokenizer/configuration, the weight index
and every indexed shard, and hash all files present in that snapshot. Missing
runtime files still stop preparation without a download. The model must load in evaluation mode with no
trainable parameters and no adapter. Verify corpus, source, freeze, dependencies,
cache bytes and every token stream before and after execution; verify parameter
version counters remain unchanged. These checks are provenance and mutation
guards, not a full attestation of resident tensor values.

Choose `cpu` or `mps` explicitly; no automatic device or cloud fallback. The
existing loader uses full precision on both, matching the supervised Mac demo.
Launch in a fresh process. Offline Hub flags, cached-only lookup and a socket
connection/name-resolution guard prohibit network fallback. Missing artifacts,
stale online imports, device failure, malformed output or changed provenance
stop the run. The runner preserves partial responses and failed status, counts
attempted forwards before each call, and refuses to overwrite or resume a run.
A failed/partial attempt must be reported, not silently discarded and rerun.

Persist all option probabilities and timings, model identity, input-token and
forward counts. The control includes loading/preflight time in total wall time;
per-response timing synchronizes the device around the local prediction call.
The supervised result used a resident HTTP server and its own timing boundary,
so this is **not a controlled latency comparison**. No timing-based model claim.
No fresh accelerator rental or paid endpoint. The runner does not stop or
restart the demo; any eventual launch must avoid overlapping two 9B allocations
and must restore the owned demo even if this control fails.

A descriptive difference can show what adaptation changed on these particular
public authored templates. The data were already inspected, the questions are
correlated, and the foundation and adapter have different training histories.
This is no independent benchmark, general-calibration claim, proof of causal
training mechanisms, or reproduction of Jev. It must not select the current
reinforcement-learning checkpoint or guide tuning on these roots.

The old supervised runner/protocol/report remain unchanged. The new analyzer
recomputes with the same frozen `robustness.run` scorer and returns compatible
metrics. The existing `robustness_report.analyze` deliberately does not accept
foundation provenance; a future comparison report must use each run's own
provenance validator and keep this post-hoc label. Plotting the recomputed
metrics can reuse the existing generic plot function.

Preparation only (review/publish the resulting freeze before deciding to run):

```bash
python -m general_lab.robustness_foundation prepare --device mps \
  --folder output/general-robustness-foundation-v1
```

Optional separate fresh process, after review and resource coordination:

```bash
python -m general_lab.robustness_foundation run \
  --folder output/general-robustness-foundation-v1
python -m general_lab.robustness_foundation analyze \
  --folder output/general-robustness-foundation-v1
```
