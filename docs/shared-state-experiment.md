# One state, several independent questions

Status, September 18, 2026: an **offline inference prototype**, separate from
the frozen outcome-training experiment and the running demo. No trained-model
quality or speed improvement has been measured with this prototype.

The existing interface answers a batch of user-defined questions by repeating
the whole state for every question. Its serializer puts the JSON keys in
alphabetical order: options, question, then state. Different questions therefore
diverge before reaching the reusable state.

The new [prototype](../general_lab/shared_prefix.py) puts state first, followed
by one question and its choices. It tokenizes each complete prompt, finds their
exact common token prefix, processes that prefix once, and copies its saved
network state into an independent branch for each question. It never uses a
previous question's answer as another question's input. The cache lasts for one
request; it is not shared between visitors or requests.

This is conventional prefix caching applied to constrained decisions, not a new
architecture or a reconstruction of Jev's implementation. Qwen's hybrid network
requires copying recurrent and convolution state as well as attention keys and
values. A shallow copy can contaminate later branches.
[Qwen implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
[cache documentation](https://huggingface.co/docs/transformers/kv_cache).

## What we checked

The seven offline checks include real, randomly initialized miniature Qwen3.5
networks with both linear and full attention, including the conditional model
wrapper used by our 9B loader. They compare cached branches with complete
uncached forwards, reverse the question order, run each question alone, and
verify that parameters remain identical. Single-token and multi-token branches
are both exercised. The probability-equivalence tolerance in these float32
processor tests is 0.000002.

This tests computation and branch isolation. A random 27-thousand-parameter
text fixture has no useful language ability, and a passing miniature check does
not establish equivalence for the 9B checkpoint or its graphics-processor kernels.

Using only the cached **9B tokenizer**, the three questions in
[`examples/general-decisions.json`](../examples/general-decisions.json) contain
650 input tokens in either layout. Their reusable prefix grows from 42 tokens
to 141 when state comes first. The prototype would forward 368 tokens instead
of 650, avoiding 43.4% of repeated input-token processing for this example.
This is **token accounting, not a measured latency improvement**. Copying caches
costs memory and time, suffix attention still reads the prefix, and ordinary
batching can already improve device utilization.

## Quality is a separate question

Changing key order changes the model's input, even if the written facts stay
the same. The current adapter was trained with the original layout. A proper
comparison therefore needs three conditions:

1. Original layout, independent complete forwards.
2. State-first layout, independent complete forwards.
3. State-first layout, shared prefix with independent cache branches.

The first versus second measures the prompt change. The second versus third
measures caching. Do not attribute a changed answer to caching before separating
these effects. Do not change the active training serializer or deploy this
prototype based on the token count alone.

Before deployment, compare held-out correctness, probability quality, question
order independence, and numerical agreement on the real checkpoint. Measure
warm and cold latency over repeated runs, interleave execution order, include
cache-copy time, and report peak memory. Include one-question requests, very
short states, several question counts, and long states near the input limit.
Only then consider an additional state-first training mixture if the existing
adapter needs it. This evaluation must not change the current experiment's
checkpoint selection.

## Reproduce

The default command loads only an already cached tokenizer; it performs no
model inference or network request:

```bash
python -m general_lab.shared_prefix \
  --input examples/general-decisions.json \
  --output output/shared-prefix-plan.json
python -m unittest test_shared_prefix -v
```

`--inference --run /path/to/run --device cuda` explicitly loads the saved model
and compares all three conditions. It writes distributions and a numerical
cache check. Its one-shot durations include warmup and ordering effects and are
not a speed benchmark. Keep this separate from active training devices and
budget it before execution. The command refuses to overwrite an existing report.
