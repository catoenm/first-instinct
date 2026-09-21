# Capture native probabilities before rounding

This separate local qualification uses the reviewed Laya runtime at
[`573e5b62696ba441230cd6be71d593331b5d23af`](https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af).
It does not load pretrained weights, download a model, rent hardware, train, or
open reserved evaluation scores. The active Qwen comparison stays unchanged.

`release_lab/laya_runtime.py` imports only the hash-checked `common.py` and
`agent.py`. It wraps a supplied, already loaded native Agent. Before each call it
requires evaluation mode, the requested device, complete input information and
the shipped calibration interpreted according to this pinned runtime. It captures
exactly one native forward, verifies token IDs, masks and marker positions, and
retains float32 logits. It reconstructs native temperature-scaled probabilities
before four-decimal rounding and requires exact agreement with the public result.
It does not silently renormalize those public values or perform a second forward.

Choice probabilities, entropy-derived confidence, and the separate act-head
probability remain distinct. Only the option distribution supplies consequence
forecast metrics when the question actually describes consequences. Record shipped
and applied temperatures: this upstream revision clamps temperatures to [0.5, 5],
including the shipped English/typed `choice:11+` value near 0.1006. Do not refit
calibration or substitute a different runtime after seeing evaluation scores.

Qualification has two parts:

- Tests exercise the actual pinned decision head with a tiny random encoder,
  positive unrounded tails whose public probabilities round to zero, menu sizes
  across temperature buckets, altered public outputs, corrupted native tokens,
  nonfinite logits, lost context, mode/device changes and hook cleanup on failure.
- Fifteen synthetic native calls cover three shipped checkpoint configurations at
  menu sizes 2, 5, 10, 12 and 36. Their logits are synthetic, their native runtime
  and calibration are real, and each call must exactly reproduce the public
  rounded probability mapping from one forward. These are software checks, not
  model predictions on task data or newly generated training questions.

The wrapper adds verification and host-copy overhead, so its timing is explicitly
instrumented. It cannot supply a native speed comparison. A future scoring pilot
still needs separately qualified, revision-pinned checkpoint bytes, a local-only
loader, real-model checks on the target device, coverage handling, and a frozen
scoring/latency plan. It may proceed only after recovery and budget reconciliation
of the current rental. This protocol grants no additional compute budget.
