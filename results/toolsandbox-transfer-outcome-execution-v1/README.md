# Fixed outcome-checkpoint transfer comparison

Execution is complete: all six eligible identities returned all 720 questions,
for 4,320 attempted, received and validated predictions. The
[full report](report.md), [machine-readable results](report.json),
[completion summary](completion-summary.json) and unchanged per-identity
`units/` journals are retained. The report's `partial` status preserves the
two original ineligible training roles; no eligible evaluation failed.
See the [results writeup](../../docs/toolsandbox-transfer-outcome-results.md).

These inputs were prepared and published before any cohort inference. Preparation
loads the cached tokenizer and verifies files; it performs no model forwards.
The [cohort manifest](../toolsandbox-transfer-outcome-cohort-v1/manifest.json)
retains all twelve original selected/latest roles. Ten finalized roles are
eligible and reduce to six exact adapter/config/model identities, each facing
the same 720 questions. The two cancelled hybrid-77 roles remain unavailable.

The runtime is the same pinned MPS float32 inference contract as the completed
[supervised reference](../toolsandbox-transfer-supervised-v1/). The
[execution protocol](../../docs/toolsandbox-transfer-execute-v1-protocol.md)
specifies one fresh process per identity, no retries, no checkpoint selection,
and a 90-minute bound per process. The original model, tokenizer, source and
input hashes are in the freeze. Private filesystem locations are excluded.

- Freeze file SHA-256: `f0036925a79b6ff3ead11d72016963b9ad1429a4c9117874eb18c20a8c45eb6f`.
- Freeze content SHA-256: `f0ba205fc6a7f152c2416fa1269a1d3cdff1fe6cf89ceaf9f32306bfeb509773`.
- Encoded input SHA-256: `a697a8c488ab64952073032f964f6b71df2f4c67cb8c7e6d43dbc122d9b84603`.
- Planned new questions: 4,320 across six identities, with all role aliases retained.

This is diagnostic transfer from two authored training environments to an
unfamiliar ToolSandbox mechanism. The 48 authored contexts are correlated;
720 questions do not constitute 720 independent tasks. Choices inferred from
the forecasts use a declared fixed continuation and are not new executed
adaptive-policy returns. The completed supervised reference remains the
comparison; this diagnostic cannot select or tune a model.
