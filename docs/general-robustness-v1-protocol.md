# Frozen local supervised-model diagnostic

Prepared September 18, 2026, before any model prediction on this corpus.

Evaluate the already selected Qwen3.5-9B supervised checkpoint at step 2,742
using the existing local demo on port 8766. The initial general training and
four earlier reinforcement runs are complete. The separate outcome-v2 training
comparison is running and **does not use this diagnostic for selection**.

The corpus has 48 root configurations in four authored families and 456
questions, including evidence-change pairs and semantic variants. The
[audit guide](general-robustness.md) describes target semantics, transformations,
scoring and limits. The corpus, executable verifiers, serving serialization,
runner and local checkpoint artifacts are checksummed before inference in
[`results/general-robustness-v1/freeze.json`](../results/general-robustness-v1/freeze.json).

Run every question once in deterministic corpus order. Send one serial request
at a time to the resident model, with no response caching, retries based on
prediction quality, prompt repairs or temperature fitting. Only public state,
question and offered descriptions reach the server. Use the choice transport
for every input to preserve the exact offered identifiers and order; the
existing native binary and ordinal wrappers serialize to the same model input.
This measures those question meanings, not the convenience wrapper's returned
ordinal expectation field.

Report all families and variants, probability drift after semantic remapping,
and whether both members of an evidence-change pair are answered correctly.
Keep deterministic accuracy separate from finite-experiment modal accuracy and
distribution error. Log scores clip at 1e-12, as recorded in the report. Average
within roots and then across roots. Preserve all raw predictions, the execution
receipt, source hashes and unchanged-corpus verification.

This small diagnostic supplies no foundation-versus-adapter causal comparison,
no test of Jev, and no evidence of general calibration. It is public authored
material, not an independent third-party benchmark. The 456 questions are
correlated views of 48 configurations from four shared templates. Do not treat
them as 456 independent tasks. All facts and finite-randomness assumptions
needed to answer are explicitly supplied.

The server reports its model/revision, selected step and device. Local model
artifacts are hashed separately; the server does not attest to the tensors
resident in memory. Verify status before/after execution and on every response;
fail if the server changes or restarts. The anti-forgery token never enters
published artifacts. Stop and preserve partial predictions on any error.

No new GPU rental or paid API is used. Existing cloud training remains
untouched. A future outcome-trained comparison may use the same frozen corpus
only after its own checkpoint selection is complete, and must be identified
as a supplementary comparison, not a retrospective selector.

```bash
python -m general_lab.robustness_local \
  --corpus results/general-robustness-v1/corpus.json \
  --freeze results/general-robustness-v1/freeze.json \
  --adapter-run output/general-supervised-complete-v1/runs/supervised-01 \
  --output output/general-robustness-v1-supervised
```
