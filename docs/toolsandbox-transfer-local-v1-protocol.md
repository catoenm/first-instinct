# Supervised local reference for the ToolSandbox transfer diagnostic

This is a prospective model evaluation on the already collected and audited
[partial ToolSandbox pilot](toolsandbox-partial-v1-results.md). The dataset's
mechanism and exact labels have been inspected, but no model predictions on
these questions have been made. The [scoring protocol](toolsandbox-transfer-v1-protocol.md)
defines endpoints and limitations. The collection's original split names do
not authorize training: every context here is diagnostic only.

Freeze the exact released supervised Qwen3.5-9B step-2742 checkpoint before
prediction. Use its existing loopback-only demo on port 8766. All 720
non-singleton public questions receive one independent serial request. Fill
144 singleton cost distributions deterministically in the scorer and exclude
them from learned forecast metrics. No tool replay, new model load, training,
paid endpoint, generation, probability cache, retry, or checkpoint search is
part of this evaluation. Future reinforcement checkpoints need their own locked
provenance and must not be selected using these results.

Preparation hashes the complete published corpus, scorer and runner sources,
protocols and tests, runtime package versions, supervised adapter files and
saved run receipt. It records the exact 720-row prediction order and public
input hashes. It verifies the existing tokenizer audit against every exported
question and the cached tokenizer files, without loading a tokenizer or model.
That audit establishes 1,078–1,458 tokens and at most 22 choices. No input is
truncated, subsampled or removed based on its target or prediction.

The runner sends only state, question and option descriptions/identifiers.
Outcome receipts, latent worlds, rational targets, utility oracle values,
root/split metadata and the singleton flag remain outside the prediction input.
The public state intentionally includes the declared prior, tool semantics,
costs and fixed continuation. This measures reasoning about a supplied finite
mechanism, not discovery of unknown real-world prior frequencies.

Verify the reported server model/revision, supervised kind and step, device
and 1,536-token/36-option limits before forwarding. Check model metadata on
every response; retain received probabilities immediately and validate each
distribution before sending another question. Write and synchronize an attempt
ledger before every request, distinguishing attempts, received responses and
validated predictions. A failed request stops the run and preserves partial
predictions. Each request has a 60-second timeout. The 5,400-second wall bound
is checked before every request, so an in-flight request may extend beyond it. A failed run is not
silently retried or represented as a complete evaluation.

The server's anti-forgery token remains in memory. Disable HTTP proxy discovery
and reject redirects. After all 720 responses, verify server identity/session,
the unchanged freeze-file hash and source/data/checkpoint hashes again, then
score and retain complete results.
Disk hashes plus service metadata are provenance checks, not an attestation of
the in-memory tensors. The experiment neither changes the server nor promotes
a checkpoint.

```sh
.venv/bin/python -m general_lab.toolsandbox_transfer_local prepare \
  --folder output/toolsandbox-transfer-supervised-v1 \
  --corpus-folder results/toolsandbox-partial-v1 \
  --adapter-run output/general-supervised-complete-v1/runs/supervised-01

# Publish the freeze before this separate command.
.venv/bin/python -m general_lab.toolsandbox_transfer_local run \
  --folder output/toolsandbox-transfer-supervised-v1
```
