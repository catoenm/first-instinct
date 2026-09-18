# Fixed outcome-v2 transfer cohort preparation

This adds a metadata-only preparation step to the unchanged
[ToolSandbox scoring protocol](toolsandbox-transfer-v1-protocol.md). It neither
loads a model nor launches or promises another evaluation. All
three training arms (`outcome`, `reward`, `hybrid`), seeds 77/83, and both
`best`/`latest` roles remain in the manifest: twelve rows in that order. The
original validation-only selection is locked before this diagnostic. ToolSandbox
scores never select, exclude, reorder, or replace a checkpoint.

Require the recovered tree, its original archive, and the collector's completed
private receipt. Verify the archive's SHA256 and byte count against the exact
public collection receipt; compare the archived hash-manifest bytes to the local
manifest; then rehash every recovered file. Reject extra, missing, linked, or
special files. Confirm the expected pod through the private receipt without
copying provider fields or credentials into the output. Only a receipt hash and
pod-ID hash are retained. Require a terminal pipeline and recorded stopped
workers. A terminated archive may contain stale `training` run receipts; those
roles are visible but unfinalized and ineligible, not claimed to be still live.

The H100 training freeze is pinned to
`22d68712a79054b7f87227de7383d8ec3954d4eb649612f79823ae4c2b364daa`.
Verify every archived frozen training source, protocol, and input manifest plus
all code hashes in each completed training receipt. The original archive omits
the top-level frozen `README.md`; record that nonruntime omission explicitly.
No runtime source or protocol omission receives this exception. Foundation
weights were deliberately not archived: pin the model specification/revision
and its digest, without inventing foundation tensor hashes.

Keep original run and retention receipts and their hashes. Reproduce the
training selector from baseline validation and logged validation/retention
events, including update zero and the strict improvement rule. Check committed
optimizer-step counts. Pin role-specific update/step numbers and partial-update
metadata. Compare each role's final retention receipt to its original recorded
selection/latest event; stale or missing retention prevents eligibility.
Completed latest roles that fail the retention gate are still included—the gate
selected `best`, and does not authorize hiding an unfavorable latest result.
Failed, bounded, missing, and unfinalized runs remain explicit ineligible rows.
This verifies recorded provenance; it cannot independently attest which tensors
were resident when an original retention measurement was made.

Bind the unchanged published 720-question transfer freeze and exact public
question mapping. Byte identity is the joint SHA256 identity of
`adapter_model.safetensors`, `adapter_config.json`, and the pinned model
specification. Only that identity may combine prospective inference work.
Preserve every role in an identity group. `selected_update=0` alone does not
establish byte identity with the supervised adapter; semantically equivalent
but differently serialized configuration also does not qualify.

The plan assigns 720 prospective questions to each unique eligible identity.
If it exactly matches the supervised reference, record that fact. Reuse of the
reference's saved prediction role additionally requires its complete 720-response
receipt, unchanged checkpoint/transfer provenance, ordered attempts and responses,
and hashes of the complete saved artifacts. The existing read-only reference
analyzer validates every probability map and recomputes all scores and report
prose; its source hash is pinned by this preparation. A present completed but
invalid reference is rejected. An absent or unfinished reference does not count
as completed inference. No metric value affects grouping or role eligibility.

The plan locks the reference's MPS/float32 runtime, package versions, tokenizer
hashes, frozen serializer/predictor source, SDPA attention, disabled cache, and
serial single-question inference path. A future reviewed executor must verify
that exact available runtime before execution or reference reuse; a mismatch
requires stopping, with no silent CUDA, dtype, batching, or prompt substitution.
Zero planned new questions for an identical completed SFT identity is conditional
on enforcing this locked runtime contract. The plan performs no inference and
does not introduce a new user-permission requirement. The existing supervised
evaluation is not modified or interrupted by preparation.

Write output outside the immutable recovered tree. Example for use only after
checksum recovery has finished (substitute the known receipt and pod identity):

```sh
python3 -m general_lab.toolsandbox_transfer_cohort \
  --root output/outcome-cloud-v2 \
  --archive output/outcome-cloud-artifacts-v2.tar.gz \
  --recovery-receipt .local/runpod-receipt-POD.json --expected-pod-id POD \
  --transfer-folder results/toolsandbox-transfer-supervised-v1 \
  --sft-results output/toolsandbox-transfer-supervised-v1/results \
  --output output/toolsandbox-transfer-outcome-cohort-v1
```

The module emits only `manifest.json` and `execution-plan.json`, with
`model_inference=false`, `model_inference_launched=false`, and
`prepared_not_launched`. No service, GPU, tokenizer,
training tool, provider API, or network client is imported.
