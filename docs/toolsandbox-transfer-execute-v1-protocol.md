# Local execution of the fixed transfer cohort

This separate executor implements the unchanged
[cohort plan](toolsandbox-transfer-cohort-v1-protocol.md) and
[scoring protocol](toolsandbox-transfer-v1-protocol.md). Implementing or preparing
it does not mean an evaluation has run. It performs no training, checkpoint
selection, service lifecycle action, provider operation, or retry.

Preparation requires the original archive, completed recovery receipts, saved
cohort manifest, and completed supervised reference. It rebuilds the cohort from
the archive-bound provenance, verifies the original twelve roles and exact byte
identity groups, and requires the read-only reference analyzer to reproduce the
supervised predictions and metrics. All missing or failed roles remain in the
freeze. Preparation hashes the cached pinned foundation snapshot and every weight
shard without loading weights. It loads only its cached tokenizer, reproduces all
720 public question inputs and token streams, and pins label IDs, tokenizer files,
package versions, source files, adapter bytes, role mappings, and the input plan.
No downloaded cache completion or automatic device fallback is provided.

The execution freeze and encoded inputs are written to a new folder outside the
immutable inputs. Operational filesystem paths live only in the ignored private
`.runtime-locator.json`, whose file hash is in the publishable freeze. Do not
publish this locator or the private collector receipt. No provider ID, connection
information, credentials, or private receipt contents enter the execution freeze.
Each subsequent fresh CLI process runs exactly one full adapter
identity, in the frozen plan order. A process lock prevents concurrent units.
Every predecessor must have unchanged complete artifacts. An existing unit
directory—even a failed or empty one—prevents resume or requery. No failed unit
is silently replaced or excluded.

All inference uses MPS float32, SDPA, evaluation mode, disabled KV cache, the
frozen `scale_lab.model.load_model` loader, and the unchanged
`Predictor.predict`/`evaluate(batch_size=1)` methods. A fresh Predictor instance
is initialized with the chosen original `best` or `latest` adapter path instead
of invoking its constructor's best-only path convention; no Predictor source is
changed. The adapter must be a regular original role directory under the verified
recovery root, with exact config and weight hashes. Runtime, source, cache,
checkpoint, token streams, loaded model configuration, and parameter version
counters are checked before and after inference. No hidden outcome, original
training receipt, role name, or rational target enters a prediction input.

For each question, append and fsync an attempt before forwarding. Persist the
received response before validating it, and persist validated probabilities
before proceeding. Count attempted, received, and validated questions separately.
A failed forward or malformed response leaves all earlier receipts intact, marks
the unit failed, and stops. There are exactly 720 allowed forwards, zero warmups,
no retries, no batching beyond one question, and no input/probability cache.
Model durations are measured with MPS synchronization. A 90-minute alarm and
wall-time checks start at unit entry and include initial provenance verification,
loading, prediction, and final verification; a Python
signal can be delayed while native code is running. The external lifecycle owner
can enforce a process termination deadline; this module does not control services.

An exact byte-identity match to the supervised adapter may reuse only the
independently verified completed reference under the same frozen inference
runtime, including verified MPS availability. Reuse performs zero forwards and is explicitly recorded separately from
new inference. It still verifies provenance and scores the complete saved stream.
All original role aliases remain attached to the identity. No approximate tensor
comparison or score-dependent reuse is allowed.

This is not a controlled latency comparison. The supervised reference used a
resident HTTP service and its original timing boundary. New identities use fresh
processes and outer MPS synchronization. Reused response milliseconds retain
those historical service measurements; `model_seconds=0` means no **new** forward
work, not a zero-latency model. The historical total is recorded separately.

HF offline flags, cached-only tokenizer/snapshot access, and a Python socket guard
apply throughout preparation and execution. An actively blocked probe verifies
the guard. Additional attempted network access fails the unit. This is Python
socket interception, not an OS sandbox guarantee. The runner imports no provider
or service client. Future execution requires the reviewed runner and matching
available runtime, with no new user-permission step introduced by this protocol.

The unchanged scorer recomputes metrics from all 720 probability maps. A durable
completion receipt hashes every output file. Partial results are retained and
never represented as a complete evaluation. Disk provenance and version counters
are not an attestation of every resident tensor. The original authored-mechanism,
prior, fixed-continuation, and generalization limitations still apply.

```sh
.venv/bin/python -m general_lab.toolsandbox_transfer_execute prepare \
  --cohort output/toolsandbox-transfer-outcome-cohort-v1 \
  --root output/outcome-cloud-v2 --archive output/outcome-cloud-artifacts-v2.tar.gz \
  --recovery-receipt .local/runpod-receipt-POD.json --expected-pod-id POD \
  --transfer-folder results/toolsandbox-transfer-supervised-v1 \
  --sft-results output/toolsandbox-transfer-supervised-v1/results \
  --output output/toolsandbox-transfer-outcome-execution-v1

# A separate fresh process, after the external lifecycle owner supplies resources:
.venv/bin/python -m general_lab.toolsandbox_transfer_execute run \
  --folder output/toolsandbox-transfer-outcome-execution-v1 --identity FULL_SHA256
```
