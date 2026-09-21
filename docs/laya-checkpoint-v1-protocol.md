# Pin checkpoint bytes before the scoring pilot

The primary checkpoint remains `convaiinnovations/laya-typed-decisions` at revision
`f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`. The public repository metadata advertises
one 842,609,220-byte `model.safetensors`, with SHA-256
`4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.
The four previously reviewed configuration/tokenizer files also match the Git blob
identities at that revision. This is a metadata check: the weight bytes have not
been downloaded or locally verified by this stage.

`release_lab/laya_checkpoint.py` binds these five files to an exact allowlist. Its
loader requires verified local files, offline libraries, the reviewed native
runtime, and an explicitly indexed CUDA device. It rejects modified metadata as
well as modified weights, extra files, remote-code configuration, and device
fallback. The native tokenizer's configuration normalization happens in a private
temporary directory; the original files remain unchanged. Loading on macOS is
explicitly refused while local foundation inference is paused.

Three CPU tests check tampering, missing/extra files, path escape, and early
rejection of Mac or online loading. They do not execute a real checkpoint load.
The guarded remote smoke test must still exercise loading, inference, source and
weight identity, complete-input handling, and native probability parity before
any task-quality comparison can count as qualified.

This plan neither starts a rental nor increases the original authorization.
Recover the current learning experiment and reconcile its costs first. The next
separately frozen scoring stage should use the already compatible 308 exposed
development forecasts, preserve native calibration, and compare complete-input
probability quality on the same questions. It must not open the reserved release
evaluation, select checkpoints by external evaluation scores, or treat this
development baseline as unfamiliar-task transfer.
