# Bounded local serving qualification

This stage may load the exported original-model package only after the complete
6,072-question conversion regression passes. It does not promote either trained
continuation. Keep the existing port-8766 demo until this check finishes.

Reuse the existing typed question serialization, native option mapping and HTTP
handler. Move the PyTorch-only command-line import out of the shared interface's
module scope, without changing any request or answer semantics. No network model
download is allowed: tokenizer and weights must load from the hashed package.
Require its exact pinned MLX/runtime versions and passing regression receipt.

The Mac backend accepts at most four questions, 36 options per question, 4,096
tokens per complete prompt and 8,192 total prompt tokens per request. Validate
every question before model work; refuse excess rather than truncate. Reuse the
one-request inference lock, localhost bind, host/origin validation, 256-KiB body
limit and request token. Retain no submitted state. Limit MLX's free allocation
cache to 256 MiB. This cache setting is not a hard process-memory limit.

Before promotion, test the real API in an isolated process on another port:
all three question kinds, a changed input that changes the expected answer,
repeatability, independently framed questions, rejected excess requests, and
complete maximum-length input. Compare outputs against direct package inference
and record process resident memory as well as MLX active allocation. Require no
probability drift beyond 0.00001 relative to direct inference on identical token
IDs, zero model work for rejected requests, and peak active inference allocation
at most 10 GiB. Run the existing HTTP tests and the new budget checks.

Use local hardware only, no training or reserved evaluation data. An HTTP smoke
pass qualifies this localhost developer demo, not public hosting, arbitrary-task
calibration, or performance on the future 16-GB Mac mini. Keep package files
immutable and store this serving receipt separately.
