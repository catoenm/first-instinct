# Canonical checkpoint identity before supervision

The v1 pilot never reached a model forward pass or optimizer step. Its first
rental failed on a missing image dependency. The infrastructure retry passed all
25 packaged tests and loaded the original foundation, but stopped at the initial
trainable-tensor hash check. Both failure archives were recovered and verified;
both rentals were deleted.

The trainer compared two different hash contracts. The established parent digest
sorts tensors by parameter name before hashing names and float32 payloads. The
new trainer imported a helper that used model registration order. The same
weights can therefore produce different digests. This was an implementation
error, not evidence that the original checkpoint had changed.

The separate `supervised_decision_pilot_v2` entry point uses the established
sorted-name contract, verifies it against the serialized parent before model
loading, and records the loaded digest before checking it. It preserves the
frozen v1 source and all failed artifacts. The objective, sampling, learning rate,
development selection, retention checks, parent weights and question bytes are
unchanged from [the prospective protocol](appworld-supervised-v1-protocol.md).

Local qualification exercises all 496 original adapter tensors, containing
43,278,336 parameters, in reversed registration order. The runtime helper and
independent saved-tensor calculation both produce the original digest. Tests
also require a changed weight to change the digest and frozen weights to be
excluded. This requires no foundation load, forward pass or optimizer update.

A new package must include the correction, regression tests, unchanged data
hashes, both closed failure receipts and the same original model identity. It
must pass extraction, source and adapter checks before a corrected launch.
The Cairo setup stays at its successfully tested version.

This corrected launch shares the original $30 allocation with the two failed
attempts. Hold $2 for their charges, leaving at most $19.60 for compute and $8.40
for storage and recovery. Do not extend the previous absolute shutdown deadline.
This is one corrected execution after a proven local bug fix, not an automatic
loop over paid failures. A new model or optimization failure stops this version
for local diagnosis; failed learning gates still prohibit scaling the recipe.

Actual training consumption remains zero until completed backward and optimizer
receipts exist. A successful startup or a prepared data pool is not a training
result. The general-purpose model and probability-quality claims still require
the original joint improvement and retention checks.
