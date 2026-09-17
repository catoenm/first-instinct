# Recorded experiment: several kinds of decisions

Original run: `20260917T041711.272063Z` (UTC). These are the measured outputs,
with model weights kept out of Git history.

- [summary.json](summary.json): per-task results, all seeds, question variants, and source-level uncertainty estimates.
- [comparison.json](comparison.json): original compact metrics for all 21 model/wording evaluations.
- [protocol.json](protocol.json) and [selection.json](selection.json): training rules and checkpoint choice made before final test predictions.
- [verification.json](verification.json): data and code hashes, exact training membership, checkpoint integrity, metric reconstruction, and reload checks.
- `full-seed-*/` and `frozen-seed-*/`: original manifests, histories, selected validation predictions, every update, and every test prediction.
- `initial-v0.1.0/`: the original released model evaluated on this new benchmark.
- `dataset/`: the original corrected manifest and all partition identifiers. Reconstruct source text from the pinned downloads.
- `code_snapshot/`: the exact training/data code and dependency files. `summarize_multitask.py` is the subsequent analysis source used for the reported intervals.
- [initial_encoder_sha256.json](initial_encoder_sha256.json): hashes of the shared v0.1.0 starting encoder used by the frozen comparisons.

The report directories omit encoder/tokenizer binaries and scorer weights,
so they cannot alone be loaded as checkpoints. Original artifact manifests are
preserved unchanged; files named by those manifests are present in the full
local runs. The selected full checkpoint is in
[release v0.2.0](https://github.com/catoenm/first-instinct/releases/tag/v0.2.0).
Frozen manifests refer to the shared `../initial_encoder` directory in a
reproduced run.

The published source data are human annotations and synthetic references;
reference agreement is not an independently audited measure of real-world
success. Adapted source data retain their upstream terms. See
[third-party notices](../../THIRD_PARTY_NOTICES.md) and the
[complete experiment report](../../docs/multitask-experiment.md).
