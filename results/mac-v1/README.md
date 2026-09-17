# Recorded experiment: mac-v1

Original comparison identifier: `20260917T033326.634462Z` (UTC).
The directory contains the measured run's evidence, not regenerated metrics.

- [comparison.json](comparison.json): headline results and limitations.
- [full/test_results.json](full/test_results.json): every full-model test prediction.
- [frozen/test_results.json](frozen/test_results.json): every frozen-baseline test prediction.
- `full/` and `frozen/`: manifests, validation histories, selected validation predictions, and all training updates.
- `dataset-v1/` and `dataset-v2/`: original manifests and partition member identifiers. Rebuild the actual source-derived rows using the README commands.
- `code_snapshot/`: exact source used for the measured training comparison.
- `dataset_code_snapshot/`: source corresponding to the corrected dataset manifest's recorded code hashes.
- [verification.json](verification.json): checks completed on the original experiment.

Weights are in the [GitHub release](https://github.com/catoenm/first-instinct/releases/tag/v0.1.0),
not Git history. The full checkpoint contains every file referenced by the full
model's `artifacts_sha256` manifest. The Git copy of its report omits binary
weights and tokenizer files. Raw dataset files are reconstructed from the pinned
source, except for the separate 100-row introductory sample bundled under `data/`.

The original manifests are preserved byte for byte. Later public-facing code
adds the legacy reproduction flag, clarifies inherited group-statistic names,
fixes raw-directory creation for a fresh clone, and defaults training to the
corrected dataset. These changes leave reproduced split bytes unchanged. Thus
current code hashes and newly written manifest hashes can differ from the
historical originals; the split hashes are the data-identity check.

The v2 manifest's original `groups` and `largest_group` fields describe its v1
source build. Its counts and revision exclusions describe the corrected build.
The original `calibration.jsonl` name and protocol language are historical: this
partition holds the old inspected test and was unused by the corrected run.
It is not an untouched calibration set.

[Read the experiment report](../../docs/experiment.md) for the question-wording
correction, split rotation, and remaining limitations. The original run passed
ten offline tests; release-download checks were added afterward.
