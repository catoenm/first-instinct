# Continued forecast practice, version 1

[Results](../../docs/forecast-audit.md) · [Protocol](../../docs/forecast-audit-protocol.md) · [Benchmark guide](../../docs/probability-benchmark.md)

Twelve models: three seeds of chosen-path rewards only, early exercises,
continued exercises, and an event-label reference. Sources were frozen at
`eb42b3c72fcdb56f36bc94977d4c42e909ff8bef` before the final run. The last training
step is preselected; validation never selects a checkpoint. First-quarter
weights are a prespecified diagnostic, not another model selected using tests.

Every model folder contains initial, quarter and final weights, applicable
initial and final critics, training counts, validation history and complete
compressed traces. Decompress the `.jsonl.gz` files to verify the hashes in the
original manifests. Reward policies also include first and last interaction
trajectory batches. The exercise schedules use exactly the same extra worlds
and observations, checked by the verifier.

`evaluation/results.json` contains 96 workflow evaluations: twelve models,
two checkpoints and four domains. `evaluation/paired_results.json` contains
24 paired-benchmark evaluations. The language benchmark has 3,072 unique
numeric cases and 6,144 text requests, grouped into related variants. Numeric
models are evaluated once per numeric case and receive no paraphrase score.

`evaluation/benchmark/requests.jsonl.gz` has no hidden labels or target
probabilities. Those are in `answers.jsonl.gz`. The optional Jev runner builds
payloads from only the request text and question. `jev-status.json` explicitly
records that live Jev results have not been collected; offline fixtures are
not provider measurements.

The final per-example numeric prediction arrays are reconstructed rather than
checked into Git. Their original hashes are recorded in
`evaluation/reconstructable-array-manifest.json`, and the originals remain with
the local run. `development.json` preserves the single pilot that guided design.

From the repository root:

```bash
python -m pip install -r requirements-calibration.txt
python -m calibration_lab.forecast_audit_verify
python -m calibration_lab.jev_benchmark
python -m calibration_lab.forecast_audit_evaluate \
  --run results/forecast-audit-v1 --output output/audit-recheck
```

Verification checks artifact hashes and source snapshots, regenerates all
unique interaction and exercise worlds, verifies matched observations and
initial weights, replays stored interaction batches, reconstructs all 96
workflow and 24 paired evaluations, and rebuilds both benchmark files exactly.

Code, synthetic benchmark records and trained checkpoints use the repository's
[MIT license](../../LICENSE). No external private data or credentials are included.
