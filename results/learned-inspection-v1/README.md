# Learned inspection, version 1

[Report](../../docs/learned-inspection.md) · [Frozen protocol](../../docs/learned-inspection-protocol.md)

Twelve numeric models, nine value networks, three training seeds, and four final
domains. The source was frozen at
`8b4b82282df37bb4f81161d5c1feec74039a5d4b` before training. `sealed.json` records
the checkpoint hashes and that the final test had not been opened at sealing.
It describes that historical point, not the current evaluation status.

Each model directory contains initial and selected weights, validation history,
a training manifest, and the complete compressed rollout trace. Reward learners
also include initial and selected value networks and the first and last sampled
trajectory batches. `trace.jsonl.gz` decompresses to the hash in its manifest.

`evaluation/results.json` contains all 60 model/objective/domain evaluations.
`summary.json` groups them by recipe, objective and domain, preserving the mean,
range and standard deviation across three seeds. Supervised models have two
evaluations per domain because the same learned belief is used with two report
grids. Raw returns from different grids are different objectives.

The large final per-example arrays are reconstructed rather than checked into
Git. `evaluation/reconstructable-array-manifest.json` records the original array
hashes; these files are retained with the local run. Use the evaluator to create
a fresh copy. `development/` preserves the three pilot summaries that informed
the protocol; those were not untouched tests.

From the repository root:

```bash
python -m pip install -r requirements-calibration.txt
python -m calibration_lab.inspection_demo
python -m calibration_lab.inspection_verify
python -m calibration_lab.inspection_evaluate \
  --run results/learned-inspection-v1 --output output/inspection-recheck
```

The verifier checks published hashes and frozen sources, regenerates every
unique training world batch, checks matched streams and initial weights, replays
saved trajectories, and reconstructs all 60 final evaluations. The exact
posterior is used by evaluation, not by training or checkpoint selection.

Code, generator, these synthetic artifacts and checkpoints use the repository's
[MIT license](../../LICENSE). No text documents, external datasets or service
credentials are included in this experiment.
