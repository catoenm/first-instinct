# Executable evidence — first collection study

See the [report](../../docs/executable-evidence.md) and
[protocol](../../docs/executable-evidence-protocol.md). These are authored Python
tasks, not imported repository changes or Jev outputs. The corrected source was
frozen at [`33c30ba`](https://github.com/catoenm/first-instinct/commit/33c30bab5e67161eef431c5761a628a455f36aef).

Three collection rules × three seeds produce nine text prediction heads and
nine visible-test reference heads, with a label-frequency reference for each.
Each collector acquires exactly 100 program labels and sees four views per
program. The frozen language encoder's weights are not included; its output
vectors are included so fitting and evaluation can be reconstructed offline.

| Files | Contents |
| :--- | :--- |
| `source/`, `run.json` | Frozen source, protocol, dependencies and configuration |
| `development-pool/` | 188 training and 76 validation candidates, source lineage, visible receipts and 1,056 text requests |
| `development-features.npz`, `.json` | Frozen language vectors, identifiers, input-text hash and encoder revision |
| `feature_mean.npy` | Centering vector computed only from the unlabeled training pool |
| `*-s*/acquisitions.jsonl` | Ordered queries, verdicts, cache references, logical test counts and physical execution times |
| `*-s*/selection.jsonl` | Bootstrap and later draw probabilities, reasons and acquisition scores |
| `*-s*/round-0.npz` through `round-5.npz` | Both fitted heads and the label prior after every collection round |
| `*-s*/history.json`, `manifest.json` | Progress and checksum-sealed model files |
| `sealed.json`, `evaluation/opened.json` | Evidence that final candidate evaluation followed model sealing |
| `verifications/` | 473 unique private execution receipts, including two quarantined cases |
| `evaluation/regular/`, `transformations/` | Generated final pools, visible receipts and requests |
| `evaluation/cases.jsonl.gz`, `answers.jsonl.gz` | 211 final candidates and separately stored private verdicts; two verdicts are null |
| `evaluation/quarantine.json` | Explicit exclusions; their cost and execution witnesses remain in the data |
| `evaluation/features.npz`, `.json` | Cached final vectors and their provenance |
| `evaluation/results.json`, `predictions.jsonl.gz` | All 81 domain/reference/model result rows and per-candidate forecasts |
| `analysis.json` | Seed summaries, per-task scores and all 31 visible-pass/private-fail counterexamples |
| `verification.json` | Full execution and numerical reconstruction receipt |
| `repair-comparison.json` | All 54 collection-round checkpoints are unchanged by the verifier repair |
| `archives/` | Complete development run and aborted first attempt, with checksums |
| `artifact-manifest.json` | Checksums for the complete published bundle |

There are 475 generated candidate records across 24 tasks, and 1,900 related
text views. The same candidate may be selected by multiple collectors. There
are **209 scored final candidates**, not 836 independent examples or nine
independent copies of the evaluation data. Nine training collectors consume
57,600 logical private test executions. Adding common validation and final
evaluation gives 75,968; caching reduces original physical execution to 473
private queries. A candidate's private verdict is never present in its encoder
input. Full source and receipts are public for audit after the experiment.

## Replay without the language encoder

From the repository root, with Python 3.14:

```bash
python -m pip install -r requirements-calibration.txt
python -m evidence_lab.verify
python -m evidence_lab.demo
```

The recorded verification re-executed 38,574 individual tests, reconstructed
108 fitted heads and all 81 result rows, and found a maximum parameter and
metric difference of **zero** on the training machine. Numerical comparisons
allow a tolerance of 1e-9, although exact replay of selection still depends on
compatible numerical behavior. The encoder vectors are checksum-checked against
the exact rendered text; re-encoding is a separate full-run operation described
in the report. Original pretrained encoder weights require a separate download.

Use `--skip-execution` for reconstruction from already-recorded verdicts. That
option does not independently check the program executions. Use `--output` to
save a fresh verifier receipt in a new location.

## The repaired attempt is not hidden

`archives/aborted-attempt.tar.gz` preserves the original source, all nine
collectors, 113 completed final acquisition records, and the receipt that caused
evaluation to stop. No aggregate final model results were produced. It used
process-randomized string hashes, so unstable exception details are not expected
to replay exactly. The corrected worker uses two fixed, distinct hash seeds.

`archives/development-seed-101.tar.gz` preserves the three development collectors
and validation results. Neither archive contributes to final means. Archive
paths are relative, and their hashes and byte lengths are in
`archives/manifest.json`.

These artifacts report fixed-suite outcomes, not proofs of program correctness.
Reference implementations were authored here without independent human review.
There is no new reinforcement-learning policy in this bundle.
