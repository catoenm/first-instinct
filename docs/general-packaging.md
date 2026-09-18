# Package the general model and evidence

`general_lab.package` builds an adapter-only archive from completed runs. It does not publish, download a model, start inference, or mutate the supplied runs. Run it after collecting and verifying the final artifacts.

```sh
.venv/bin/python -m general_lab.package \
  --supervised-run output/general-supervised-run \
  --data output/general-qwen35-9b-v2 \
  --rl-run reward=output/general-reward-run \
  --rl-run hybrid=output/general-hybrid-run \
  --evaluation foundation=output/general-eval-foundation \
  --evaluation supervised=output/general-eval-supervised \
  --report general=output/general-report \
  --source-audit output/general-public-v1/natural-instructions-audit.json \
  --freeze output/general-runtime-freeze-v1.json \
  --output output/first-instinct-general-v1.tar.gz
```

The run, evaluation, and report paths above are placeholders for actual completed outputs. The supervised run, prepared manifest directory, and output are required; all other input groups are optional. Repeat `--rl-run`, `--evaluation`, or `--report` with unique `NAME=PATH` pairs. Reports use `general_lab.report`'s `report.json`/`report.md` format; evaluations use `general_lab.evaluate`'s metrics and named prediction files.

The archive has a stable layout:

```text
first-instinct-general-v1/
  README.md, example.json, verify.py
  artifact-manifest.json, SHA256SUMS
  runs/supervised/
    run.json, training.jsonl, baseline-metrics.json
    best/adapter_config.json, best/adapter_model.safetensors
    latest/adapter_config.json, latest/adapter_model.safetensors
    tokenizer/...
    [named validation predictions]
  runs/rl-NAME/
    run.json, training.jsonl, optimizer-steps.jsonl, rollouts.jsonl
    best/..., latest/..., tokenizer/...
    [named checkpoint metrics, forecasts, policy traces]
  provenance/
    prepared-manifest.json
    [public-source-audit.json, runtime-freeze.json]
  evaluations/NAME/[metrics.json, named predictions]
  reports/NAME/[report.json, report.md]
  docs/[training protocol, public sources, probe documentation]
  data/general/probes.jsonl
  scale_lab/[minimal inference modules]
  general_lab/[typed question interface]
  [pinned requirements, license, notices]
```

`best/` is the selected checkpoint and `latest/` is the final trained checkpoint. Both are preserved. Each directory remains compatible with the existing `scale_lab.infer.Predictor(run=...)` layout. The supplied tokenizer is included as a reproducibility snapshot; the existing Predictor downloads/loads its tokenizer and foundation from the exact revision in `run.json`. Base weights are therefore still required separately. The generated README includes an inference command and explains how to load latest weights without relabeling them as the selected model.

The companion `first-instinct-general-v1.tar.gz.manifest.json` records the archive's SHA-256, size, run selections, and every included file's checksum. Archive paths, ordering, ownership, permissions, and compression timestamp are deterministic. The helper reopens the completed temporary archive and checks its contents before moving it into place. Existing archives and sidecars are not overwritten. After independently checking the archive checksum and extracting it, `python verify.py` checks payload files against the internal manifest.

## Required consistency checks

- The supervised receipt must be complete, identify a supported pinned model revision, record completed updates and changed language parameters, and match the supplied prepared-manifest hash.
- Selected and latest adapters must both exist. Their configuration must match the foundation. A lightweight tensor inspection rejects full-model tensors and accepts only language low-rank adapter tensors; no model is loaded.
- The included core inference files must match the source hashes recorded by supervised training. Use `--project-root` to supply the appropriate frozen checkout if the working code has changed.
- An optional reinforcement run must start from the exact supplied supervised selected adapter. Its receipt must record optimizer steps, language-parameter changes, and a nonzero pure policy gradient. Packaged tensor fingerprints independently verify that latest differs from the starting adapter; gradient counts themselves remain receipt claims.
- If `selected_update` is zero, the selected tensor fingerprint must equal the supervised starting adapter. The release index and README explicitly say that this is **not an RL-trained selection**, even though latest contains completed reinforcement updates.
- Evaluation receipts must match an included run, the foundation revision, the prepared manifest, and any supplied probe snapshot. Each declared split needs the expected number of prediction records. Reports must refer to prediction hashes included in the same bundle.

Default packaging rejects running, failed, and bounded-stop runs. `--allow-bounded-stop` deliberately permits a finalized partial run only when it still has completed updates, valid adapters, and the other required evidence. Its recorded status remains `bounded_stop`; the bundle does not call the planned training complete. Selected-checkpoint evaluation may be incomplete for such a run, so only actually present, allowlisted evidence is packaged.

## Documentation freeze

An optional freeze receipt uses:

```json
{
  "documentation_sha256": {
    "docs/general-training-v1-protocol.md": "SHA256",
    "docs/general-data-sources.md": "SHA256",
    "docs/general-probes.md": "SHA256",
    "data/general/probes.jsonl": "SHA256"
  },
  "prepared_manifest_sha256": "SHA256",
  "commit": "RECORDED_GIT_COMMIT"
}
```

The three document hashes are required when a freeze is supplied; the probe hash and prepared-manifest hash are optional and checked when present. Other descriptive receipt fields, such as a locally recorded timestamp or source hashes, are retained. This is locally recorded provenance backed by inspectable history, not an externally trusted timestamp. Without a freeze receipt, the README explicitly identifies the documents as a packaging snapshot and makes no verified pre-run chronology claim.

## Exclusions and bounds

Every included path is selected from explicit names or narrow experiment-output patterns. Optimizer state, training-only critic pickle files, raw public/source corpora, tokenized training rows, foundation weights, credentials, `.local` paths, and symlink inputs are excluded. The helper does not recursively copy run directories. Unexpected fields in external prediction files are rejected so a raw training export cannot pass as a prediction file. Credential-like fields in JSON receipts are also rejected.

The default uncompressed limit is four gibibytes and 4,096 files. `--max-bytes` can lower or explicitly raise the byte limit. The archive supports inference and review; it does not support exact optimizer-state resumption.

Validation: `python -m unittest test_general_package` exercises deterministic packaging, extracted-file verification, selection honesty, mismatched provenance, unfinished runs, malformed predictions, symlinks, unwanted files, and overwrite refusal. Tensor inspection was also checked against the project's existing real 129,934,448-byte adapter containing 496 low-rank tensors.
