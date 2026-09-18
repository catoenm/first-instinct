# Read-only outcome ToolSandbox cohort report

`general_lab.toolsandbox_transfer_report` analyzes an execution folder from the published `toolsandbox-transfer-execution-v1` executor. It requires the explicit published execution-freeze file checksum, the original supervised reference folder (containing `freeze.json`, `question-mapping.json`, and `results/`), and the unchanged corpus folder. It performs no inference, tool execution, checkpoint selection, training, service changes, or remote operations.

```sh
python -m general_lab.toolsandbox_transfer_report \
  --execution /path/to/transfer-execution \
  --sft-reference /path/to/published-supervised-reference \
  --corpus /path/to/toolsandbox-partial-v1 \
  --freeze-sha256 PUBLISHED_EXECUTION_FREEZE_FILE_SHA256 \
  --output /path/to/new-report-folder
```

The equivalent API is `analyze(execution_folder, sft_reference_folder, corpus_folder, expected_freeze_sha256)`, followed by `markdown(report)`. Output must be a new directory outside the immutable input folders. Raw files are never changed. An actively changing execution snapshot is rejected; analyze a stable copy if inference is still running.

The analyzer checks the execution freeze's content and supplied published checksum, all required frozen source files, the fixed inference contract, all 720 public input hashes and retained token-row accounting. The original supervised analyzer independently verifies and rescores its retained responses. Every original role remains in fixed arm/seed/selected-or-latest order. Exact adapter identities group byte-identical roles, and reference reuse requires the original supervised adapter identity and unchanged response rows.

For a completed identity, the executor's frozen `finished` verifier checks mode-specific artifact hashes and journals. This report additionally validates all response probabilities, ordered indices, attempt input hashes and token counts, raw/validated response agreement, timing chronology, and recorded runtime identity. All 720 responses are rescored with the original frozen scorer; stored metrics and prose must reproduce. A malformed completed unit is marked **invalid**, with no metrics. Missing, failed, and partial identities remain explicit. Complete newline-delimited prefixes of incomplete journals may be counted, but partial data are never scored; an unfinished last line is disclosed. Operational runtime locators, recovery archives, model weights, and foundation caches are not needed for public reanalysis. The existing supervised analyzer may verify its historically recorded optional local files if they happen to be present; it imports no model runtime.

The primary comparison is checkpoint-minus-supervised expected utility and regret at the **initial state**, paired within each of the same 48 roots and then averaged equally. Phone-conditioned results retain the stated observation-probability weights inside each root and are reported separately. Outcome and cost forecast errors remain separate, using summed squared probability error / excess expected Brier and the frozen expected clipped log score. Known singleton costs remain outside model forecast denominators. Per-operation, collection-split, and paired-root records remain in JSON.

The report does not choose a best arm or checkpoint, pool identity aliases as new evidence, manufacture confidence intervals, or equate 720 questions with 720 independent tasks. These are 48 correlated variants of one authored four-world mechanism. Expected action values refer to the declared fixed continuation, not executed adaptive-policy returns. Fresh-process model timing and historical supervised HTTP timing are not a controlled speed comparison.

Input and analysis-source checksums are emitted with the report. They support reproducible rescoring of recorded outputs; they do not attest model tensors in memory. The eight focused tests use a hand-authored finite probability corpus and synthetic journals, including alias preservation, weighted endpoints, tampering, partial data, and public analysis without private locators. They execute no model or environment.
