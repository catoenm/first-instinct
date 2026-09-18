# Harbor command-selection qualification evidence

[Results and limitations](../../docs/harbor-command-results.md) ·
[Pre-execution protocol](../../docs/harbor-command-selection.md)

This snapshot contains all 12 controls and all three live trajectories. It is
an integration check, not a training run or selector benchmark.

- `freeze.json`: original source hashes and trial limits, copied unchanged.
- `source-at-run/`: exact original files matching the freeze, including the
  version 1.0.0 history-logging defect found in the post-run audit.
- `tasks/`: all six actual task packages and their original file-hash manifest.
- `controls.json`: selected Harbor result fields for every reference/no-op run.
- `live/*/choices.json`: byte copies of full selector and execution receipts.
- `live/*/choices-with-snapshotted-requests.json`: explicit derivatives restoring
  each request from its immutable mailbox. Use these for observation histories;
  seven redundant request fields in original receipts acquired later history.
- `live/*/mailbox/`: byte copies of the public proposal requests and responses.
- `live/*/harbor-result-projection.json`: selected fields from each Harbor result,
  with the original file's hash. Host-specific job configuration and operational
  logs are omitted. These projections are not the complete original results.
- `live/*/report.json` and `summary.json`: computed returns and aggregate counts.
- `ARTIFACTS.json`: hashes of every other file in this evidence directory.
- `history-audit.json` and `verify.py`: affected events and an offline check of
  source hashes, original requests, exact selector inputs/actions, and rewards.

The first three configuration proposals received relayed public history;
remaining proposals read exact request files. See the results document for the
whitespace difference in that relay. The snapshot is not a complete transcript
of the language-model proposer's internal conversation.

The reference solution and verifier are published for inspection but were not
available inside the live agent container or supplied to the proposer. The
proposer access boundary was enforced through scoped file instructions and
public mailbox inputs, not a claim of a separate operating-system sandbox for
the Codex subagent itself.

No model weights changed. Action-selection probabilities are not calibrated
task-success forecasts. Saved greedy traces are not fresh training-policy
rollouts for Proximal Policy Optimization.
