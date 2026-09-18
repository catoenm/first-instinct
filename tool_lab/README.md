# Propose commands, choose one, execute, repeat

A language model proposes concrete shell commands from the public task and
command history. The resident First Instinct model chooses a command or finishes.
Harbor executes the command in a disposable container. Its actual output becomes
the next observation. A separate verifier checks the final files and database.

This is a working environment adapter and trajectory collector, **not a new
training run**. The current generators implement three small task families.
See the [design and bounds](../docs/harbor-command-selection.md),
[qualification results](../docs/harbor-command-results.md), and
[richer task proposals](../docs/harbor-task-families.md).

## Run an example

Prerequisites: Python 3.12 or later, Docker, the repository's resident selector
service at `http://127.0.0.1:8766`, and a proposer to service the mailbox.
Creating this environment does not install or start the 9B model. Use the
[existing demo setup](../docs/general-demo.md) for that service. Run from the
repository root; leave the model's Python environment unchanged.

```sh
python3 -m venv .local/harbor-venv
.local/harbor-venv/bin/python -m pip install -r requirements-harbor.txt
python3 -m unittest test_harbor_selection test_harbor_report -v
python3 -m tool_lab.generate --output output/harbor-example/tasks --seeds 101 102
HARBOR_TELEMETRY=0 PYTHONPATH=. .local/harbor-venv/bin/harbor run --config tool_lab/example-job.json
python3 -m tool_lab.report --job output/harbor-example/jobs/command-selection-example --output output/harbor-example/report.json
```

The example runs one task, with no retries and at most six decisions. The agent
waits for each proposal for at most 180 seconds; the whole episode has a
900-second limit. Use a fresh output directory and job name for another attempt.
Changing seeds generates more instances of the same mechanisms, not new skills.

## Proposer mailbox

While Harbor runs, service new files at
`output/harbor-example/mailbox/<session>/step-NN.request.json`. Each contains a
`request` and its `request_sha256`. Give the proposer only that public request,
never the task package, reference solution, verifier, or expected output.

Have the proposer atomically write the corresponding `step-NN.response.json`:

```json
{
  "request_sha256": "COPY THE REQUEST HASH HERE",
  "proposer": "your model and prompt version",
  "candidates": [
    {"description": "Inspect the data directory", "command": "ls -l /app/data"},
    {"description": "Read the configuration", "command": "cat /app/data/services.json"}
  ]
}
```

These are protocol examples, not a substitute for proposals conditioned on
each new observation. Provide 2–5 distinct commands, each at most 800 characters,
with descriptions at most 160 characters. The harness adds a finish option and
shuffles the menu. It sends the exact commands to the selector, then executes
only the selected command through Harbor. The pilot uses Codex subagents in the
user's existing account; this module contains no unattended model API client.

## What the data means

`agent/choices.json` records exact menus, chosen commands, selection
probabilities and observed outputs. Harbor's separate verifier produces success
or failure. The report computes success minus 0.01 per executed command and
returns for preceding decisions; infrastructure failures receive no training
return. Failed tasks remain in the evidence.

The sample configuration uses greedy selection. Although the adapter also
supports sampling, saved inference traces are not automatically suitable for
Proximal Policy Optimization. That trainer must collect from its current policy
and preserve the exact encoded inputs and action likelihoods. Selection
probabilities are also not calibrated probabilities of eventual task success.
