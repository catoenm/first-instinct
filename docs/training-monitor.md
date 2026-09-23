# Training monitoring

Saved TensorBoard curves do not establish that a rental is currently running.
Check the latest owned-rental receipt, provider state, and completed optimizer
updates. The original supervised and reinforcement-learning runs described below
are complete; the original supervised checkpoint remains selected.

The earlier local TensorBoard setup used **http://127.0.0.1:6006/** and copied
logs every twenty seconds. It does not modify or restart training, upload metrics
to a hosted service, or require a separate account.

Select `supervised-01` on the left and use **Step** as the horizontal axis:

- `loss/train`: loss on each training batch; lower is better, but batches differ.
- `loss/validation`: mean per-task loss on the fixed validation subset. The
  untouched model is step zero; new measurements arrive every 500 updates.
- `accuracy/validation_macro`: average task accuracy on that same subset.
- `progress`: actual examples, tokens, percentage and elapsed training minutes.
- `optimization` and `throughput`: learning rate, gradient size and speed.

The four completed reinforcement-learning runs have charts that include
sampled reward, expected validation reward, forecast error, and optimization
measurements. Losses averaged across the two optimization passes are labeled by
objective. Held-out test and challenge results are kept out of live monitoring.

The `machine` run shows GPU utilization, memory and temperature, plus estimated
GPU rental cost. That estimate includes setup and evaluation time, excludes
storage, and is not a provider invoice. The **Text** tab shows the pipeline phase
and last successful refresh. If a connection fails, existing curves remain;
check that timestamp instead of assuming a static curve means training stopped.

Events are imported from the original JSON logs. Their wall times are ingestion
times, so use Step and `progress/elapsed_minutes` for training history. Original
raw measurements remain the authoritative record. Events are stored locally in
`output/general-monitor-v1/events`; they remain available after the rental is
stopped. The earlier software-inspection demo is a separate page on port8765.

## Startup and shutdown protection

The decision-focused comparison has two separate watchdogs. Before allocating
a GPU, start the manual [startup-guard workflow](../.github/workflows/training-startup-guard.yml)
for its unique rental name and verify that authentication succeeded and its
cutoff step is running. It executes on GitHub's hosted runner and requests a
provider stop after at most twenty minutes unless canceled. Its provider token
is an encrypted repository secret; the workflow never allocates or deletes pods.

Cancel this startup watchdog only after verifying an authenticated, running
deadline guard inside the pod. That second guard protects the complete training
and recovery period. A local timeout alone depends on the laptop remaining
awake and connected before the remote guard is installed. Neither a running
provider instance nor successful installation is evidence of an optimizer update.

Do not rely on the old Runpod `--stop-after` flag. Runpod removed it after finding
that the backend accepted deadlines without enforcing them.
[Upstream explanation](https://github.com/runpod/runpodctl/pull/330).

## Reproduce from recovered runs

Install the optional monitor dependencies in a separate environment:

```sh
python -m venv .local/monitor-venv
.local/monitor-venv/bin/python -m pip install -r requirements-monitor.txt
.local/monitor-venv/bin/python -m general_lab.monitor snapshot \
  --root output/general-cloud-v1 > output/general-monitor-snapshot.json
.local/monitor-venv/bin/python -m general_lab.monitor export \
  --snapshot output/general-monitor-snapshot.json --logdir output/general-monitor-v1/events
.local/monitor-venv/bin/python -m tensorboard.main \
  --logdir output/general-monitor-v1/events --host 127.0.0.1 --port 6006 \
  --reload_interval 10 --load_fast false
```

Repeated exports skip already imported training steps. The live SSH connection
is a private local helper, separate from the reusable exporter.
[TensorBoard documentation](https://github.com/tensorflow/tensorboard/blob/master/README.md).
