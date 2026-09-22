# Longer paired-supervision capacity experiment

One H200 has been allocated at $4.59/hour for the prepared longer experiment.
The stage ceiling is $35 from the existing authorization, with a six-hour
independent rental limit. At this launch snapshot, bootstrap is underway;
there are **zero observed optimizer updates and no new performance results**.

The experiment starts from the original Qwen3.5-9B supervised adapter, rather
than a previously selected capacity checkpoint. It pairs verified decisions
and inspection choices with consequence distributions from the same curriculum,
plus general-task replay. This stage tests supervised learning capacity. It does
not isolate a reinforcement-learning contribution or establish new-mechanism
transfer.

## What changed about the training dose

The previous direct-teacher arm completed32updates. This schedule allows256,
with ordinary plateau stopping deferred until128accepted updates. Every update
presents48decision/inspection questions,14forecasts and32general replay
questions. The12starting-decision teachers recur with changing answer positions.
By update128, the schedule covers all742canonical teachers,1,535distinct
forecasts and4,096general replay presentations. These are planned exposures,
not additional independent worlds or evidence of data consumed.

Progress is measured even before the joint selection threshold is crossed.
Unsafe changes, rejected updates and elapsed-time limits still stop immediately.
Each32updates, the existing evaluations check executed database/report outcomes,
forecast quality, answer-order sensitivity and general-task retention. The
training phase can use up to4.5hours including intermediate evaluations, while
final evaluation and recovery retain separate30-minute and15-minute reserves.

The original selected model remains unchanged until the existing improvement
checks pass. Even a capacity pass here does not qualify a public release:
the database mechanism is already trained, and transfer to unfamiliar mechanisms
still needs evaluation.

## Local qualification completed before rental

- The corrected730-file input bundle passed21focused tests in a fresh extraction.
- Exact entry checks reject changes to the data freeze, examples, trainer,
  runtime controls and starting adapter before importing model code.
- The actual launcher's setup-failure branch preserved a verified archive
  without loading a model or making an optimizer update.
- The GPU entry will additionally check actual probabilities and gradients
  for the admitted target types before starting the optimizer.

Before model loading, an integration review caught an optimizer-membership
mismatch: the trainer included only language parameters, while its rollback
guard requires every trainable policy parameter. Startup paused on the same
rental. The corrected optimizer tracks both sets; the supervised loss still
forbids critic gradients, so critic weights remain unchanged. An additional
test runs this exact optimizer through a real guarded update. The original
bundle is preserved, and the replacement bundle and entry were qualified
before startup resumed. No model update was discarded or additional rental
created.

The CPU bundle check peaked below349MB, and the entry/failure check below630MB,
without additional swap. No foundation model ran on the Mac.

The cloud launcher, provider shutdown guard and artifact recovery use the same
owned rental. Recovery verifies the archive and every included file before
deleting the GPU. Training receipts record completed backwards, rejected steps,
accepted updates and checkpoint hashes separately.

See the [frozen protocol](paired-capacity-v1-protocol.md),
[aggregate launch record](../results/paired-capacity-v1/launch.json),
[trainer](../tool_lab/paired_capacity_train.py), and
[runtime controls](../tool_lab/paired_capacity_runtime.py).
