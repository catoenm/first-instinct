# Transactional reservation evidence, version 1

See [results and limitations](../../docs/puffer-reservation-results.md) and
[reproduction commands](../../puffer_lab/README.md).

- `qualification/`: immutable original source freeze, real SQLite branch
  receipts and replays, attempt ledgers, exact conditional targets, guards,
  and the headless PufferLib interface check. The original summary predates
  the separate 64-guard extension; total database attempts are **1,374**.
- `learning-freeze.json`: protocol/source hashes recorded before training.
- `learning/seed-41/`, `learning/seed-73/`: configurations, all sampled rollouts,
  update logs, validation selection, fixed diagnostic trajectories, and three
  small-policy checkpoints each. Each seed trained on 131,072 transitions.
- `reference/`: unmodified upstream headers and original licenses for the
  offline PufferLib environment-interface check.
- `audit.json`: offline reconstruction of the saved training and evaluation
  trajectories, source checks, target recomputation and checkpoint checks.
- `manifest.json`: SHA-256 inventory of every other file in this directory.

The `.pt` files are state dictionaries for the **7,370-parameter local policy**;
load them with `torch.load(..., weights_only=True)`. The `.npz` files contain
numeric arrays and are loaded with `allow_pickle=False`. No Qwen weights or
host-specific compiled binaries are included. The raw execution receipts
contain private simulator state for audit; the policy consumed only the public
39-feature observations in the rollout arrays.

This is one authored environment. The small policy was trained with our CPU
PyTorch implementation of Proximal Policy Optimization. The native CUDA
PuffeRL trainer was not run and the main 9B language model was not updated.
No calibration objective was trained in this experiment.
