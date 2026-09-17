# Calibration laboratory evidence

- [Post draft](../../docs/calibrated-decisions-post.md)
- [Experiment report and reproduction](../../docs/calibration-results.md)
- [Every method, seed, and domain](tables.md)
- [Follow-up tables](thresholds/tables.md)
- [Complete downloadable evidence](https://github.com/catoenm/first-instinct/releases/tag/calibration-v1)

`summary.json` summarizes all seeds; `results.json` retains per-run metrics,
reliability bins, and all 25 workflow settings. `selection.json` records choices
made before final evaluation; its `final_test_opened: false` describes that
selection-time snapshot, not the current state of the experiment. All final
tests are now public and must not be treated as unseen data in later tuning.

`verification.json` records reconstruction checks. `artifacts_sha256.json`
indexes the full corresponding run in the release archive, including files too
large for this source directory. `code_snapshot/` contains the exact original
training code and protocol. The follow-up uses new evaluation seeds and lives
under `thresholds/`; it reuses the fixed first-experiment model checkpoints as
references on those new states.

Main run: `20260917T071256.505747Z`, source frozen at `7eb8546`.
Follow-up: `20260917T071725.964689Z`, source frozen at `53c3389`.
Development used only seed 101. No method was tuned after its respective final
evaluation. A later maintenance change bounds the number of logged rollout
examples for user-selected batches smaller than 16; it does not change the
published runs' 1,024-example batches or any gradient update.

The 35 trained networks, synthetic data, and generated evidence use the repository
MIT license. The text encoder's pretrained weights and upstream language datasets
are not included in this experiment bundle.
