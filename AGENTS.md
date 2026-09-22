# Repository conventions

- Keep the root README short: purpose, current model, quickstart, and navigation.
  Put experiment details in the relevant report and link major work from `docs/README.md`.
  Do not append a running activity log to the README.
- Put regression tests and shared test fixtures under `tests/`; use package imports
  such as `tests.test_general_rl`. Run discovery once with `python -m unittest
  discover -s tests -t . -p 'test_*.py' -v`.
- Reuse existing training, environment, evaluation, and accounting modules.
  Add a new module only for a distinct responsibility; avoid copying an entire
  pipeline to vary a recipe or add one check.
- Preserve published receipts, source snapshots, checkpoints, and hashes.
  Historical reproduction uses the recorded Git revision or saved source bundle.
  Never rewrite old hashes to make a changed checkout appear identical.
- Keep running experiments on their qualified source bundle. Repository cleanup
  must not alter remote running code or delete the local inputs needed for recovery.
- Keep generated data, model weights, local launchers, and run logs in the existing
  ignored locations. Commit reusable code, focused documentation, and public results.
