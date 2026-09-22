# Final-checkpoint evaluation: local qualification complete

The evaluation-only package now binds the preserved reward step-28 and combined
step-31 adapters, their matching critics, the original data freeze, and executable
source hashes. It makes no optimizer updates and opens no reserved release scores.
The two final checkpoints remain unevaluated on a GPU.

Eight tests passed locally and were repeated against a freshly extracted portable
bundle. Four additional corruption controls rejected a changed manifest, swapped
final adapters, a changed critic, and a truncated retention cohort. The bundle
contains 355 hashed input/source files and is 337,385,624 bytes. Qualification used
no model inference or new environment executions; guarded memory stayed below
253 MB and added no swap. No hardware was allocated.

Input manifest:
`83d6e9dabdb1e178a2a88aeacd135923e037103ebf477499dd8420fb58bfb735`.
Portable archive:
`d66d23aa59cc65268161d414070dc517489d602e5d78edf48bdebc7e47256b30`.

The [evaluation protocol](oracle-capacity-completion-v1-protocol.md) preserves all
earlier cohorts and scoring contracts. The [entry point](../tool_lab/oracle_capacity_completion.py)
refuses foundation inference outside Linux CUDA. Its completion receipt requires
every cohort and unchanged weights. A separate
[phase-budget controller](../tool_lab/evaluation_budget.py) tests that a training
cutoff leaves time for final evaluation and recovery; it does not modify the
completed pilot's runner.

Before execution, the remaining work is a bounded cloud launcher and artifact
collector using this exact bundle, fresh billing reconciliation within the
existing authorization, and actual CUDA execution. Local qualification is not
a performance result. The original supervised release remains selected.
