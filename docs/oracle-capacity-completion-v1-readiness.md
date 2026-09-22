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
253 MB and added no swap. No hardware was allocated during local qualification.

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

The cloud launcher and artifact collector now use this exact bundle. Billing
was reconciled within the existing authorization before the allocation below.
Actual CUDA evaluation has now begun; local qualification is not a performance
result. The original supervised release remains selected.

## Evaluation allocation — September 22, 2026

The exact cloud entry and its failure-archive path have now passed local
qualification. The entry also verifies that all 36 report cases use the existing
catalog executor, so this cohort requires no external application worker.

A single H200 was allocated at $4.59/hour for evaluation only, with a 90-minute
provider shutdown deadline and a maximum $10 allocation from the remaining
original project budget. The two fixed final checkpoints are reward update 28
and the combined arm after 16 teacher updates plus 15 reward updates.
Remote setup and all input checks passed. The first saved adapter loaded with
the expected tensor identity and is evaluating the canonical question panel;
the second checkpoint is queued. The independent remote shutdown guard and
autonomous launcher are active. No complete quality result is available yet.
This evaluation makes no additional learning updates or release promotion.
