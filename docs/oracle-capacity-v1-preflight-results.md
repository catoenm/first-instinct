# Capacity preflight: zero updates, a diagnostic-selection bug

The GPU run stopped during qualification, before the three training arms began.
There were **zero optimizer updates**, no new trained checkpoint and no release
promotion. All 358 output files were recovered and verified before the pod was
deleted. Estimated compute cost was $1.08, excluding storage.

The failure was in the diagnostic, not evidence that reinforcement learning
cannot update the model. It selected the longest collected actor input. That
input ended with `finish`, had zero remaining reward and a zero critic estimate.
Its advantage was therefore zero. Multiplying the policy loss by zero correctly
produced a zero gradient, which the diagnostic incorrectly rejected. Fifteen of
the other seventeen collected transitions had nonzero advantages.

A local regression test reproduces the failure using an actually executed
database trajectory. A separate successor probe selects the longest observed
record with a nonzero earned advantage for the gradient connectivity check.
It changes no reward and filters no training rollout. If no informative record
exists, it fails explicitly rather than inventing a learning signal. The
successor qualifier is implemented and locally checked; it has **not passed GPU
qualification**. The original failed qualifier and all frozen training inputs
remain unchanged.

## Useful measurements recovered before the failure

Both unchanged models completed all 2,064 canonical questions and their reversed
menus: 8,256 predictions in total. Metrics below average equally across the 24
goal/cost/remaining-horizon cells of one existing database mechanism. These are
training-owned diagnostics, not held-out transfer results.

| Unchanged model | Action accuracy, canonical | Action accuracy, reversed | Outcome Brier, canonical | Outcome Brier, reversed |
|---|---:|---:|---:|---:|
| Original foundation | 31.7% | 48.8% | 0.5952 | 0.6565 |
| Current supervised adapter | 41.2% | 40.0% | 0.6248 | 0.5897 |

Lower Brier scores are better. They include irreducible uncertainty, so a correct
probability forecast need not score zero. Supervised adaptation helps some views
of this panel and hurts others; these results do not support a blanket claim that
it improved or damaged decision making.

When answers are matched by identity across reversed menus, the current adapter
changes its greedy next action on 17.2% of cases under the same cell weighting.
The foundation changes on 75.8%. Deterministic argmax tie-breaking is included in
these rates. Their average changes in probability distributions are also
nonzero, so this is not solely a tie-breaking effect. Menu-order sensitivity
remains a concrete weakness to measure when evaluating any trained successor.

The 12 real qualification episodes contained 18 decisions. Independently checked
execution receipts confirm their rewards, public histories and token inputs.
The saved current-policy likelihood check reports exact agreement between
sampling and training-mode rescoring. It does not establish that the remaining
gradient checks would pass; the process stopped at its first actor probe.

## Verification and accounting

The independent audit rehashed the archive and every recovered file, checked
source/data lineage, recomputed every control metric and menu comparison, and
reconstructed the zero-signal probe from earned returns. Five focused regression
and audit tests passed locally without loading the foundation model.

Prepared data remains 516 direct decision/inspection questions, a 7,202-question
forecast pool and 4,096 replay questions. **None was consumed by an optimizer in
this attempt.** The 8,256 control predictions and one actor diagnostic backward
are separate qualification work.

The entire $20 stage hold remains conservatively reserved, along with earlier
holds, within the original $500 authorization. Unallocated authorization is
$90.64. No GPU remains active and no replacement was rented during this audit.

See [machine-readable audit](../results/oracle-capacity-v1/preflight-audit.json)
and [the original comparison protocol](oracle-capacity-v1-protocol.md). The next
step is to package and qualify the narrowly corrected probe before any separately
bounded continuation, keeping the training comparison and its gates unchanged.
