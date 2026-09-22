# Better forecasts, unchanged task completion

Both previously unmeasured final checkpoints now have complete, independently
audited evaluations. Neither meets the predefined joint improvement condition.
The original supervised model remains selected.

| Checkpoint | Decision accuracy | Database return | Goals completed | Forecast Brier error |
|---|---:|---:|---:|---:|
| Original supervised parent | 41.2% | 0.0633 | 5/12 | 0.6248 |
| Direct decision supervision, 32 updates | 55.3% | 0.0700 | 5/12 | 0.6234 |
| Reward learning, 28 updates | 39.8% | 0.0583 | 5/12 | 0.5904 |
| 16 teacher updates followed by 15 reward updates | 53.9% | 0.0700 | 5/12 | 0.6109 |

Higher accuracy and return are better; lower Brier error is better. Panel metrics
weight goal, cost and remaining-horizon cells equally. The last two rows now
measure the actual saved final weights, replacing the earlier evaluation gap.
They do not represent equal amounts of training.

The reward arm reduced forecast error by 0.0344, but slightly reduced database
return. The combined arm improved return by 0.0067 and forecast error by 0.0139.
The predefined improvement check required return to increase by at least 0.10
and forecast error to fall by at least 0.02, while passing retention checks.
Both final models passed the retention checks and failed the joint improvement
check. General-task accuracy was 86.89% and 86.84%, against 86.84% for the parent.
Report-workflow return was unchanged.

Every training arm also received forecast supervision and general-task replay.
These results therefore do not isolate reinforcement learning as the cause of
the forecast gains. The database is one authored, training-owned mechanism with
four world/goal tasks. This establishes neither unfamiliar-task transfer nor
Jev equivalence. Reserved release evaluations remain closed.

## Where the remaining decision errors appear

A separate, descriptive analysis of the already recorded predictions finds that
the reward model still chooses an optimal initial action in 0 of 6 starting
contexts. The combined model improves this to 1 of 6, matching the direct
supervision model. Reversing the offered options reduces the direct-supervision
and combined models' initial optimal choices to 0 of 6.

Across the full action panel, the combined model changes its chosen action in
16.6% of comparisons when the menu is reversed, using the same equal-cell
weighting. The state, question and available actions are otherwise identical.
This is sensitivity to presentation order, not evidence of a changed world.
These post-hoc diagnostics were not used to select a checkpoint and do not
prove why learning failed to improve task completion.

The closed learning receipts also establish the actual exposure:

| Arm | Starting next-action teacher presentations | Repetitions per starting question | Reward episodes | Sampled optimal first actions |
|---|---:|---:|---:|---:|
| Direct supervision | 48 | 8 | 0 | — |
| Reward learning | 0 | 0 | 336 | 56 |
| Combined | 24 | 4 | 180 | 36 |

There are only six underlying starting next-action questions. Their order is
fixed within each context throughout teacher and actor training. The combined
arm also received 24 starting inspection-value presentations, from six separate
questions. Repetitions and menu variants are not new underlying tasks. Useful
initial actions were sampled, so their complete absence from experience is not
an explanation; these counts do not establish that the learning signal was
sufficient.

The next useful investigation is the learning signal at those early decisions:
its actual frequency, action margins, cost differences and menu-position
coverage. A prospective follow-up should separate whether the model can learn
these exercised decisions from whether it transfers to new mechanisms. More
copies of these worlds would not constitute broader training data. The current
evidence does not justify increasing model size.

## Execution and accounting

This continuation performed **zero optimizer updates**. For each saved model it
completed 2,064 canonical questions, 2,064 reversed-menu questions, 12 executed
database episodes, 36 report episodes, 308 report forecasts and 622 general-task
questions. Loaded and final adapter tensor hashes matched, the saved critics
were unchanged, and no gradients accumulated.

Both evaluations took about 10.4 minutes combined, including model loading.
The collector recovered and verified all 389 artifact files before deleting the
H200. Estimated compute cost was $1.36 excluding storage. The independent audit
reconstructed metrics and executed outcomes without model inference on the Mac;
it took 4.26 seconds, used less than 582 MB and added no swap.

See the [aggregate results](../results/oracle-capacity-completion-v1/summary.json),
[evaluation protocol](oracle-capacity-completion-v1-protocol.md),
[evaluation implementation](../tool_lab/oracle_capacity_completion.py),
[independent metric routines](../tool_lab/oracle_capacity_final_audit.py), and
[original training report](oracle-capacity-probe-v2-results.md). The original
training consumption counts are unchanged; this evaluation adds no training data.
