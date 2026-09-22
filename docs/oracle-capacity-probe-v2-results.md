# Decision supervision learned the questions; task success barely moved

**Follow-up:** the [final-checkpoint evaluation](oracle-capacity-completion-v1-results.md)
has now measured both saved final adapters. Forecasts improved, task completion
remained 5/12, and neither qualified a replacement. The report below preserves
what was measured during the original training run.

The corrected Qwen3.5-9B capacity pilot completed 91 accepted updates across three
arms. None selected a replacement for the original supervised checkpoint. The
supervised arm met its stopping rule; the two reinforcement-learning arms reached
their 40-minute limits before their final weights were evaluated. The combined
method therefore remains inconclusive.

All 522 artifact files were recovered before the H200 was deleted. Estimated
compute cost was $11.08 excluding storage. An independent audit verified executed
rewards, public model inputs, consumed schedules, completed evaluations, stopping
rules and saved language-adapter arrays. See the [aggregate results](../results/oracle-capacity-probe-v2/audit-summary.json)
and [auditor](../tool_lab/oracle_capacity_final_audit.py).

## What the completed evaluations show

All arms started from the same supervised checkpoint and received consequence
supervision and general-task replay. The decision objective changed: direct oracle
labels, reward-based Proximal Policy Optimization, or 16 supervised updates followed
by reward learning. This does **not** isolate the effect of reinforcement learning
on forecast quality, because forecast supervision is present throughout.

| Measured checkpoint | Accepted updates in run | Evaluated update | Decision accuracy | Database return | Forecast Brier error |
|---|---:|---:|---:|---:|---:|
| Original supervised parent | 0 | 0 | 41.2% | 0.0633 | 0.6248 |
| Direct decision supervision | 32 | 32 | 55.3% | 0.0700 | 0.6234 |
| Reward-based decision learning | 28 | 16 | 40.2% | 0.0583 | 0.6012 |
| Combined arm, **supervised phase only** | 31 | 16 | 51.2% | 0.0683 | 0.6253 |

Higher return and lower Brier error are better. Every listed evaluation completed
the goal in 5 of 12 database episodes. General-task macro accuracy remained
between 86.68% and 87.00%, compared with 86.84% initially. The predefined joint
improvement required return to increase by at least 0.10 and forecast error to
decrease by at least 0.02, while preserving the safety checks. No evaluated
checkpoint passed that joint condition.

The reward arm's saved final checkpoint contains 28 reward updates. The combined
arm's final checkpoint contains 16 supervised and 15 reward updates. Neither final
checkpoint has a complete evaluation. Their intermediate measurements must not be
presented as the performance of those final weights or as a completed comparison
at equal training doses. All three selected `best` adapters remain byte-identical
to the original parent. Final trained adapters changed all 496 language-adapter
tensors; the pretrained foundation matrices remained frozen.

These are **training-owned diagnostics** on one database mechanism. They do not
establish transfer to unfamiliar tasks, broad calibration, or Jev equivalence.
Reserved release evaluations remain unopened.

## What was actually consumed

| Learning source, summed across arms | Presentations | Unique question or input identities |
|---|---:|---:|
| Verified next-action / inspection labels | 576 | 252 |
| Executed policy decisions | 784 | 57 |
| Consequence forecasts | 1,820 | 628 |
| General-task replay | 2,912 | 907 |

The policy decisions came from 516 executed training episodes, reusing four
world/goal tasks and 12 world/goal/cost cells. Repeated presentations across arms
are counted as repetitions. Qualification and evaluation are separate. The
prepared pools contain 516 decision questions, 7,202 forecasts and 4,096 replay
questions; those pool sizes are not amounts consumed. The oracle reused 1,692
previously verified branches and introduced no new environment mechanism.

## What this changes about the next experiment

A descriptive breakdown locates an important remaining weakness: among six
initial public states, direct supervision changed optimal first choices from
0/6 to 1/6; the reward arm's measured checkpoint remained at 0/6. Much of the
supervised accuracy improvement concerns later states. This suggests a focus on
initial evidence gathering and cost-sensitive choices, but does not prove a cause.

The immediate experimental gap is simpler: evaluate the preserved final weights
before claiming that the combined method worked or failed. A subsequent training
runner should reserve an explicit final-evaluation window, stop collecting new
episodes before that window, and keep partial work distinguishable from evaluated
checkpoints. That change needs local deadline/interruption tests before another
rental. Increasing model size is not supported by this pilot's evidence.

The [original protocol](oracle-capacity-v1-protocol.md),
[preflight correction](oracle-capacity-probe-v2-protocol.md) and all completed work
remain preserved. The audit ran without foundation inference or new tool executions
on the Mac; seven focused tests passed and peak guarded audit memory was under 1 GB.
