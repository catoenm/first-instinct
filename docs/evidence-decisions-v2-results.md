# Forecast practice helps; reward learning remains seed-sensitive

The costly-evidence pilot is complete. Direct consequence supervision improved
executed decisions and consequence probabilities in both seeds. Reinforcement
learning produced the strongest individual decision result, but did not do so
reliably: three of four arms containing reward learning stopped at the policy
divergence limit. We are keeping the supervised demo checkpoint unchanged.

These are **held-out combinations within three authored mechanisms**, not new
task families or evidence of general tool-use ability. The next stage needs
broader mechanisms and a more reliable update guard, not a larger model yet.

## Every arm, including early stops

All arms started from the same Qwen3.5-9B supervised step-2,742 adapter.
Forecast-only means binary consequence supervision plus general-task replay.
Reward-only means Proximal Policy Optimization plus the same replay schedule.
Hybrid adds consequence supervision to that reinforcement-learning objective.
See the unchanged [prospective protocol](evidence-decisions-v2-protocol.md).

| Method / seed | Updates run / selected | Mean return ↑ | Success ↑ | Forecast Brier ↓ | Forecast log loss ↓ | General macro accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original checkpoint | 0 / 0 | 0.3164 | 70.31% | 0.3115 | 0.8404 | 77.99% |
| Forecast-only / 1507 | 40 / 40 | 0.4832 | 78.12% | 0.0964 | 0.3278 | 77.85% |
| Forecast-only / 1609 | 40 / 40 | 0.5022 | 79.69% | 0.0683 | 0.2267 | 77.72% |
| Reward-only / 1507 | 31 / 31* | 0.6154 | 84.90% | 0.3020 | 0.8251 | 77.99% |
| Reward-only / 1609 | 2 / 0* | 0.3164 | 70.31% | 0.3115 | 0.8404 | 77.99% |
| Hybrid / 1507 | 40 / 40 | 0.6871 | 87.50% | 0.1074 | 0.3563 | 77.94% |
| Hybrid / 1609 | 2 / 2* | 0.3010 | 68.23% | 0.2745 | 0.7441 | 77.85% |

\* Stopped for policy divergence. The frozen implementation checked divergence
after an optimizer step, then allowed validation to select that update. Thus
reward-1507 and hybrid-1609 selected the very updates that triggered the stop.
We preserve and disclose those results; they are not approved deployment
checkpoints. Reward-1609 selected the unchanged starting checkpoint despite
having performed four optimizer steps. Future training must reject an over-limit
update before checkpoint eligibility, restoring both parameters and optimizer
state when rejecting an attempt.

Return includes verification rewards and actual action costs. Forecast Brier
here is the mean squared error of a **single binary success probability**;
lower is better. It is not action-selection confidence. General accuracy is
the mean across 92 task groups and 1,128 questions. Its movement is small and
does not demonstrate improved broader capabilities. General macro log loss
ranges from 0.4957 to 0.5210 against the original 0.4997; all selected arms
remain within the predefined retention tolerances on this cohort.

There are 192 test episodes and 1,152 forecast questions, but only **six held-out
fixture roots**, two per mechanism. Related worlds, goals and regimes stay
together. The [paired-root returns](../results/evidence-decisions-v2-final/reanalysis.json)
show each root separately. These observations are correlated; neither 192
episodes nor 1,152 forecasts are independent-task sample sizes. Two seeds and
unequal stopping times do not support an equal-dose ranking of the methods.

## What improved, and what still fails

Forecast practice substantially improved binary probability scores in both
seeds and also improved greedy execution return. Reward-only seed 1507 improved
decisions while leaving forecast Brier close to the original. This supports
treating action reward and consequence forecasting as distinct learning targets.
Hybrid seed 1507 improved both, but seed 1609 did not improve return.

**Stale evidence remains a failure across every arm.** Return stays near -0.02:
the model often commits without resolving the deliberately outdated observation.
Forecast-only also fails the previous-failed-write regime (return -0.05 in both
seeds), whereas hybrid seed 1507 reaches 0.91 there. The curriculum still needs
more varied reasons to inspect, recover, or stop, with labels grounded in
executed alternatives. More copies of the successful fresh-evidence templates
would hide these weaknesses.

The early divergence is measured, but its cause is not established. Both reward
and hybrid seed 1609 stop after two updates, so forecast loss alone cannot
explain it. Their small rollout batches and the combined replay/actor gradient
are candidates for a bounded stability experiment. In hybrid-1609, mean full
policy divergence rose from 0.0077 after the first pass of update two to 0.0347
after the second (limit 0.02); the maximum individual value was 0.1899. All
reinforcement arms had nonzero pure actor gradients into language adapters and
zero critic-to-language gradients. This was actual trajectory reward training,
but the results do not establish a robust training recipe.

## Prepared data versus consumed training

The prepared training pool has three mechanisms, 12 root fixtures, 48 base
world-and-goal tasks, 384 world/goal/regime variants and 2,304 forecast questions.
There are also 4,096 general replay rows. The table counts **unique identifiers /
optimizer presentations** over each full run, including work after the ultimately
selected checkpoint. Each completed update normally makes two optimizer passes.

| Arm | Forecast questions | General replay questions | Policy transitions | Collected episodes |
|---|---:|---:|---:|---:|
| Forecast-only / 1507 | 960 / 1,920 | 603 / 1,280 | 0 / 0 | 0 |
| Forecast-only / 1609 | 960 / 1,920 | 591 / 1,280 | 0 / 0 | 0 |
| Reward-only / 1507 | 0 / 0 | 474 / 992 | 463 / 926 | 248 |
| Reward-only / 1609 | 0 / 0 | 32 / 64 | 26 / 52 | 16 |
| Hybrid / 1507 | 960 / 1,920 | 603 / 1,280 | 609 / 1,218 | 320 |
| Hybrid / 1609 | 48 / 96 | 32 / 64 | 26 / 52 | 16 |

Unique transition identifiers distinguish rollout occurrences, not distinct
underlying tasks. The original control consumed no training rows. Forward-only
evaluation and gradient diagnostics are excluded from optimizer presentations.
Full identifier counts, selected-step counts, source hashes and exact metrics
are in [the compressed analysis](../results/evidence-decisions-v2-final/analysis.json.gz).

## Recovery, lineage and reproduction

The collector recovered and hash-verified all 325 artifacts before stopping and
deleting the owned H200. A fresh provider listing showed no remaining pods.
The conservative compute estimate is **$6.21 excluding storage**, bringing
tracked project compute to **$157.63 excluding storage** under the original
$500 authorization. The initial provider billing response was incomplete;
these are runtime estimates, not a final reconciled invoice.

All seven roles had the same initial trainable-tensor hash:
`17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27`.
The original adapter file hash is
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.
The foundation revision is
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Training changed 43,278,336 internal adapter parameters; original foundation
matrices remained frozen. This is post-training, not foundation pretraining.

The [public result bundle](../results/evidence-decisions-v2-final/) contains
compressed trajectories, forecast/general probabilities, optimizer and rollout
ledgers, gradient diagnostics, checkpoint lineage, test fixtures and the exact
executed source snapshot. Model weights remain in the recovered local archive.
To verify the public hashes and recalculate metrics and paired-root differences
with the Python standard library:

```sh
python -m tool_lab.evidence_result_bundle
```

That small check verifies prediction arithmetic and cohorts; it does not rerun
tools or inference. The full local recovery audit additionally reconstructed
trajectory observations, rewards and final-state checks, validated frozen data,
and checked model-parameter/optimizer records:

```sh
.venv/bin/python -m tool_lab.evidence_report \
  --root output/evidence-decisions-cloud-v2 \
  --output output/evidence-decisions-cloud-v2-analysis.json
```

The [next curriculum qualification](decision-curriculum-v3-results.md) remains
untrained. We will add meaningful application dependencies, preserve whole-family
transfer tests, and qualify the stability change before another bounded 9B
comparison. These experiments do not recover Jev's undisclosed implementation.
