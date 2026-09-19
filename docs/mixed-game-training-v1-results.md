# Games and consequence forecasting: the overnight 9B results

The experiment completed on September 19, 2026. Reinforcement learning improved
decisions in the trained environment families. Adding consequence forecasts
improved held-out forecast scores and short-horizon 2048 performance, but was
worse than reward-only training on reservations. Neither reinforcement arm
showed convincing improvement on the broader non-game decision benchmark.
Lights Out remained largely unsolved. This supports a narrow learning result,
not a claim of general reasoning transfer or a reconstruction of Jev.

All nine pipeline stages completed. The collector recovered and verified 193
files, replayed saved evaluation actions on the original Linux environment,
and deleted the single H200 rental. Estimated compute was **$33.82**, excluding
storage, over about 7.37 hours including setup, evaluation, and recovery. The
cumulative compute estimate is **$133.08**, excluding storage, against the
authorized $500 budget. These are elapsed-time estimates, not a provider invoice.

## Environment results

Both reinforcement arms started from the same game-supervised checkpoint.
“Before” in this table means that shared checkpoint, not the original model.
Checkpoint selection used validation only; these are subsequent test results.

| Test measure | Before reinforcement | Reward only | Reward + forecasts |
| --- | ---: | ---: | ---: |
| Reservation success, prior-weighted | 36.58% | **98.44%** | 82.89% |
| Reservation partial failure | 39.23% | **1.56%** | 6.54% |
| 2048 mean merge score, at most 24 moves | 29.00 | 47.67 | **92.00** |
| 2048 game-over rate within horizon | 4.17% | 4.17% | 8.33% |
| Lights Out solves, at most 12 presses | 0 / 24 | 0 / 24 | 1 / 24 |
| Forecast mean squared distribution error, lower is better | 0.3178 | 0.3192 | **0.2258** |
| Forecast mean log loss, lower is better | 0.6660 | 0.6741 | **0.5288** |

Each game test contains **eight distinct starting boards**, each presented with
original wording, reversed options, and rewording. The 24 presentations are not
24 independent boards. Reservation results enumerate six hidden worlds for
each of eight held-out profiles, in three presentations, weighted by the public
world probabilities. These are small, single-training-seed comparisons within
known mechanisms, not population estimates. More game score does not mean
winning 2048; games are truncated at 24 moves. The hybrid's higher game-over rate
also shows that its game behavior did not improve on every measure.

The forecast metric averages tasks equally and compares predicted distributions
with target distributions. The field called `brier` in the machine-readable
receipt is squared distance to the target distribution; for soft targets it
omits irreducible outcome variance. It is not a direct estimate of deployment
calibration. The hybrid reduced this error by about **29%** relative to its
starting checkpoint; reward-only training did not improve it.

Forecast targets combine exact reservation probabilities, deterministic game
consequences, and next-tile-location estimates from 128 native executions per
2048 query. They do not predict long-horizon success. The 384 test forecast presentations contain 240 distinct questions from 91
groups across 11 tasks, sampled with replacement; details are recorded in
[outcomes.json](../results/mixed-game-training-v1/outcomes.json). The extra
forecast supervision is additional data and computation. This experiment cannot
separate the benefit of that data from the benefit of the particular joint loss.

## Did games help other decisions?

The fixed non-game benchmark has 1,128 questions from 92 tasks and 1,100 question
groups. Each task has equal weight. It was excluded from this run's training and
checkpoint selection, but had been opened in earlier research.

| Selected checkpoint | Accuracy | Acceptable-set log loss, lower is better |
| --- | ---: | ---: |
| Original supervised model | 77.85% | **0.4997** |
| General-only supervised continuation | 77.85% | **0.4997** |
| Supervised continuation with games | **78.26%** | 0.5295 |
| Then reward-only reinforcement | 77.99% | 0.5265 |
| Then reward + forecast reinforcement | 78.17% | 0.5472 |

The game-supervised change was +0.41 percentage points, with a descriptive 95%
paired group-bootstrap interval of **−0.76 to +1.45 points**. Relative to the
original, reward-only changed accuracy by +0.14 points (−0.94 to +1.17), and the
hybrid by +0.32 points (−0.76 to +1.49). All include no improvement. These
post-run intervals describe this fixed benchmark's groups, not variation across
training seeds or unseen task families.

Probability scores worsened: hybrid minus original log loss was +0.0475
(descriptive interval +0.0276 to +0.0687). This metric concerns probabilities
assigned to acceptable options, not probabilities that an action will succeed.
Passing the smaller validation retention guard did not guarantee preservation
on the separate transfer test. The original demo checkpoint remains appropriate.

General-only continuation selected **step zero**: subsequent checkpoints were
worse on validation log loss. Game continuation selected step 250. Both stopped
after three non-improving validation checks. Thus the earlier upward accuracy
curve did not establish that simply training longer would improve this model.
This was supervised decision fine-tuning, not new foundation-model pretraining.

## Training actually consumed

| Arm | Completed updates | Examples or executed decisions | Selected update | Elapsed time |
| --- | ---: | ---: | ---: | ---: |
| General supervised | 751 | 48,064 row presentations; 15,410,708 tokens | 0 | 55.8 min |
| Games supervised | 1,002 | 64,128 row presentations; 20,116,205 tokens | 250 | 68.1 min |
| Reward-only reinforcement | 94 | 2,256 episodes; 29,310 transitions | 80 | 148.6 min |
| Reward + forecast reinforcement | 87 | 2,088 episodes; 27,104 transitions | 80 | 147.7 min |

The two reinforcement runs together executed **4,344 training episodes and
56,414 decisions**. These counts include work after the selected checkpoint;
selected-checkpoint counts are preserved in the results. Both stopped at the
training time cap, with time reserved for final evaluation. They had equal
maximum time budgets, not equal realized transitions or tokens. The hybrid
consumed 5,568 forecast presentations across optimization passes, not 5,568
distinct situations.

Proximal Policy Optimization used fresh trajectories from actual upstream
PufferLib Lights Out and 2048 game steps and our native reservation environment.
This was our PyTorch learner, not the native PuffeRL trainer. It changed about
43.3 million adapter parameters inside the Qwen3.5-9B language network, across
496 tensors. Original pretrained matrices stayed frozen. The critic used
detached features, so it could not cause the observed language-weight changes
on its own. Reward-only also includes entropy regularization and general replay.

The unchanged-policy mechanics check passed the configured clipping guard,
but was not numerically exact: the largest action-probability difference was
0.01334 and sampled likelihood ratios ranged from 0.93284 to 1.07929 before an
update. Investigating this discrepancy is a follow-up; do not describe this as
bit-identical policy recomputation.

## What to do next

Before committing to a much larger run, repeat this comparison with multiple
training seeds and more held-out boards/profiles. Add a matched no-game
reinforcement arm: both current reinforcement arms include reservations and
games, so their gains cannot isolate the contribution of games. Add a
forecast-supervision-only arm to separate prediction learning from interaction.

The most useful new data would be verified tool-use episodes with fresh task
families, changing costs, missing evidence, and delayed outcomes, alongside
counterfactual forecasts. Keep entire task families out of training. Improve
Lights Out's difficulty progression and investigate probability drift before
scaling compute. The promising thesis is that verified consequence supervision
can improve forecasts alongside action learning, with tradeoffs across tasks.
We have not shown that it improves general decisions or identified Jev's method.

## Evidence and reproduction

- [Prospective protocol](mixed-game-training-v1-protocol.md) and
  [unchanged source/data freeze](../results/mixed-game-training-v1/freeze.json).
- [All five transfer checkpoints and paired comparisons](../results/mixed-game-training-v1/transfer.json).
- [Training receipts, held-out outcomes, and parameter audits](../results/mixed-game-training-v1/outcomes.json).
- Native replay receipts for [reward](../results/mixed-game-training-v1/replay-audits/reward.json)
  and [hybrid](../results/mixed-game-training-v1/replay-audits/hybrid.json).

The native auditor replayed eight evaluation files per arm: 960 recorded episode
presentations per arm, including shared baselines, validation, and final test.
It verified 7,138 and 7,199 recorded transitions respectively, along with public
inputs, action identities, native rewards, terminal states, and aggregates.
This verifies environment execution; it does not re-run model inference or make
the repeated presentations independent. Forecast metrics were independently
recomputed from the saved predictions. All 193 recovered artifact hashes and
the separate replay-audit manifest were verified before analysis.

Run `python -m games_lab.transfer_report --root output/mixed-game-cloud-v1
--output output/mixed-game-transfer-report-v1.json` on the recovered archive to
recompute the transfer report. Reporting modules were added after the training
freeze and did not alter model execution, checkpoint selection, or the demo.
The five focused reporting/replay tests pass. The root README and prior
experiments remain unchanged.
