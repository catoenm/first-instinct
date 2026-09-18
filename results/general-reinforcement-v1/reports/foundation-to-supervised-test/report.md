# Paired decision-model evaluation

All changes below are trained minus base. Positive accuracy and negative losses indicate their respective directions; the report does not declare an overall winner.

17,277 paired rows, 15,048 world/document groups, 80 tasks. 147 rows have multiple acceptable answers.

| Row-weighted metric | Base | Trained | Change [95% group-bootstrap interval] |
|---|---:|---:|---:|
| accuracy | 0.6333 | 0.7812 | +0.1479 [+0.1398, +0.1562] |
| acceptable_set_log_loss | 0.7513 | 0.4775 | -0.2738 [-0.2888, -0.2602] |
| single_label_multiclass_brier | 0.4757 | 0.3043 | -0.1714 [-0.1795, -0.1627] |

| Equal-task macro metric | Base | Trained | Change |
|---|---:|---:|---:|
| accuracy | 0.6062 | 0.7991 | +0.1929 |
| acceptable_set_log_loss | 0.8000 | 0.4472 | -0.3528 |
| single_label_multiclass_brier | 0.5026 | 0.2801 | -0.2225 |

| Complete-group accuracy: every row correct | Base | Trained | Change | Groups |
|---|---:|---:|---:|---:|
| All evaluated members correct | 0.6143 | 0.7666 | +0.1523 | 15048 |

For two-row groups, this measures both members correct. The full group-size distribution is recorded in report.json.

| Subgroup | Rows / groups / tasks | Accuracy, base → trained | Set log loss, base → trained |
|---|---:|---:|---:|
| environment | 1000 / 1000 / 2 | 0.4610 → 0.8760 | 1.1625 → 0.2801 |
| public | 13077 / 12448 / 46 | 0.6618 → 0.7346 | 0.6924 → 0.5796 |
| verified | 3200 / 1600 / 32 | 0.5709 → 0.9422 | 0.8632 → 0.1220 |

Single-label multiclass Brier is the sum over options and ranges from 0 to 2. Multiple-acceptable-answer rows are excluded. These option distributions are not automatically calibrated success forecasts.

Intervals use 1,000 paired resamples of complete groups (seed 41). Rows sharing a world stay together. Macro means are descriptive. Intervals cover world-sampling variation, not training-seed variation, task selection, or pretraining contamination.

## Reinforcement-learning runs

Best checkpoints were selected by the criterion recorded in each run; latest results and all paired intervals are in report.json. Parameter-change evidence refers to latest weights.

| Run | Selected update | Latest changed / policy gradient nonzero | Status |
|---|---:|---|---|
| rl-s47-reward | 0 | True / True | complete |
| rl-s47-hybrid | 0 | True / True | complete |
| rl-s53-reward | 100 | True / True | complete |
| rl-s53-hybrid | 0 | True / True | complete |

| Run / split | Expected reward, baseline → best | Event Brier, baseline → best | Posterior mean squared error, baseline → best | Raw identities verified |
|---|---:|---:|---:|---|
| rl-s47-reward / new_domain | 0.3428 → 0.3428 | 0.1300 → 0.1300 | 0.0025 → 0.0025 | True |
| rl-s47-reward / shift | 0.3252 → 0.3252 | 0.1681 → 0.1681 | 0.0151 → 0.0151 | True |
| rl-s47-reward / test | 0.2710 → 0.2710 | 0.1534 → 0.1534 | 0.0028 → 0.0028 | True |
| rl-s47-reward / validation | 0.1977 → 0.1977 | 0.1116 → 0.1116 | 0.0021 → 0.0021 | True |
| rl-s47-hybrid / new_domain | 0.3428 → 0.3428 | 0.1300 → 0.1300 | 0.0025 → 0.0025 | True |
| rl-s47-hybrid / shift | 0.3252 → 0.3252 | 0.1681 → 0.1681 | 0.0151 → 0.0151 | True |
| rl-s47-hybrid / test | 0.2710 → 0.2710 | 0.1534 → 0.1534 | 0.0028 → 0.0028 | True |
| rl-s47-hybrid / validation | 0.1977 → 0.1977 | 0.1116 → 0.1116 | 0.0021 → 0.0021 | True |
| rl-s53-reward / new_domain | 0.3428 → 0.3395 | 0.1300 → 0.1319 | 0.0025 → 0.0073 | True |
| rl-s53-reward / shift | 0.3252 → 0.2930 | 0.1681 → 0.1777 | 0.0151 → 0.0269 | True |
| rl-s53-reward / test | 0.2710 → 0.2643 | 0.1534 → 0.1594 | 0.0028 → 0.0069 | True |
| rl-s53-reward / validation | 0.1977 → 0.2019 | 0.1116 → 0.1162 | 0.0021 → 0.0061 | True |
| rl-s53-hybrid / new_domain | 0.3428 → 0.3428 | 0.1300 → 0.1300 | 0.0025 → 0.0025 | True |
| rl-s53-hybrid / shift | 0.3252 → 0.3252 | 0.1681 → 0.1681 | 0.0151 → 0.0151 | True |
| rl-s53-hybrid / test | 0.2710 → 0.2710 | 0.1534 → 0.1534 | 0.0028 → 0.0028 | True |
| rl-s53-hybrid / validation | 0.1977 → 0.1977 | 0.1116 → 0.1116 | 0.0021 → 0.0021 | True |

Binary event Brier ranges from 0 to 1. The four forecast evidence masks share one event and are clustered by world. Missing receipts or evaluations remain missing; receipt claims are not independent tensor verification.
