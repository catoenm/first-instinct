# Paired decision-model evaluation

All changes below are trained minus base. Positive accuracy and negative losses indicate their respective directions; the report does not declare an overall winner.

64 paired rows, 32 world/document groups, 16 tasks. 0 rows have multiple acceptable answers.

| Row-weighted metric | Base | Trained | Change [95% group-bootstrap interval] |
|---|---:|---:|---:|
| accuracy | 0.9062 | 0.8906 | -0.0156 [-0.0469, +0.0000] |
| acceptable_set_log_loss | 0.2858 | 0.3464 | +0.0605 [+0.0025, +0.1354] |
| single_label_multiclass_brier | 0.1542 | 0.1736 | +0.0194 [-0.0004, +0.0430] |

| Equal-task macro metric | Base | Trained | Change |
|---|---:|---:|---:|
| accuracy | 0.9062 | 0.8906 | -0.0156 |
| acceptable_set_log_loss | 0.2858 | 0.3464 | +0.0605 |
| single_label_multiclass_brier | 0.1542 | 0.1736 | +0.0194 |

| Complete-group accuracy: every row correct | Base | Trained | Change | Groups |
|---|---:|---:|---:|---:|
| All evaluated members correct | 0.8125 | 0.8125 | +0.0000 | 32 |

For two-row groups, this measures both members correct. The full group-size distribution is recorded in report.json.

| Subgroup | Rows / groups / tasks | Accuracy, base → trained | Set log loss, base → trained |
|---|---:|---:|---:|
| other | 64 / 32 / 16 | 0.9062 → 0.8906 | 0.2858 → 0.3464 |

Single-label multiclass Brier is the sum over options and ranges from 0 to 2. Multiple-acceptable-answer rows are excluded. These option distributions are not automatically calibrated success forecasts.

Intervals use 1,000 paired resamples of complete groups (seed 41). Rows sharing a world stay together. Macro means are descriptive. Intervals cover world-sampling variation, not training-seed variation, task selection, or pretraining contamination.
