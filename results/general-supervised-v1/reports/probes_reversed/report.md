# Paired decision-model evaluation

All changes below are trained minus base. Positive accuracy and negative losses indicate their respective directions; the report does not declare an overall winner.

64 paired rows, 32 world/document groups, 16 tasks. 0 rows have multiple acceptable answers.

| Row-weighted metric | Base | Trained | Change [95% group-bootstrap interval] |
|---|---:|---:|---:|
| accuracy | 0.8438 | 0.9219 | +0.0781 [-0.0312, +0.1875] |
| acceptable_set_log_loss | 0.3772 | 0.2595 | -0.1178 [-0.3328, +0.1354] |
| single_label_multiclass_brier | 0.2121 | 0.1123 | -0.0998 [-0.2025, +0.0129] |

| Equal-task macro metric | Base | Trained | Change |
|---|---:|---:|---:|
| accuracy | 0.8438 | 0.9219 | +0.0781 |
| acceptable_set_log_loss | 0.3772 | 0.2595 | -0.1178 |
| single_label_multiclass_brier | 0.2121 | 0.1123 | -0.0998 |

| Complete-group accuracy: every row correct | Base | Trained | Change | Groups |
|---|---:|---:|---:|---:|
| All evaluated members correct | 0.7500 | 0.8438 | +0.0938 | 32 |

For two-row groups, this measures both members correct. The full group-size distribution is recorded in report.json.

| Subgroup | Rows / groups / tasks | Accuracy, base → trained | Set log loss, base → trained |
|---|---:|---:|---:|
| other | 64 / 32 / 16 | 0.8438 → 0.9219 | 0.3772 → 0.2595 |

Single-label multiclass Brier is the sum over options and ranges from 0 to 2. Multiple-acceptable-answer rows are excluded. These option distributions are not automatically calibrated success forecasts.

Intervals use 1,000 paired resamples of complete groups (seed 41). Rows sharing a world stay together. Macro means are descriptive. Intervals cover world-sampling variation, not training-seed variation, task selection, or pretraining contamination.
