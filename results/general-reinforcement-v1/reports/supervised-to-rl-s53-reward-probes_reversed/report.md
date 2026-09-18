# Paired decision-model evaluation

All changes below are trained minus base. Positive accuracy and negative losses indicate their respective directions; the report does not declare an overall winner.

64 paired rows, 32 world/document groups, 16 tasks. 0 rows have multiple acceptable answers.

| Row-weighted metric | Base | Trained | Change [95% group-bootstrap interval] |
|---|---:|---:|---:|
| accuracy | 0.9219 | 0.9375 | +0.0156 [+0.0000, +0.0469] |
| acceptable_set_log_loss | 0.2595 | 0.3265 | +0.0670 [-0.0021, +0.1598] |
| single_label_multiclass_brier | 0.1123 | 0.1229 | +0.0107 [-0.0054, +0.0316] |

| Equal-task macro metric | Base | Trained | Change |
|---|---:|---:|---:|
| accuracy | 0.9219 | 0.9375 | +0.0156 |
| acceptable_set_log_loss | 0.2595 | 0.3265 | +0.0670 |
| single_label_multiclass_brier | 0.1123 | 0.1229 | +0.0107 |

| Complete-group accuracy: every row correct | Base | Trained | Change | Groups |
|---|---:|---:|---:|---:|
| All evaluated members correct | 0.8438 | 0.8750 | +0.0312 | 32 |

For two-row groups, this measures both members correct. The full group-size distribution is recorded in report.json.

| Subgroup | Rows / groups / tasks | Accuracy, base → trained | Set log loss, base → trained |
|---|---:|---:|---:|
| other | 64 / 32 / 16 | 0.9219 → 0.9375 | 0.2595 → 0.3265 |

Single-label multiclass Brier is the sum over options and ranges from 0 to 2. Multiple-acceptable-answer rows are excluded. These option distributions are not automatically calibrated success forecasts.

Intervals use 1,000 paired resamples of complete groups (seed 41). Rows sharing a world stay together. Macro means are descriptive. Intervals cover world-sampling variation, not training-seed variation, task selection, or pretraining contamination.
