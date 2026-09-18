# Paired decision-model evaluation

All changes below are trained minus base. Positive accuracy and negative losses indicate their respective directions; the report does not declare an overall winner.

7,048 paired rows, 5,048 world/document groups, 14 tasks. 184 rows have multiple acceptable answers.

| Row-weighted metric | Base | Trained | Change [95% group-bootstrap interval] |
|---|---:|---:|---:|
| accuracy | 0.7765 | 0.7765 | +0.0000 [+0.0000, +0.0000] |
| acceptable_set_log_loss | 0.6218 | 0.6218 | +0.0000 [+0.0000, +0.0000] |
| single_label_multiclass_brier | 0.3325 | 0.3325 | +0.0000 [+0.0000, +0.0000] |

| Equal-task macro metric | Base | Trained | Change |
|---|---:|---:|---:|
| accuracy | 0.7758 | 0.7758 | +0.0000 |
| acceptable_set_log_loss | 0.6238 | 0.6238 | +0.0000 |
| single_label_multiclass_brier | 0.3361 | 0.3361 | +0.0000 |

| Complete-group accuracy: every row correct | Base | Trained | Change | Groups |
|---|---:|---:|---:|---:|
| All evaluated members correct | 0.7407 | 0.7407 | +0.0000 | 5048 |

For two-row groups, this measures both members correct. The full group-size distribution is recorded in report.json.

| Subgroup | Rows / groups / tasks | Accuracy, base → trained | Set log loss, base → trained |
|---|---:|---:|---:|
| environment | 1000 / 1000 / 2 | 0.7950 → 0.7950 | 0.5289 → 0.5289 |
| public | 2048 / 2048 / 4 | 0.8843 → 0.8843 | 0.3334 → 0.3334 |
| verified | 4000 / 2000 / 8 | 0.7167 → 0.7167 | 0.7927 → 0.7927 |

Single-label multiclass Brier is the sum over options and ranges from 0 to 2. Multiple-acceptable-answer rows are excluded. These option distributions are not automatically calibrated success forecasts.

Intervals use 1,000 paired resamples of complete groups (seed 41). Rows sharing a world stay together. Macro means are descriptive. Intervals cover world-sampling variation, not training-seed variation, task selection, or pretraining contamination.
