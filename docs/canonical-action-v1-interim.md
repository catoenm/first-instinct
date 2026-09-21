# First two completed arms: no new checkpoint selected

This is an interim audit of seed 20260924 only. The combined arm and the second
seed have not completed in this report. It is not a paired comparison, a release
evaluation, or evidence that reinforcement learning cannot help.

The corrected action runtime passed the actual nine-billion-parameter preflight.
The first reward arm also recorded exactly zero unchanged-policy probability drift
over all 302 training transitions. That resolves the specific collection-versus-
learning numerical mismatch encountered in the first preflight; it does not prove
that every optimization choice is correct or explain historical transfer failures.

| Checkpoint | Accepted updates | Development reward | Expected outcome Brier error | General retention accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original supervised parent | 0 | 0.2339 | 0.6410 | 86.84% |
| Forecast-only, first check | 8 | 0.2883 | 0.6370 | 86.84% |
| Forecast-only, final check | 16 | 0.2339 | 0.6293 | 86.84% |
| Reward-only, final check | 8 | 0.1783 | 0.6416 | 87.00% |

Lower Brier error is better. The forecast-only arm missed the joint improvement
requirement twice and stopped after sixteen updates. The reward-only arm breached
the reward-regression safeguard and stopped after eight. Both retain the original
supervised step-2742 checkpoint as their selection. No gate was relaxed.

## What these checks cover

Decision evaluation consists of 36 report-workflow case variants: four underlying
world/goal tasks, one authored root group and one mechanism. The 308 consequence
forecast questions also belong to that report mechanism. They are exposed
development data. General retention covers 622 questions with equal-task weighting.
These counts must not be presented as hundreds of independent environments or a
broad unfamiliar-mechanism result.

The reward-only regression is one case changing from successful completion to an
incorrect final update. The model observed the same history and paid the same
cost; its final choice changed between two target locations. This lowered that
case's return by two, producing the aggregate decrease of 2/36. A small change
across an action-selection boundary can produce this result. The guard still
applies, but the sample does not support a sweeping conclusion about the method.

## Actual consumption

| Arm | Executed training episodes | Policy transition presentations | Forecast presentations | General replay presentations |
| --- | ---: | ---: | ---: | ---: |
| Forecast-only | 0 | 0 | 448 | 1,024 |
| Reward-only | 128 | 302 | 0 | 512 |

The 448 forecasts are distinct question identities. Forecast-only replay used 922
unique questions and 102 repeated presentations. Reward-only replay used 485
unique questions and 27 repeated presentations.

Reward learning executed 80 shell/application episodes and 48 retail resets. The
shell/application component covered 33 world/goal tasks across five mechanisms and
80 case variants, with 177 actor transitions. Retail covered 20 goal/world pairs
and 46 cost cells, with 125 actor transitions. These are executions of existing
curricula, not newly invented task families. Across both components, 302 policy
presentations contained 291 question identities and 239 distinct token inputs.
Policy, value and entropy share one backward per transition; they are not three
independent data presentations. Diagnostic backwards and evaluation executions
remain outside the training counts above.

## Verification and limits

The independent receipt audit reconstructs visible actor inputs, tokenization,
earned rewards, scheduled worlds/costs, completed learning presentations, probability
metrics, retention and selection decisions. It checks the recorded native and
behavior policy guards. It does not rerun GPU optimization or tools, independently
inspect checkpoint tensors, or claim complete rental recovery. Only the two closed
arms' 30 reporting files were copied and hash-verified for this interim analysis.

During audit development, two readers required schema adaptation: the correction
receipt uses `runtime_correction_sha256`, and live actor targets are empty rather
than the older collector's unused `[0]` placeholder. The auditor requires empty
targets in real receipts and restores that placeholder only in a private copy
passed to the frozen older verifier. Original data and training code stayed
unchanged. Failed audit attempts remain recorded; the corrected audit passed.

The active comparison proceeds under its original bounds. Broader fresh-mechanism
evaluation remains necessary before claiming a generally useful decision engine.
