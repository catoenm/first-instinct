# Recovered real-tool experiment: no new checkpoint selected

The rental is closed and all artifacts have been recovered. The original
supervised step-2742 checkpoint remains selected. This is an **incomplete
comparison**: the first forecast-only and reward-only arms completed, the first
combined arm reached its fixed time limit, and the second seed never started.
There is no production training or release promotion from this run.

| Evaluated checkpoint | Accepted updates at evaluation | Development return | Expected forecast Brier error | General accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original supervised parent | 0 | 0.2339 | 0.6410 | 86.84% |
| Forecast-only final | 16 | 0.2339 | 0.6293 | 86.84% |
| Reward-only final | 8 | 0.1783 | 0.6416 | 87.00% |
| Combined last evaluation | 8 | 0.2600 | 0.6388 | 86.84% |

Higher return and lower Brier error are better. The forecast-only arm stopped
after two checks without joint improvement. Reward-only crossed the predefined
return-regression guard. Combined was safe at its last evaluation but did not meet
the joint requirement of at least 0.03 return gain and 0.02 Brier reduction. No
threshold was changed after observing these results.

The combined arm accepted ten optimizer updates before its 48-minute limit. During
update eleven it collected another sixteen episodes and completed ten policy
backwards, then discarded that partial update before any optimizer step. Its
saved ten-update diagnostic checkpoint has **no task evaluation**; the table reports
update eight. The pipeline then stopped, leaving the second seed unrun.

## Scope and consumption

Decision development covers 36 variants of **four underlying world/goal tasks in
one report-workflow mechanism**, with one authored root group. All 308 forecast
questions also belong to this mechanism. General retention uses 622 questions
with equal-task weighting. These are exposed development measurements, not a broad
transfer or public release benchmark.

| Arm | Collected training episodes | Collected actor transitions | Policy presentations in accepted updates | Forecast presentations in accepted updates | General replay presentations in accepted updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forecast-only | 0 | 0 | 0 | 448 | 1,024 |
| Reward-only | 128 | 302 | 302 | 0 | 512 |
| Combined, partial | 176 | 456 | 402 | 280 | 640 |

Combined collection contains 110 shell/application episodes and 66 retail resets.
Its shell component covers 36 world/goal tasks, 110 case variants and five
mechanisms across six authored root groups. Retail covers 20 goal/world pairs and
61 cost cells. The 456 transitions contain 347 distinct token inputs. Forty-four
transitions from the final collection were never backpropagated; ten more were
backpropagated only in the discarded partial update. Those are not accepted
learning presentations.

The combined forecasts have 280 unique question identities. Its 640 replay
presentations contain 598 unique questions and 42 repetitions. The forecast-only
and reward-only counts, including their unique/repeated breakdowns, are documented
in the [interim report](canonical-action-v1-interim.md). Value and entropy terms
share the policy backward; they are not extra independent examples. Diagnostic
backwards and evaluation executions are counted separately from training.

Prepared pools contain 2,510 verified training forecasts, 268 shell case variants
across five mechanisms, and a 4,096-question general replay pool. Prepared pool
size does not imply that all examples were consumed. Retail reuses existing goals,
worlds and costs; extra resets do not create new independent task families.

## What we verified

The complete archive contains 712 hash-verified files. The audit reuses completed
arm checks only after exact receipt hashes match, and separately reconstructs the
combined arm's visible inputs, earned rewards, world/cost schedules, consumed
prefix, discarded tail, probability metrics, retention and checkpoint gates.
Recorded unchanged-policy checks have exactly zero probability drift across 302
reward-only and 456 combined training transitions.

Saved adapters contain 496 trainable tensors with 43,278,336 elements. Every
evaluated updated adapter has changed tensors throughout the language network;
this was actual network adaptation through low-rank adapters, with the original
base matrices frozen. Every selected `best` artifact is byte-identical to the
original parent. The interrupted update's restoration is verified from transaction
receipts and saved arrays; the auditor does not reexecute GPU optimizer state.

This result does not establish that reinforcement learning cannot work. The
reward-only regression came from one final choice switching between two target
locations in one report case. The development sample is too narrow for a general
conclusion. Neither a larger model nor more presentations of the same templates
addresses that limitation by itself.

## Next decision

Run a small, separately frozen comparison against the public Laya typed-decision
checkpoint on these already exposed forecasts, retaining native calibration and
full input information. This can show whether our adaptation improves on the
foundation and whether a much smaller decision model is competitive on this
specific workflow. Broader executable mechanisms and fresh transfer checks remain
necessary before a production run.

The GPU was deleted after recovery. Estimated compute for this rental was $9.90,
excluding storage. The conservative budget retains the entire original stage
reservation; it does not treat the smaller observed bill as new authorization.
