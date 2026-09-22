# The mixed-mechanism pilot finished without a better decision policy

All six nine-billion-parameter training arms completed eight accepted updates.
They stopped at the predeclared two-check limit because none improved both
verified decision return and consequence forecasts. The original supervised
step-2742 checkpoint remains selected. No release or scaling gate passed.

The independent audit reproduced every development measurement from predictions
and execution receipts, matched actual consumption to the frozen schedules,
reconstructed stopping and checkpoint selection, and checked the saved adapter
arrays. All 496 language adapter tensors changed in every trained arm. The
selected adapters are exactly the unchanged original tensors. This was actual
language-network training; its modest changes did not improve the tested decisions.

| Method | Seed suffix | Decision return | Task success | Expected Brier error | General accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original, same baseline in all arms | — | -0.03444 | 50.0% | 0.419792 | 86.84% |
| Forecast supervision | 24 | -0.03444 | 50.0% | 0.416859 | 87.00% |
| Forecast supervision | 25 | -0.03444 | 50.0% | 0.416784 | 87.31% |
| Reward only | 24 | -0.03444 | 50.0% | 0.418575 | 86.52% |
| Reward only | 25 | -0.03444 | 50.0% | 0.418767 | 86.84% |
| Combined | 24 | -0.03444 | 50.0% | 0.417861 | 87.00% |
| Combined | 25 | -0.03444 | 50.0% | 0.417925 | 87.00% |

Lower Brier error is better. Forecast supervision reduced it by about 0.003;
reward-only training by about 0.001; combined training by about 0.002. The required
reduction was 0.02, together with a return gain of 0.03. Decision return and success
were unchanged in every evaluation. General retention remained within its
predeclared bounds, and no update breached the policy-divergence guard. These
small descriptive differences do not establish one method as generally superior.

The development cohort is the same exposed report mechanism: 36 executed decision
cases and 308 forecast questions with complete public tool descriptions. General
retention has 622 questions. This is not fresh mechanism transfer or broad evidence
of calibration. Reserved TOUCAN/telecom evaluation remains unopened.

## Prepared data and actual learning

The pool contains 5,654 verified forecast questions across seven groups, 268
shell/application case variants and 4,096 general replay questions. A large pool
did not mean this short experiment consumed all of it.

| Actual use across the six separate arms | Presentations | Distinct recorded question IDs | Repetitions |
| --- | ---: | ---: | ---: |
| Forecast supervision | 448 | 219 | 229 |
| Live actor transitions | 660 | 314 | 346 |
| General replay | 1,536 | 487 | 1,049 |

There were 48 physical optimizer attempts, all accepted. No incomplete or rejected
learning presentation is hidden in these counts. Gradient diagnostics are counted
separately in the audit. Matching questions deliberately repeat across comparison
arms; these totals are not one model's training history. Each forecast/combined
arm consumed 112 forecasts and each arm consumed 256 replay presentations.

The four reward-bearing arms executed 288 training episodes: 160 shell/application,
64 retail and 64 revisioned database episodes. Their 660 selected commands are the
actor transitions above. The shell/application episodes cover 73 case variants
from 32 underlying world/goal tasks, six authored root groups and five mechanisms.
Retail covers 15 world/goal pairs and 30 world/goal/cost cells. The new database
mechanism covers four world/goal tasks and 12 cost cells. These are different
counting units; episodes and wording variants are not new mechanisms.

The database forecast pool was prepared earlier from counterfactual executions;
this run did not create another copy of those branches. During reinforcement
learning, the current model chose commands, real workers executed them, and
independent goal checks supplied rewards. Existing physical worlds, verified
outcomes, source ownership and train/evaluation splits were preserved.

## Interpretation and next experiment

This recipe did not improve decision performance. It does not establish that
reinforcement learning is ineffective, that larger models are needed, or that
more copies of these templates would help. Each arm took only eight small updates,
and forecast-bearing arms consumed 112 questions from a 5,654-question pool.
The decision gate tested a separate exposed report mechanism; we did not record
matched before/after decision evaluations on every exercised training mechanism.
Consequently this run cannot distinguish weak learning within those mechanisms
from weak transfer out of them.

The next local work should prepare a fixed learnability diagnostic for the
existing executed mechanisms, keep it separate from transfer and release tests,
and predeclare a bounded dose comparison. Use matched world, goal and cost
coverage; measure the same decision/forecast questions before and after training.
Keep general replay, explicit checkpoint lineage and unchanged release gates.
A meaningful learning signal should precede another scale-up. The earlier
coverage-balanced supervised run already improved forecasts substantially while
failing its tool-accuracy gate; do not repeat that experiment under a new name.

## Recovery and cost

All 722 archived files passed recovery hashes before the H200 was stopped and
deleted. The archive is 1,949,977,321 bytes, with SHA-256
`25d1a45defe331273b67180dd34ddb257d6af5eef5f841254471af42f318374a`.
Estimated compute was $4.55 excluding storage; provider billing was still delayed.
The entire $30 authorization hold remains retained within the original budget.
No GPU rental remains active.

The local audit loaded tokenizers and saved adapter arrays, not the foundation
model. It stayed under 1 GiB process-group memory with no added swap. It verifies
recorded executions and saved weights; it does not replay optimizer numerics or
reexecute the tools. Three corruption-focused auditor tests passed.

[Prospective protocol](revisioned-pilot-v1-protocol.md) ·
[Original launch readiness](revisioned-pilot-v1-readiness.md) ·
[Independent aggregate audit](../results/revisioned-pilot-v1/audit.json)
