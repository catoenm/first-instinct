# History pilot: better forecasts, no release promotion

The bounded Qwen3.5-9B supervised pilot stopped after 160 accepted updates because
two consecutive development checks found no eligible improvement. Later-tool
decisions and average consequence forecasts improved, but first-tool accuracy
missed the advancement requirement and a small state-change forecast group
worsened. The original supervised step-2,742 adapter remains selected. Neither
experimental checkpoint is a release candidate, and the run will not be extended.

| Development measure | Original parent | Update 80 | Update 160 |
| --- | ---: | ---: | ---: |
| First-tool accuracy, equal server weighting | 86.87% | 86.80% | 86.11% |
| Later-tool accuracy, equal server weighting | 72.85% | 74.06% | 74.42% |
| General accuracy, equal task weighting | 87.24% | 86.66% | 86.54% |
| Outcome expected Brier error, lower is better | 0.34093 | 0.25857 | 0.22776 |
| Outcome log loss, lower is better | 0.59314 | 0.45475 | 0.39792 |
| Small application decision diagnostic | 16/24 | 18/24 | 18/24 |

The development sets contain 2,217 first-tool questions across 27 server groups,
212 later-tool questions across 19 server groups, 3,465 general questions across
157 tasks, 366 outcome questions, and 24 application decisions. These are exposed
development cohorts, not new transfer evidence. The outcome score averages the
three predeclared groups; it is the existing expected Brier definition, not a
newly substituted binary squared-error scale.

At update 160, later-tool accuracy rose 1.57 percentage points and average outcome
Brier error fell 33.2%. General accuracy fell 0.70 points, within the one-point
retention limit; its log loss rose from 0.34015 to 0.34456, also within its limit.
First-tool accuracy fell 0.77 points instead of gaining the required three.
The application state-change forecast group's Brier error rose from 0.001881 to
0.002466 and log loss from 0.019838 to 0.022279. That small absolute regression
independently fails the frozen requirement that both probability scores must not
worsen in any outcome group. Product-slice and later-history retention passed.

## What the diagnostic says

Question-weighted first-tool accuracy actually increased: 1,895 to 1,918 correct
out of 2,217, with 59 corrections and 36 regressions. Eight server groups improved,
six declined and thirteen stayed equal. A seven-question group contributes to the
equal-server decline, but groups with 73 and 35 questions also lost seven and
three correct decisions. We preserve the prospective weighting and gate rather
than changing them after seeing this result. Improvement is uneven across tools.

Later-tool questions gained 17 correct decisions and lost nine. General questions
gained nine and lost 22. This does not establish broad capability growth or show
that more updates would resolve the tradeoff. Data, learning rate and preservation
regularization changed together; this was a practical recipe pilot, not a causal
ablation of later-history data.

## What was actually consumed

The maximum schedule prepared 20,480 presentations and 19,923 unique questions
for 320 updates. The early-stopped run consumed **10,240 presentations of 10,079
unique questions**, totaling **7,978,447 input tokens**, in 160 optimizer updates:

| Training pool | Presentations |
| --- | ---: |
| General replay | 5,120 |
| First-tool teacher examples | 2,560 |
| Later-tool teacher examples | 1,280 |
| Execution-verified questions | 1,280 |

The consumed history questions cover 1,036 normalized requests. The verified pool
contains 1,119 distinct questions and 161 repeated presentations. Across all pools,
8,712 inherited ownership groups are represented; they are not 8,712 independently
executed physical tasks. This pilot executed **zero new worlds or counterfactual
branches** and performed **no new online reinforcement learning**. It reused the
admitted tool traces and previously execution-verified labels.

Separately, 10,240 original-parent forward calls over scheduled general training
questions cached 3,184,538 tokens for the preservation penalty. That cache covers
the maximum schedule, not just consumed updates. Its predictions supply no new
truth labels. Runtime qualification performed four backward presentations without
optimizer updates. All 160 training attempts were accepted; none was rejected or
left uncommitted.

## Lineage, verification and recovery

The foundation remains `Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The run changed 496 internal adapter
tensors containing 43,278,336 trainable elements; the foundation matrices stayed
frozen. Saved-array checks found finite values and changes in all 496 tensors at
both experimental checkpoints. No foundation model was loaded on the Mac.

| Adapter | File SHA-256 |
| --- | --- |
| Selected original supervised step 2,742 | `882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a` |
| Experimental update 80 | `d02ebaad54ed5fbce8022484816df1e7984b75ab9297ce7d179da49641fd52be` |
| Experimental update 160 | `e70b93daa42a412b3e978f16c433747b731ec6f368531e551cabf18ce2f65ead` |

An independent CPU audit reconstructed every saved metric, group/slice score and
consumption count. It verified reference scope, recorded update probes and restart
receipts; it did not reexecute GPU optimizer steps. The largest recorded mean and
individual step divergences were 0.002294 and 0.024813, below the frozen 0.02 and
0.10 limits. Numerical failure did not cause the stop.

All 73 recovered files matched their manifest. The owned H200 rental was stopped
and deleted; estimated GPU compute was **$6.16**, excluding storage. The budget
continues to reserve this stage's full $25 and earlier billing/storage holds,
leaving **$192.64 conservatively unallocated from the original cumulative $500**.
There is no active training rental.

## Next experiment

The independent release evaluation is prepared but remains unscored. Its selector
correctly refused the completed pilot because no development checkpoint qualifies.
No reserved score will be used to rescue or tune these checkpoints.

The next learning stage is a separately qualified, identical-start comparison of
forecast supervision, reward-only Proximal Policy Optimization, and their
combination on real tool trajectories. The original selected parent may support
a bounded diagnostic under a new prospective protocol; this result authorizes no
automatic longer supervised run. First qualify matched world/cost sampling,
actor reward accounting, context bounds and independent mechanism evaluation.
Whole reserved families remain reserved. Measure decision return and consequence
probability quality together, with general-task retention.

[Prospective pilot protocol](history-pilot-v1-protocol.md) ·
[Independent audit](../results/history-pilot-v1/audit.json) ·
[Exact gates](../results/history-pilot-v1/summary.json) ·
[Saved adapter audit](../results/history-pilot-v1/weights-audit.json) ·
[Post-run diagnostics](../results/history-pilot-v1/development-diagnostics.json) ·
[Recovery receipt](../results/history-pilot-v1/recovery.json) ·
[Reserved evaluation readiness](release-evaluation-v1-readiness.md)
