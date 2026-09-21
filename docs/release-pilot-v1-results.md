# First release continuation: useful forecast gains, no promotion

The nine-billion-parameter pilot trained for 80 updates and stopped at its
predefined two-check limit. The original supervised checkpoint remains selected;
the trained adapters are preserved for analysis. The demo did not change.

| Development measure | Original | Update 40 | Update 80 |
|---|---:|---:|---:|
| Tool accuracy, equal weight per server | 86.87% | 86.66% | 86.58% |
| Tool log loss | 0.43247 | 0.37736 | 0.35558 |
| General accuracy, equal weight per task | 87.24% | 86.63% | 86.83% |
| General log loss | 0.34015 | 0.34358 | 0.34579 |
| Outcome expected Brier error, equal weight per group | 0.34093 | 0.27268 | 0.25220 |
| Outcome log loss | 0.59314 | 0.47696 | 0.44176 |
| Application-decision accuracy, 24 questions | 66.67% | 75.00% | 75.00% |

Lower Brier error and log loss are better. General retention and all four declared
product accuracy slices passed. Average outcome Brier error improved by about 26%,
but the required improvement was not consistent across outcome groups. At update
80, application-change Brier error rose from 0.001881 to 0.002619 and log loss rose
from 0.019838 to 0.024918. Task-completion forecasts improved on both measures, as
did the separate report mechanism. The tool-accuracy improvement requirement of
three percentage points also failed. Better average probability scores do not
override these gates.

The development cohort contained 2,217 tool questions across 27 server groups,
3,465 general questions across 157 tasks, 366 outcome questions across three
groups, and 24 application decisions. These are development results. Fresh
reserved tool transfer was not scored; no final-transfer claim follows. The
small application slices and original base-model data exposure limit inference.

Actual learning consumption was **5,120 presentations of 5,120 unique tokenized
questions**, totaling **3,507,173 input tokens**: 2,560 general replay questions,
2,240 tool choices and 320 verified questions. There were no uncommitted backward
presentations. The separately recorded startup diagnostic used two examples and
zero optimizer updates. The 371,278-question release corpus remains prepared
data; this pilot did not train on all of it. Its consumed rows inherited 4,802
ownership groups, not 4,802 independently executed worlds. No new worlds or
counterfactual branches were executed during this training run.

The real H200 checks passed: short/long left-padding agreement, a 4,090-token
backward pass padded to 4,096, finite gradients, and an intentional restart after
the first update. A fresh process restored the exact trainable tensor hash,
496 optimizer states, random-number state and schedule cursor. The original
43,278,336 trainable adapter scalars were optimized; their collective hash changed. This was internal
adapter training with mixed supervised targets, not online reinforcement learning.

Artifacts were recovered and hash-verified before the rental was deleted. An
independent checker recomputed every saved development aggregate from predictions
and checked the complete committed sequence against its frozen schedule. Estimated
compute cost was $3.36, excluding storage; provider billing had not yet posted the
new rental when the next budget reconciliation occurred, so a $6 conservative
allowance was retained.

The follow-up is [coverage-balanced training](release-balanced-v1-protocol.md),
prepared prospectively after the first check. At that check the model had seen
only four application-change examples and seven task-success forecasts. Tools
with large menus improved more than equal-server accuracy. The next recipe
rotates task/server/question-kind groups and increases verified-data exposure,
while keeping the same original parent, development cohort and acceptance gates.
This is a practical revision based on development evidence, not a causal proof
that sampling caused the first result.

Frozen pilot identity:
`beb9e125591bc4ab8a7d829003e30e2f032e9c974d9c7513678ac06628c950dd`.
Recovered archive:
`6edd722cb5c05e60513d283c7dd72434afcd82ec4c429a36b69703fbc8c72cb9`.
The exact starting model, loss, schedule and thresholds are in the
[pilot protocol](release-pilot-v1-protocol.md). Raw questions, protected application
traces, optimizer checkpoints and individual predictions remain private.
