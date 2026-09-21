# Real-tool comparison: prepared data and local checks

The next diagnostic uses the original supervised Qwen3.5-9B adapter. The earlier
history continuation remains unpromoted. The [prospective protocol](live-tools-pilot-v1-protocol.md)
compares forecast supervision, executed-reward policy learning, and their combination
with two paired seeds. Successful local preparation is not evidence of model gains.

The input package contains 2,510 unique verified forecast questions:

| Training mechanism | Questions |
| --- | ---: |
| Configuration repair | 308 |
| SQLite repair | 308 |
| Application message delivery | 468 |
| Filesystem scope | 224 |
| Reservation transactions | 972 |
| Retail workflows | 230 |

Retail includes 190 previously admitted history forecasts and all 40 newly audited
mutating-history branches. Their labels still concern the named command followed
by stopping. Exact-token checks found no collision with the prepared development
or reserved evaluation inputs. All retail worlds remain in one connected training
group; the 230 questions are not 230 independent tasks.

The live curriculum reuses 268 shell/application case variants in five mechanisms.
Retail adds three existing goals over ten physical initial states, twenty goal/world
combinations and six fee pairs. A completed reward arm traverses a 144-presentation
balanced retail schedule with 120 distinct goal/world/fee cells. Across the mixed
curriculum it can execute at most 384 training episodes. Each completed arm has
1,536 scheduled general-replay presentations from a 4,096-question pool; a forecast
arm additionally has 672 scheduled forecast presentations. Actual consumption comes
from execution receipts and the backward ledger, including early stops.

Seventeen tests passed inside the extracted exact package. They check real local
filesystem/SQLite execution, both action interfaces, all three objectives, critic
separation, stale-policy rejection, altered receipt rejection, schedule ownership,
paired cost distributions, and transactional rollback. Separately, five earlier
shell/application receipts reconstructed fifteen transitions under the new adapter.
Those historical trajectories lack the new policy identity and remain ineligible
for optimizer training. Four fresh retail runtime checks executed seven actual
calls and eleven actor turns, with independent terminal verification and no model.

The first local test attempt correctly refused the Linux-only shell backend on the
Mac. The native fixtures then exposed two integration defects: legacy action rows
carried an unused first-option target, and the new receipt adapter needed explicit
probability validation. Both were corrected before freezing. The failed attempts
remain recorded. No existing frozen trainer or earlier result was edited.

Data freeze: `06914d15fd54344325815cc595461929d66b53723811db0c5c9d1f865cbfbfe2`.
The exact 937-file package is 306,316,668 bytes, with archive digest
`b356ba2afb713a31ce6e58b6108aafb54b136694b82608f2a638704746a79dc4`.
The largest local qualification process group stayed below one GiB, with no added
swap. No foundation model was loaded on the Mac.

The first cloud attempt passed the Linux workers but failed the unchanged-weight
action probability check before any optimizer update. Its complete failure archive
was recovered before deleting the rental. The separately frozen
[action-forward correction](canonical-action-forward-v1-protocol.md) preserves this
data, the learning recipe, the parent model and the original numerical tolerance.
Twenty-two tests passed in the corrected package.

The corrected path subsequently passed on the actual nine-billion-parameter model:
sixteen freshly executed episodes produced thirty-nine action transitions, with
exactly zero probability difference when rescored for learning. Every sampled-action
likelihood ratio was one. The actor and forecast objectives produced nonzero language
gradients; the value objective affected only the detached critic. All fifty-five
guard inputs also agreed before an update. These checks performed zero optimizer
steps. They establish a usable learning interface, not improved decision quality.

Rescoring the previous forty-one inputs separately reproduced a maximum probability
difference of 0.02265 with the old collection path. That is a new diagnostic measurement,
not the unsaved maximum from the failed assertion. Those old inputs never become
optimizer rollouts. This does not establish the cause of earlier transfer failures.

All four execution-parity stages passed, and the corrected six-arm pipeline entered
its first arm on one H200. The corrected rental has a 6.5-hour maximum and a $38 cap,
with $2 retained for the first attempt inside the same original $40 stage allocation.
Independent provider shutdown and artifact recovery are configured. This document
records qualification and launch, not completed training. The independent release
scores remain unopened; deployment still requires fresh transfer and broader retention.

After recovery, `python -m tool_lab.live_pilot_accounting --run <recovered-run>
--data <frozen-data> --output <new-report.json>` separates diagnostic backwards,
physical optimizer attempts, accepted or rolled-back presentations, repeated question
IDs and collected trajectories. It also reports distinct actor token inputs and
retail goal/world/cost cells without calling them independent tasks. This is an
accounting check; it does not replace execution verification or metric auditing.
