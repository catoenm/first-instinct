# Paired questions from the competing-writer executions

The data interface now supplies **138 canonical candidate questions** from the
qualified database receipts, with 138 reversed-menu diagnostics. No environment
was rerun and no model was loaded or trained. All presentations retain complete
information in both the pinned Qwen and native Laya formatters.

| Question | Canonical candidates |
| --- | ---: |
| Next command under a displayed fixed continuation | 6 |
| Best of eight explicitly listed procedures | 6 |
| Goal status immediately after a command | 36 |
| Command return code | 36 |
| Goal status after a specified procedure | 48 |
| Read-first utility versus four named alternatives | 6 |

These are **six initial visible contexts, four world-and-goal tasks and one
authored mechanism**, derived from the existing 96 primary branches. Costs and
questions do not create additional independent tasks. All related schedules,
goals and menu permutations share one training-candidate ownership group.
None has been admitted to a training mixture or consumed by an optimizer.

The 120 outcome questions include **54 uncertain distributions**. For each
question, both compatible worlds share exactly the same visible history and
contribute probability one half. The 96 execution replays are audited but add
no extra probability mass. The other 18 questions use sets of best choices
within their stated comparison, rather than treating action preferences as
probabilities of successful outcomes.

Immediate goal forecasts inspect the actual snapshot after the first command
in an existing trace. They do not fabricate a stop action or a new terminating
branch. Procedure forecasts use the recorded terminal state. Return-code
forecasts use the actual command response. This keeps command success, goal
success and success after a continuation distinct.

The next-command question specifies a common continuation of read, conditional
write and finish, stopping immediately if the episode terminates. This is a
hypothetical continuation, not a recommended policy; refreshing and writing can
violate the revision-limited goal. Likewise, the observation-cost label compares
only the named procedures. Neither label claims an unrestricted optimal action
or globally optimal value of information.

The renderer preserves the original public state exactly and rejects private
fields. A separate audit recomputes outcome distributions and procedure values
from executed states and costs. Eight corrupted-copy controls were rejected:
changed outcomes, choices, world weights, replica ownership, private fields,
question meaning, answer order and missing coverage. Three focused unit tests
also passed; they performed no database execution.

All **276 token presentations** pass native-format parity and retain full
information. The largest Qwen input is 765 tokens, below its existing 4,096 limit;
the largest Laya input is 605 tokens, within its unchanged native limits. No
description was clipped and no input was selectively dropped. The CPU guard
completed in about 3.20 seconds, with sampled peak memory below 562 MiB and no
additional swap.

The [protocol](revisioned-questions-v1-protocol.md) and
[aggregate evidence](../results/revisioned-questions-v1/summary.json) preserve
these boundaries. Next comes mixture admission and a live decision interface
whose reward follows the model's actual continuation, plus broader independently
verified situations. This small authored set alone does not justify a larger
training run or demonstrate progress on model capability.
