# A qualified live learning path across mechanisms

The next controlled 9B experiment is ready to run. The new application worker
executes the model's chosen tool, returns its actual observation, and independently
checks the resulting database and reward. It supports failed-send recovery,
prerequisites, duplicate prevention, and stopping when the goal is already met.
It runs the pinned ToolSandbox simulation in a separate Python process; it does
not call real messaging services.

[Protocol](mixed-decisions-v1-protocol.md) ·
[Frozen inputs and local evidence](../results/mixed-decisions-v1-preflight/)

## What is prepared

| Quantity | Count |
| --- | ---: |
| Mechanisms / independent fixture roots | 5 / 5 |
| Initial shell file worlds / application database worlds | 8 / 9 |
| World-and-goal tasks | 34 |
| Goal, evidence and cost context cases | 228 |
| Consequence questions: training / validation / transfer | 2,520 / 504 / 504 |
| Paired decision and observation-value questions, diagnostic only | 228 |
| General replay pool, reused unchanged | 4,096 |
| 9B optimizer presentations at this preflight snapshot | 0 |

The underlying accepted collections executed 3,024 shell branches and 2,268
distinct application alternatives. The application alternatives were each
replayed once, for 4,536 executions. These are execution counts, not counts of
independent tasks. The new learning schedule preserves the conflicting labels
of genuinely indistinguishable hidden worlds instead of discarding uncertainty.

Configuration, SQLite repair, and application delivery train the model. Reporting
is held out for validation, and atomic artifact publication is held out for the
final transfer check. The original supervised 9B weights initialize forecast-only,
reward-only Proximal Policy Optimization, and combined learning for two seeds.
All arms receive the same general replay. The demo stays on its existing checkpoint.

## Checks that passed

The application worker reproduced **138 selected alternatives and 1,062 actual
tool calls** on Linux, matching the saved observations, database states and
outcomes exactly. Two workers kept their databases separate. Episode, command
and request limits rejected excess work before execution.

The shell catalog reproduced **252 alternatives and 744 commands** under the
unprivileged Linux execution path. The first comparison matched all branches
but failed during parent-side cleanup because the test container lacked file
cleanup permissions. It was not accepted as a qualified runtime. A separate
corrected run passed both comparison and cleanup; both receipts remain available.

The final mixed learning check used identical random tiny-network starts across
all three arms. The reward and combined arms each executed four episodes and
13 decisions. Actor gradients reached the language network; critic gradients
into that network were zero. Forecast and replay gradients were nonzero. All
three ordinary optimizer steps passed the guard. Separate adversarial tests
proved that a rejected step restores both parameters and optimizer state while
retaining its attempted-training receipts. **These checks establish working
learning mechanics, not improved 9B performance.**

Repeated-read stress tests executed eight episodes and checked 52 real actor
inputs. The longest reached **2,134 tokens**: ten exceeded 1,536, and two exceeded
2,048. The pilot therefore uses 4,096 tokens and rejects longer histories without
truncating them. This sample does not prove that every reachable history fits.

## Engineering work is counted separately

The live application tests consumed 25 top-level calls. The corrected early
engineering chain consumed ten. Two application-only tiny-network checks used
27 calls each. Three mixed-network checks each used eight live episodes, twelve
shell commands and 38 application calls; intermediate repetitions are preserved
separately. Context stress used 32 shell commands and 57 application calls.
The shell qualification has two separate 252-branch executions as described
above. These repetitions create no new independent tasks or training labels.

Two earlier application Linux startup failures executed no task calls: an image
cache collision, then a non-executable temporary mount. A preparation attempt
also rejected an incorrect split-name assumption before encoding or training;
the corrected preparation preserves the original `transfer` ownership.

The public preflight snapshot includes authored execution evidence, schedules,
source hashes and checkpoint lineage. General replay/evaluation pools are
identified by their existing hashes. Future consumption and results must come
from the actual run ledgers, not these prepared counts.

The main limitation remains the small number of mechanisms and roots. This pilot
can test whether the learning recipe is stable and transfers to one reserved
workflow. It cannot establish a general-purpose Jev reproduction. Dynamic command
proposals and noisy conflicting current sources remain future data work.
