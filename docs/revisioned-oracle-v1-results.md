# An exact decision target reveals a concrete training weakness

We can now grade every next action in the competing-writer database environment
against the best policy that uses only public observations. Applying this new
oracle to the completed pilot's recorded decisions found a clear weakness: its
greedy first choice was suboptimal in all six starting goal/cost situations.
The original release checkpoint and the completed pilot's result remain unchanged.

| Goal | Costs | Optimal first command | Recorded greedy first command |
| --- | --- | --- | --- |
| Increment the latest value | Cheap inspections/writes | Read current state | Write cached value |
| Increment the latest value | Expensive inspection | Revision-checked write | Write cached value |
| Increment the latest value | Expensive writes | Stop | Read current state |
| Edit only the approved revision | Cheap inspections/writes | Read current state | Write cached value |
| Edit only the approved revision | Expensive inspection | Revision-checked write | Write cached value |
| Edit only the approved revision | Expensive writes | Stop | Read current state |

These are six public starting contexts for four underlying world/goal tasks in
one existing mechanism. Across four learning arms, they produced 64 initial-state
visits and 32 distinct history/update distributions. None had an optimal greedy
choice; sampling selected an optimal command on 10 of the 64 visits. Those visits
are repetitions, not 64 independent problems. Recorded menu positions varied, so
the pattern is not simply always selecting the first slot; this does not prove
order invariance.

The errors have different costs. With cheap operations, blindly writing the
cached value has expected utility −1: it can overwrite the colleague's change.
Reading first and then acting optimally yields 97 for the increment goal and
97.5 for the revision-limited goal. With expensive writes, an unnecessary read
loses only two units relative to stopping. Exact-action accuracy therefore needs
the accompanying regret measure. These numbers are abstract reward units, with
100 for goal completion, not success probabilities.

## Where the labels come from

The oracle reuses 1,422 previously executed terminal paths. It reconstructs
1,692 distinct world/history/action branches at 258 public histories, including
24 where observations still cannot distinguish the two hidden worlds. Every
offered command has a recorded outcome in every compatible world. Repeated
prefixes and the existing execution replays add no probability mass.

Backward induction computes exact rational expected returns with all future
costs and the remaining four-decision horizon. A separate audit enumerates
91,608 combinations of recorded continuation paths. It rejects 76,350 combinations
that would select different actions at the same public history in different
hidden worlds. The remaining public policies reproduce every first-action value
and every optimum exactly. A policy cannot secretly inspect the writer schedule.

This is stronger than the earlier comparison of four named procedures. Its
optimality still applies only to this six-command environment, finite horizon
and declared equal world prior. It does not establish an optimal strategy for
arbitrary software tasks or a model's own future continuation.

The fixed diagnostic panel contains:

| Question | Canonical questions |
| --- | ---: |
| Optimal next command | 258 |
| Final goal outcome after a named command and optimal public continuation | 1,548 |
| Whether read-first beats the best other first command after all costs | 258 |

The 2,064 canonical questions also have 2,064 reversed-menu checks. All 4,128
presentations retain complete information in the pinned Qwen formatter; the
largest is 806 tokens. There are 66 legitimately uncertain continuation forecasts.
Action targets remain acceptable-choice sets, separate from outcome probability
targets. This environment produced no exact optimal-action ties, though the
representation supports them.

The forecast continuation is explicit: after the named first command, choose the
highest expected remaining-utility action from public history, breaking exact
ties by ascending command identifier. Labels follow actual recorded outcomes
under that policy. They do not predict the learned actor's continuation or an
immediate command response. The inspection question is the net advantage of
read-first over every other offered first action, not an unrestricted claim
about the value of any possible information.

## What the recorded model choices tell us

The post-hoc analysis covers all 92 database actor transitions in the four
reward-bearing arms, across their 64 episodes. After deduplicating repeated
histories at the same update, optimal greedy-choice rates ranged from 1/13 to
5/16. Those denominators describe different visited-state mixtures and cannot
rank the methods. Behavior-weighted first-action regret ranged from 37.1 to 41.9
raw reward units; part of that regret comes from the declared 20% uniform
exploration even if the native policy were optimal.

There were ten within-arm repeated-history pairs across the four arms, each
separated by six completed updates. None changed from a nonoptimal greedy choice
to an optimal one or the reverse. Average probability mass on the optimal action
rose slightly in each arm; expected first-action regret changed in mixed directions.
These are visitation-selected pairs, with two or three pairs per arm, not a
representative learning curve. An update-8 trajectory uses weights after update7;
it is not an evaluation of the saved update-8 checkpoint.

This evidence points to action learning within the exercised database mechanism
as a problem worth testing, alongside transfer. It does not identify the cause
of the cached-write preference. A matched fixed-panel evaluation of the original
foundation, supervised parent and trained checkpoints is still needed to separate
starting capability, fine-tuning effects and the small reinforcement-learning dose.

The next bounded learning design should include an execution-verified optimal
decision-supervision control, general replay and a meaningful training dose,
with fixed before/after decisions and consequence forecasts. The new panel is
training-owned; fitting it would demonstrate learnability, not unfamiliar-task
generalization. Keep the existing transfer and release gates separate and intact.
No new rental or training run was started by this data qualification.

## Verification and accounting

Five oracle tests and ten corrupted-copy checks passed, including hidden-world
policy leakage, wrong values, changed continuation, outcome mass, private state
and ownership. Two recorded-choice tests check likelihoods, exploration costs
and rejection of model-supplied targets. Qualification took about 8.4 seconds
under the local guard, peaked below 0.81 GiB, and added no swap. The recorded-choice
analysis also completed locally without a foundation-model load.

This stage executed zero new worlds or counterfactual branches, made zero model
calls, and consumed zero optimizer presentations. The new questions are diagnostic
artifacts and are not yet admitted to a training loss. Entire-world ownership,
the original selected checkpoint, reserved transfer data and compute authorization
are unchanged. No GPU is active.

Oracle freeze: `f95a345d1825757eef78094cab85e399f91e892fb84fb3cbbe80245b29f3767e`.
Recorded-choice diagnostic freeze:
`d298c7629cad7fade68ee660c1ca0df4c5b599ab95cd6f401d8268539cd2f1d5`.

[Oracle protocol](revisioned-oracle-v1-protocol.md) ·
[Recorded-choice protocol](revisioned-oracle-diagnostics-v1-protocol.md) ·
[Qualification and diagnostic aggregates](../results/revisioned-oracle-v1/) ·
[Completed learning comparison](revisioned-pilot-v1-results.md)
