# Bounded decision-learning capacity comparison

This tests whether the existing 9B model can learn the exercised database
mechanism. It is not a production run, fresh transfer study or release selection.
Preserve all earlier runs and the original supervised step-2742 release.

Use the qualified revisioned-oracle-v1 panel without generating new templates or
worlds. Explicitly admit copies of its 516 action/inspection questions and 1548
optimal-public-continuation forecasts to this training-owned experiment. Keep
their common ownership, exact inputs, targets and continuation contracts. Reuse
the prior 5654 forecast pool and 4096 general replay pool. No claim of held-out
performance follows evaluation on this panel. Reserved release packs stay closed.

Run three arms from identical original supervised weights, zero critic and seed
20260926: oracle supervision throughout; reward learning throughout; and sixteen
oracle-supervised updates followed by reward learning. Every arm also receives
the same consequence supervision and general replay at every update. This is
an intervention on decision supervision, not the earlier forecast-only ablation.
Use at most64 updates, language learning rate8e-7, critic rate1e-4, batch size2,
decision weight1, forecast weight0.25, replay weight1, value weight0.5, entropy
weight0.01, policy clipping0.2 and a20% uniform action-exploration mixture. Retain
the mean0.02/individual0.10 divergence limits and exact transactional rollback;
a rejected update stops its arm with no automatic retry or rate change.

Each teacher update uses twelve oracle questions: one next-action and one
inspection-value question for each of six goal/cost cells. Rotate the remaining
horizon through4,3,2,1 and shuffle histories within each cell. Each reward update
executes both hidden worlds for all six goal/cost cells: twelve current-policy
episodes. Consequence supervision uses two existing questions per each of seven
groups plus six oracle-continuation questions paired to the selected histories.
General replay uses32 questions per update. Record exact schedules, repeats,
executed trajectories and completed backward presentations separately. Retain
partial collection receipts and never use model predictions as rewards or labels.

Before optimization measure both the unchanged foundation (adapters disabled)
and supervised parent on the complete canonical and reversed oracle panel.
These are fixed controls, not candidates selected using reserved scores. Check
actual-device gradients through decision and optimal-continuation losses, detached
critic behavior, current-policy likelihoods, unchanged-weight divergences and
real executed rewards. Keep the canonical action-forward contract. Stop on failed
qualification before any sustained training.

At each arm's baseline and updates16,32,48,64 measure the same full panel, all
twelve greedy live database episodes, the622-question general retention cohort,
and the existing36-decision/308-forecast exposed report cohort with its complete
public contract. Report canonical and reversed results separately, averaging
oracle scores equally by goal/cost/remaining-horizon cell. Keep the native action
probabilities distinct from outcome probabilities and exploration behavior.

A capacity improvement requires at least0.10 higher normalized live database
return and0.02 lower canonical optimal-continuation expected Brier error, with
general accuracy drop at most0.01 and log-loss increase at most0.02. The exposed
report safety limits remain return decline at most0.02 and Brier increase at
most0.02. Also stop on a database-return decline over0.02 or oracle Brier increase
over0.02 from the arm baseline. Stop after two checks without a selected capacity
improvement once32 updates have been attempted. Select diagnostic checkpoints
only among passing capacity improvements, by database return minus0.25 times
oracle Brier. These diagnostic gates do not replace the original transfer/release
requirements; passing cannot promote a release. A single seed and one authored
mechanism cannot establish generalization.

Each arm has a40-minute cap including loading/evaluation; preserve partial work
on timeout. The whole allocation is one H200 for at most three hours, with an
independent authenticated provider-side shutdown and verified artifact recovery.
Reserve at most$20 from the remaining original authorization: maximum$16.20
compute at$5.40/hour plus$3.80 storage/recovery. Check fresh billing and actual
rate before allocation. No new budget, extension, second rental or larger model.

Qualify admission, schedules, objective gradients/rollback, metrics and the input
bundle locally before renting, under the existing memory/process guard. Do not
load the foundation model on the Mac. Complete recovery and an independent
result/consumption/lineage audit before proposing another experiment.
