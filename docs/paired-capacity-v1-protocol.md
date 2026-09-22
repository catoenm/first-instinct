# A meaningful dose of paired decision supervision

The next capacity intervention uses the same Qwen3.5-9B foundation and original
supervised step-2742 adapter. The new input contracts, paired targets and guarded
update consumer have passed local checks. This is a longer **supervised** stage
using execution-verified decision and consequence labels. It is not a new
Proximal Policy Optimization comparison or evidence that reinforcement learning
has succeeded. First establish whether the broader exercised decisions can be
learned; the subsequent reward-learning comparison needs identical starting
weights and its own explicit objective separation.

Use the fixed recipe in `tool_lab/paired_capacity_plan.py`: at most 256 accepted
updates, a minimum useful dose of 128 before ordinary plateau stopping, native
scores, a learning rate of 0.0000008, and general-task replay. Each update draws
six teacher questions from each of six eligible families, plus all six initial
database decisions and all six initial inspection questions. Reservations have
no teacher targets and remain forecast-only. Also draw two forecasts from each
of seven families and 32 general replay questions. The three objective means
have equal weight. No critic gradient comes from these supervised targets.

This gives 48 teacher, 14 forecast and 32 replay presentations per update.
Cycle canonical questions within each family, and cycle a question's option
positions whenever it is presented again. Initial database questions appear
once per update with their menus rotated. Keep them out of the other teacher
bucket so a step does not double count them. At update128, every canonical
teacher question must have been presented and every initial teacher must cover
every answer position. Publish actual measured coverage; the broader forecast
pool need not all be consumed. Costs, wording and order variants remain related
to their original worlds and cannot become independent tasks.

Evaluate the unchanged database/report/general development cohorts before
training, at accepted-update multiples of32 and after the final saved weights.
Keep the original capacity improvement and retention gates unchanged. Selection
requires increased database return of at least0.10 and decreased database
forecast Brier error of at least0.02, with the existing report/general safeguards.
This remains a capacity test of exposed mechanisms, not release qualification.

Ordinary plateau stopping begins only after128 accepted updates and two safe
evaluated checkpoints without an improvement of at least0.0001 in
`database_return - 0.25 * oracle_brier`. Track progress even before the joint
selection gate passes; otherwise improving runs could be stopped too early.
Stop immediately for safety failures or rejected guarded updates. A deadline
before the minimum dose is a bounded, incomplete learning test, not proof that
the architecture cannot learn the task. Save and evaluate the last accepted
weights even when no candidate qualifies for selection.

Prospective phase caps are 30 minutes for setup/preflight, 4.5 hours for the
training phase (including intermediate development checks), 30 minutes reserved
for final evaluation, and 15 minutes for recovery, with an independent provider
hard stop at six hours. Before each update, reserve enough time to finish it;
do not consume final-evaluation or recovery time to extend training. The proposed
stage ceiling is $35 from the existing remaining authorization, subject to fresh
owned billing and price reconciliation. This is not a reservation or new budget.

The data/schedule package alone is not a launch permission substitute: finish
the trainer, actual-device entry checks and exact artifact-recovery bundle first.
Those are implementation checks within the user's existing authorization;
another user confirmation is not required. No GPU is allocated by this local
preparation. No reserved model-score pack is opened or release checkpoint changed.
