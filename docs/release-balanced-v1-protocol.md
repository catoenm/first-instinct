# Coverage-balanced supervised follow-up

This is a prospective follow-up prepared after release-pilot-v1's first check.
It may run only if that pilot completes without qualifying. Preserve the first
run and its unchanged stopping rule. Both start from the original supervised
step-2742 adapter, not from a failed continuation. No larger foundation, newly
generated examples, held-out data reuse, or relaxed advancement threshold.

At the first pilot's update 40, general retention passed, overall forecast Brier
error fell from 0.34093 to 0.27268, and tool log loss improved from 0.43247 to
0.37736. But equal-server tool accuracy changed from 0.86874 to 0.86658, and two
small outcome groups worsened. The update schedule had presented only four
application-change forecasts and seven application task-success forecasts.
Question-averaged tool accuracy improved, including large menus, while equal
server accuracy did not. These observations motivate broader coverage per update;
they do not establish a cause or guarantee that this follow-up will work.

Keep the admitted release pack, original parent, formatter, 4,096-token limit,
optimizer, learning rate, microbatch size and exact development cohort. Change
sampling prospectively: rotate general tasks, tool server groups, and verified
family/question-kind pairs. A question belonging to several server groups still
has a single global presentation limit. Exhausted groups drop out. General/tool
questions appear at most once; verified questions at most twice within this run.
The same questions may have appeared in the earlier experiment; count that
cumulative exposure separately rather than claiming newly generated data.

Each update contains 32 general, 24 tool and eight verified questions, with equal
weight per question. The maximum is 240 updates (15,360 presentations), evaluated
every 80 updates, with the same two-check patience. This jointly changes coverage,
verified-example weight and maximum learning duration; it is a practical recipe
revision, not an isolated ablation of one factor. Seed 20260922 fixes the schedule
before any follow-up scores. Freeze exact rows, schedule and source hashes before
launch. Inspect the census to confirm every verified question kind receives
substantial coverage at the first check.

All advancement and retention checks remain exactly those of release-pilot-v1:
tool macro accuracy +0.03 from the original; general accuracy drop at most 0.01
and log-loss increase at most 0.02; no declared slice loses more than 0.03 accuracy;
both proper probability scores do not worsen in any declared outcome group.
The original checkpoint stays eligible. Stop retention/numerical failures and
two checks without an eligible improvement. The earlier development feedback
informs this recipe, so it is no longer an independent confirmation experiment.
Fresh reserved tool transfer remains untouched for final evaluation.

Repeat real GPU context/gradient qualification and the separate-process update-one
restart check. The trainer limit is 8,280 seconds. Provision at most one H200 with
a provider-side stop deadline no later than three hours from creation, only after
the first rental's artifacts are recovered and its deletion confirmed. Maximum
compute at $5.40/hour is $16.20. The first attempt's actual billing plus retained
safety allowance, this compute ceiling and storage/recovery must fit the SAME
original $25 pilot allocation. If they do not, reduce the paid duration before
launch; never silently create a new $25 budget. No automatic main run or demo
promotion follows a failed pilot.
