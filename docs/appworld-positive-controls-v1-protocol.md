# Successful alternatives before supervised learning

The first intervention collection completed 219 isolated worlds and 5,497
application calls. All 29 reference demonstrations and exact replays passed.
All 132 stopping, skipped-call, wrong-target and invalid-authentication branches
failed the upstream task requirements. That is useful negative control coverage,
but an inadequate mixture on its own: a model could associate a failed command
with an unsuccessful whole trajectory without learning recovery.

Before any model inference, add two prospectively defined interventions at the
same fixed histories. Repeat the most recent prior business GET call and then
execute the original continuation. Where the next call has an authentication
argument, first attempt it with the declared invalid session, then retry it with
the original publicly acquired session and execute the remaining continuation.
These are supplied scripts, not adaptive model proposals. Every argument comes
from the already recorded public trace or the declared invalid-session constant.

Execute each new script and an independent replay in separate worlds. Require
the two executions to match, the original prefix to remain identical, and direct
final database hashes to match the successful reference. Check the strict task
verifier as well. If repeating a GET has a real side effect, exclude it from the
successful-observation category rather than assuming that all GET calls are
read-only. Exceptions remain missing labels. Do not overwrite or recollect the
219 completed worlds or the running independent negative-branch replays.

At most 49 repeated-observation histories and 20 retry histories are available,
so this extension is bounded at 138 new worlds and 96 application calls per world.
Combined with the first 219 worlds and 132 independent negative replays, the total
ceiling is 489, below the previously allocated local 608-world ceiling. No model
calls or paid compute are involved in this extension.

Supply at most four plans in a menu: a successful continuation, stopping, an
unsuccessful alternative, and a verified successful extra-observation or retry
plan. Use the same declared costs and paired consequence questions. Report menu
variants as questions, not new worlds. Qualify each training program's positive
and negative alternatives after the public-information and token filters. Add a
command-only shortcut check before freezing a paid learning run.
