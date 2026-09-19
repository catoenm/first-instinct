# Overnight 9B continuation and mixed-game reinforcement learning

This supersedes the unlaunched reservation-only proposal for the user's requested
H200 run. It adds actual PufferLib Lights Out and 2048 environments to the verified
reservation task, and first tests whether more supervised training helps.
No previous experiment, source freeze, result or live demo checkpoint is changed.

## Supervised continuation

The prior run ended at its planned one-epoch limit, with its final checkpoint
selected. Validation accuracy and log loss were still improving. This is a reason
to test more supervised fine-tuning, not evidence that the original Qwen
pretraining was incomplete or that another epoch necessarily improves transfer.

Start from released step 2,742 on the pinned Qwen3.5-9B foundation. Use a fixed
100,000-row reservoir of the previous general training data plus 20,000 new
verified game questions. The game questions cover 2,500 distinct training boards
per game. Rotations and reflections share one board group and split. Each game
has 32 validation boards and 64 test boards, with four questions per board.
All labels refer to the offered answers; several equally good choices can be
acceptable. No test board is used to choose a checkpoint.

Lights Out action labels come from an exact binary linear-system solver,
enumerating the null space to find all shortest solutions. The 2048 choice task
explicitly asks about immediate merge points, not the best long-term strategy.
An independent list-based merge oracle is checked against native Puffer moves.
Other questions cover immediate consequences. All game input text is derived
from visible boards and rules. No solved path or private random seed enters it.

Add a general-only control from the identical original checkpoint: 120,000
general training rows, sharing 100,000 rows with the game mixture. Both arms
use the identical validation set and selection procedure, identical maximum
example counts and time limits. Input token counts and actual consumed steps
can differ and must be reported. This separates adding games from simply giving
the model more optimization; it remains one seed, not a general causal conclusion.

Run one continuation pass per arm, at most two hours each, rank-16 internal language adapters,
learning rate 0.00002, microbatch 16, accumulation four, validation every 250
updates, sixteen questions per task. The original supervised trainer selects the
lowest macro validation acceptable-set log loss, including step zero. An external
watcher stops after three complete validation checks without improvement.
Keep the original and selected adapters and report game/general results
separately. Restarting the optimizer is explicit; this is not bit-exact resume.
Before training, freeze twelve questions per task from the old general test and
challenge splits as a non-game transfer benchmark. Those questions enter neither
training nor checkpoint selection. Evaluate the original, both selected
supervised continuations, and both selected reinforcement models. The benchmark
was opened in previous research, so label it an existing benchmark. Better game
scores alone do not constitute transfer.

## Actual game execution

The two game headers are unmodified PufferLib source from commit
`6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`, stored with licenses and hashes.
Only graphics calls are disabled. The adapter invokes upstream `puf_step`;
this is not native PuffeRL training. Our PyTorch learner updates the Qwen model.

Lights Out episodes have a twelve-press budget and upstream rewards. 2048 uses
a declared 24-move horizon and cumulative upstream reward, including merge
rewards and penalties. Reaching that horizon is truncation, not winning 2048.
Report merge score and game-over rate. Auto-reset boards are never described
as the terminal observation of the previous episode. Initial boards are drawn
from the frozen split. Overlapping intermediate states can arise during play;
record encountered training prompts without claiming disjoint state spaces.
Native pseudorandom streams are platform-dependent; both model arms and their
comparisons run on the same pinned Linux machine. Local seeded generation is
preserved in frozen artifacts rather than claimed bit-identical across platforms.

## Reinforcement comparison

Both arms start from the same selected supervised continuation, with the exact
adapter file hash passed to each trainer and recorded. Reward-only uses Proximal
Policy Optimization, entropy, and general replay. Hybrid adds consequence
forecasting. The critic fits detached features and cannot directly change the
language representation. Base weights remain frozen; language adapters train.

The authoritative settings are `games_lab/mixed_data.py:CONFIG`: one paired seed,
96-update cap, 24 fresh complete episodes per update (eight per family), two
optimization passes, microbatch eight, learning rate 0.000005, ratio clip 0.2,
entropy weight 0.02, critic weight 0.5, replay weight 0.25, forecast weight 0.5.
Normalize Monte Carlo advantages within each family. Average the transition
losses across the collected batch, so longer episodes contribute more updates.
Do not claim equal compute or equal realized interaction counts after stopping.
Full action-distribution divergence above 0.03 ends that update's remaining
optimization pass; the next update collects fresh trajectories. Nonfinite
values or a failed unchanged-policy mechanics check abort the arm.

Each hybrid update has 32 extra forecast examples sampled from a frozen stream:
half reservation consequences, one quarter deterministic game consequences,
one quarter stochastic 2048 spawn forecasts. The last targets estimate whether
the next tile lands in the top row from 128 independently seeded actual native
executions per queried board/action. They are finite-sample estimates, not exact
population probabilities. Preserve trials/counts and the ideal uniform-spawn
reference separately. They are not long-horizon success forecasts. Sampling
with replacement creates repeated examples, which must not be counted as new
independent situations. All auxiliary supervision and compute are additional
to the reward-only arm and must be disclosed.

Validate every sixteen updates. Each family receives one third of evaluation
weight; report its native reward separately because reward units differ. Within
reservation, enumerate worlds weighted by their public priors. Evaluate four
validation game boards per game, and eight test boards per game, in three
presentations. Test remains sealed until each arm's validation selection is fixed.
Select higher mean realized return by at least 0.002, provided general retention
accuracy drops no more than two percentage points and log loss rises no more
than 0.05. Step zero remains eligible. Stop after three non-improving checks or
2 hours 40 minutes per arm, including final evaluation. A single-seed experiment
on three known mechanisms is not evidence of broad generality or Jev parity.

## Overnight operation and spending

Use one personal Runpod H200 as explicitly requested. GPU rate ceiling $5.50
per hour, independent twelve-hour provider stop deadline, at most $66 compute
plus $9 storage allowance for this allocation. Prior cumulative compute estimate
is $99.2633267 excluding storage, within the user's $500 total authorization.
No additional paid language-model service or automatic replacement rental.

Freeze and verify source/data before model execution. Install authenticated local
and cloud provider stop guards before setup. The complete remote pipeline runs
supervised continuation, reward training, hybrid training, and artifact archiving
without depending on the Mac staying connected. Model/process deadlines leave
time for artifact recovery. A local collector verifies the archive and every
file before stopping and deleting this specific rental; the provider guard
stops billing at the deadline even if the Mac is offline. Preserve failures and
never promote a research checkpoint to the demo automatically.
