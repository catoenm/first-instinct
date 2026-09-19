# Mixed executed decisions: bounded 9B comparison

This prospective study follows the completed evidence-decisions-v2 experiment.
It preserves that experiment and the demo. The foundation remains the original
Qwen3.5-9B supervised step-2742 adapter. It is not initialized from a v2 winner.

The question is whether learning to predict executed consequences improves
decisions on an unfamiliar mechanism, and whether combining that supervision
with reinforcement learning improves both utility and probability quality.
This does not establish how Jev is trained.

## Data and ownership

Use only the accepted shell `decision-curriculum-v3-qualified` and application
`application-curriculum-v1-replay-qualified` collections. The earlier application
collection was rejected and remains excluded. Configuration repair, SQLite
repair, and ToolSandbox message delivery are training mechanisms. CSV reporting
is a whole validation mechanism; atomic artifact publication is a whole transfer
mechanism. Related worlds, goals, cost regimes, trajectories and wording variants
remain together. There is only one root fixture per mechanism: five roots total.

The application tools change local simulated databases, never real accounts.
The shell catalog executes real commands against disposable files. Receipt and
state verification is independent of the model. The training process sends only
registered case and action identifiers to the isolated application worker.
The language model receives only public state, question and described options.

There are 3,756 prepared questions: 3,528 consequence forecasts and 228 paired
decision/value questions. Initially only forecasts train the supervised
intervention; decision and observation-value labels remain diagnostic. Every
forecast names immediate execution or a fixed public continuation. Live policy
rollouts instead explicitly promise that the actor will choose again after each
action. The two reward contracts must not be substituted for one another.

Forecast batches keep all compatible hidden-world labels for an identical public
input together. Do not discard conflicting labels, average them into a hard
answer, or use the model's forecast as truth. Four distinct public forecast
groups per training mechanism are scheduled per update; each group's per-world
examples remain intact. Loss averages the examples actually in the batch.
Menus and encoded inputs must not depend on an unobserved label.

Each reward-learning update executes 24 episodes, eight per training mechanism,
with a frozen schedule cycling through regimes and concrete cases. Different
arms share the case schedule; their action sequences can diverge with learning.
The actor sees actual observations and controls its entire remaining horizon.
Pay each command cost when it happens and verified terminal utility exactly once.
Stopping on an initially satisfied goal earns completion reward.

Reuse the exact previous 4,096-example general replay pool and previous general
retention/transfer pools. Every arm receives the same 16 replay IDs per paired
seed/update. Those pools have been used before in this project; do not describe
them as previously unseen research benchmarks.

## Learning and stability

Run forecast-only, reward-only Proximal Policy Optimization, and their combination
for seeds 1507 and 1609. Alternate arm order between seeds. All six reload the
identical original language tensors. The critic starts at zero and uses detached
language features. The critic does not choose actions or train the language
representation. Both supervision and actor gradients must reach the internal
language adapters; the base matrices remain frozen.

At most 40 accepted updates per arm; one optimizer pass per rollout; language
learning rate 0.000001, critic rate 0.0001, microbatch four, replay weight 0.5,
value weight 0.5, entropy weight 0.01, and clipping width 0.2. Sampling and
likelihood rescoring use the same policy: 80% native action distribution and
20% uniform exploration. Forecast probabilities retain the native distribution.
These are prospective stability changes from v2, not retroactive corrections.

Before each optimizer attempt, record native pre-step distributions on 24 fixed
training-only action probes shared by every arm and seed. Reward arms also
include all sampled trajectory states in their guard. A mean full-distribution
Kullback–Leibler divergence above 0.02, any individual divergence above 0.10,
or a nonfinite result rejects the step. Restore trainable weights, optimizer
moments/counters and random state. Stop the arm without retrying or reducing
the rate. A rejected update is never checkpoint-eligible.

Unchanged-policy rescoring must pass before learning. The first update records
separate actor, forecast, replay and critic gradients; actor gradients must be
nonzero and critic gradients into language weights must be zero. Diagnostics
are not optimizer presentations. The learning ledger records each started and
completed backward batch, physical optimizer attempt, acceptance or rejection.
Repeated or rejected presentations remain counted separately from unique IDs.
An interrupted partial batch is not claimed as completed training.

The context limit is 4,096 tokens. Actual repeated-read trajectories reached
2,134 tokens, exceeding the earlier 1,536-token reference-history check.
There is no silent truncation. Any over-limit rollout stops the arm as an input
failure, not as an unsuccessful task label. The stress set is not a proof that
every reachable history fits.

## Selection and advancement

Measure whole-family validation at update zero and every ten accepted updates.
Select by mean return minus 0.25 times three-category Brier score, including the
original checkpoint. Eligibility requires general macro accuracy to fall by
no more than 0.02, general log loss to rise by no more than 0.05, validation
return to fall by no more than 0.02, and Brier to rise by no more than 0.02.
Stop immediately on a failed validation safety gate; stop after two validations
without eligible improvement once at least 20 updates have been attempted.
If the original model is within 0.02 return of the reference continuation and
has Brier below 0.04, do not spend gradient steps on that arm.

Open the reserved publication mechanism only once for each final selected
checkpoint and the original control. Never use those measurements for checkpoint
selection or recipe changes in this study. Advance a method only if both seeds
improve held-out-family return by at least 0.03 and lower categorical Brier by
at least 0.02, while retaining the general bounds. Report all arms, stopped runs,
per-regime results and lineage. Summed three-category Brier differs from v2's
binary-component metric. One transfer root cannot support a reliable claim of
broad generalization; report paired cases without pretending they are independent
mechanisms. A failed transfer result calls for a new reserved mechanism in a new
study, not tuning against the same test.

## Local qualification and rental limits

Before rental, require source/receipt audits, cross-runtime application parity,
Linux catalog parity, tiny-network live learning, and the guarded optimizer's
rollback checks. Record prepared versus actually consumed data and all local
engineering executions separately. Dynamic command proposals and Harbor are
separate future stages; this experiment uses a fixed, verified menu.

Allocate at most $40 from the original cumulative $500 authorization, not a new
budget. The latest provider receipt reports $161.763003300759 across ten known
project pods. Recheck for an existing owned rental before creating anything.
At most one H200, quoted at no more than $5.40 per hour, with a six-hour hard
rental deadline: $32.40 maximum compute and $7.60 storage/recovery reserve.
Each arm is bounded to 0.7 hours; reserve setup, control evaluation, archive and
recovery time. Install a provider-side stop guard before training. Archive even
failures; verify downloaded hashes before deleting the owned pod. No Phantom
resources or paid model APIs. A failed local qualification means no rental.
