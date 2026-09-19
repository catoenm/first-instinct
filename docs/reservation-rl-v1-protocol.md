# Language reinforcement learning in the verified reservation environment

This prospective experiment follows the consequence-learning pilot. It runs
fresh sampled trajectories through the unchanged C reservation environment and
trains the Qwen3.5-9B language adapters from executed rewards. Its scope is one
known six-world mechanism. Harbor, native CUDA PuffeRL, new task families, and
larger foundation models are outside this comparison.

Both arms start from the released supervised adapter at step 2,742, SHA-256
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`,
on the already pinned Qwen revision. The later consequence adapter is not the
starting point, so its validation overlap cannot select this experiment's start.
Original sources, evidence and demo weights remain unchanged.

## Data and split

The new sampler freezes 1,024 training profiles, four validation profiles and
eight test profiles before model inference. Each profile specifies all action
fees, a three-to-seven-turn horizon, and a positive prior over all six worlds,
including the two combined-failure worlds. All variants and worlds of each
profile belong to the same split. Evaluation enumerates every world, weighted
by the profile's disclosed prior, and averages three fixed presentations.
This is a profile-combination holdout within the existing mechanism, not an
independent mechanism benchmark. Old opened evaluation cases select nothing.

Every prompt includes the complete fee schedule. Exact encoded forecast prompt
overlap across splits is fatal during preparation. Actual evaluation action
prompts are also checked against observed training prompt hashes. No artificial
split identifier, private world, target probability or verifier state enters
model inputs. A deterministic option shuffle depends only on public text.

An auxiliary stream uses policy-independent exploratory histories and uniformly
sampled first commands. Replaying each history in all compatible worlds and
weighting their immediate consequences gives exact categorical probabilities
for request status, account existence, and two stock events. There are 3,072
training, 96 validation and 192 test forecast rows. These are correlated event
questions within the same mechanism, not independent environments. The hybrid
arm consumes 32 rows per update. Targets come from the qualified C core;
96 additional randomly parameterized episodes are compared against independent
SQLite execution before rental. General replay and retention reuse the prior
training and validation subsets, never the old general test data.

## Training comparison

- Reward: Proximal Policy Optimization plus entropy and general replay.
- Hybrid: the same objective plus exact immediate-consequence distributions.

One paired seed (307), at most 96 updates and 32 complete episodes per update
per arm; at most 3,072 sampled episodes per arm. This is an initial comparison,
not a multi-seed generality claim. Both arms use the same starting weights and
scenario schedule, but sampled trajectories diverge as policies change. The
hybrid receives additional labels and model computation, which must be reported.
Early stopping can produce different realized counts; report those counts and
compare common validation updates rather than claiming equal consumed budgets.

`puffer_lab/environment_rl_data.py:CONFIG` is authoritative. Internal rank-16
language adapters train at learning rate 0.000005; the detached value estimator
trains at 0.001. The foundation and vocabulary matrices remain frozen. Each
update uses at most two full-batch gradient-accumulated optimization passes,
microbatches of four, policy ratio clipping 0.2, entropy weight 0.02, value weight
0.5, replay weight 0.25 and (hybrid only) forecast weight 0.5. Replay has 16 rows
per update. The forecast objective is the strictly proper categorical log score.

Each sampled action is executed, with per-action fees and verified terminal
reward from the C core. Complete-episode undiscounted returns provide Monte
Carlo advantage targets; the batch normalizes return-minus-value advantages.
There is no bootstrapping across episode boundaries. Old sampled probabilities,
exact token inputs, menus, transitions, returns and parameter-update ledgers are
saved. An unchanged-policy check must pass before optimization. Pure policy and
entropy gradients are measured before auxiliary losses are added. Value features
are detached from language representations. Each pass checks full categorical
divergence from the rollout policy; above 0.03 it skips the remaining pass and
collects fresh trajectories next update. Nonfinite values abort the arm.

## Selection and reporting

Evaluate validation every 16 updates. Starting step zero is eligible. Select
the greatest weighted mean realized return, requiring improvement above 0.002
and retained general accuracy within two percentage points and log loss within
0.05 of that arm's baseline. Stop after three checks without an eligible
improvement. Forecast metrics never select checkpoints. Only after selection
is fixed, evaluate both the original and selected adapters on the sealed test
profiles and forecast stream. Preserve step-zero selection and negative results.

Report success, realized return, partial failures, steps and each presentation;
report forecast excess log loss and squared probability error separately.
Action probabilities are not success forecasts. Log environment execution,
rendering/tokenization, model inference and optimization times, plus peak GPU
memory. A CPU-only simulator benchmark is not total language-agent throughput.

## Compute and recovery

One personal Runpod H100 with at least 80 GB memory, rate ceiling $4/hour.
Ten-hour provider stop deadline: at most $40 compute plus up to $10 storage
allowance, within the previously authorized $500 total. Previous cumulative
compute estimate is $99.2633267 excluding storage. No paid model API is used.
Each arm has a 3.5-hour process budget, including evaluation, with the last
15 minutes reserved for final diagnostics. No automatic replacement rental.

Install independent local and cloud provider stop guards before training.
Use the prior pinned runtime. Freeze source and input hashes before inference,
run offline mechanics tests on the rental, run the reward arm then the hybrid
arm, and preserve failure artifacts. Copy and checksum-verify the full archive
before stopping and deleting the rental. Keep the demo on its original model.
