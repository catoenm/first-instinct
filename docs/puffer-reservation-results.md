# A learnable transaction environment, before scaling the language model

Two small policies learned to recover from individual reservation failures.
Both achieved 100% success on the fixed familiar and validation checks, but
only 50% on previously excluded combinations of missing stock and a missing
account. This establishes a usable reward signal in one authored environment.
It does not establish broader language ability, calibration, or Jev parity.

The trained policy has **7,370 parameters** and takes 39 numerical features.
The existing Qwen3.5-9B checkpoint is unchanged. This experiment used our local
PyTorch implementation of **Proximal Policy Optimization**, running on CPU over
the same C core as the PufferLib adapter. The native CUDA PuffeRL trainer was
not executed. No additional compute was rented.

## What the environment verifies

The goal is to allocate one unit each of A and B to a customer request while
preserving unrelated database records. A customer can be missing, inventory
can be short, and sequential writes can leave partial state. The agent chooses
among inspection, atomic or sequential reservation, replenishment, customer
creation, scoped undo, and finish. Every choice consumes a declared fee and a
turn. Final reward comes from checking the resulting state, including foreign
keys, inventory conservation, allocations, schema and protected rows.

The fast C simulator was compared with an independent implementation that
actually executes SQLite statements. The qualification contains:

- 648 distinct branches, each replayed once: 1,296 primary database attempts.
- Eight qualification guards, six earlier debug attempts, and 64 additional
  random guard branches: **1,374 total attempts**, within the declared 1,424 cap.
- Matching intermediate states, observations, terminal classes and rewards.
- A headless lifecycle check against the real pinned PufferLib header:
  3,000 transitions and 738 episodes on the recorded machine.

These are branches within **one mechanism**, four public qualification profiles
and six hidden worlds. They are not 1,374 independent skills. SQL statement
counts include audit queries and must not be described as model tool calls.
The C simulator produces subsequent training rewards; SQLite is its execution
oracle during qualification, not a database opened for every training step.

## Learning results

The [prospective learning protocol](puffer-reservation-learning-v1-protocol.md)
fixed two seeds, 64 updates, 64 episode states, and 32 rollout steps per update.
Each seed consumed **131,072 sampled training transitions**. Total transitions
including validation and diagnostics were 135,419 for seed 41 and 135,340 for
seed 73. There were no replacement runs or settings changes after viewing the
final diagnostics.

Both runs selected update 40 by validation return, retaining the earlier
checkpoint on ties. Each performed 2,048 optimizer steps over all 64 updates.

| Policy | Familiar success | Combined-fault success | Familiar mean return | Combined mean return |
| --- | ---: | ---: | ---: | ---: |
| Seed 41, initial | 18.75% | 0% | -0.2806 | -0.0506 |
| Seed 41, selected | **100%** | 50% | **0.9338** | 0.3975 |
| Seed 73, initial | 12.50% | 0% | 0.0350 | -0.0112 |
| Seed 73, selected | **100%** | 50% | **0.9338** | 0.3925 |
| Authored inspection-and-repair continuation | 62.50% | 50% | 0.5644 | **0.4200** |
| Uniform random actions | 15.23% | 3.13% | -0.0065 | -0.0615 |
| Finish immediately | 0% | 0% | 0 | 0 |

Success is weighted by the disclosed prior and equally across eight contexts.
The deterministic familiar diagnostic enumerates 32 context/world cases; the
combined diagnostic enumerates 16. Random actions use 16 repetitions per case.
Familiar cases reuse the training cost values, horizons and supported worlds;
they are a learning check, not an unseen-task benchmark. Validation uses
different inspection and atomic prices within the same mechanism.

Both learned policies succeed in every combined-fault case with a six-action
horizon and fail in every case with a three-action horizon. In the short cases,
both choose `atomic → create_account → atomic`: the first attempt exposes the
missing account, and the last exposes a shortage when no turns remain. The
public continuation ties their success rate and achieves better mean return
on this diagnostic. Success on familiar cases therefore does not mean the
learned policy is optimal or robust to composition.

The combined diagnostic changes both the fault composition and the supplied
prior. It cannot isolate those two causes. Two seeds in a tiny finite world
also provide no basis for a claim about general tool-use performance.

## What we now have for calibrated outcomes

The qualification produces **288 exact conditional outcome/cost targets** over
32 distinct public histories. Each target means: take this first action, then
follow the named public continuation within the remaining horizon. Compatible
hidden worlds are weighted by the disclosed prior; zero-prior worlds remain
implementation controls and are excluded from the target distribution.

Only **8 of 288 targets** have success probabilities strictly between zero and
one. The labels preserve the joint distribution of outcome and future cost,
but this is currently a narrow source of uncertainty. These targets were
**not used as a calibration objective in the small-policy training**. That run
learned from scalar trajectory rewards with an action policy and value estimate.
The policy's action probabilities are not success forecasts. This experiment
is preparation for our calibrated-outcome work, not evidence that such a
training objective improves decisions or calibration.

The best first action under the fixed continuation changes with price and
deadline: inspection for cheap queries, an atomic attempt for cheap attempts,
and customer creation for the short-horizon profile. These are conditional
values under that continuation, not globally optimal action values.

## Evidence and next experiment

The [published receipts](../results/puffer-reservation-v1/) include the original
source freezes, database-attempt ledgers, branch replays, conditional labels,
all 128 sampled rollout files, losses, validation selection, diagnostic actions,
and initial/selected/latest small-policy checkpoints. The original qualification
summary remains intact; its 1,310 attempt count predates the separately recorded
64-guard extension. The aggregate count is 1,374.

The [offline audit](../puffer_lab/audit.py) verifies frozen source and artifact
hashes, recomputes conditional labels, reconstructs every saved training action,
checks observations/rewards/terminal flags, recalculates advantage targets, and
replays the saved evaluation actions. Initial and selected checkpoints are also
checked against their corresponding saved rollout probabilities. Audit replays
are counted separately and do not modify weights or execute additional SQLite
tasks. See [reproduction instructions](../puffer_lab/README.md).

The next language-model experiment should retain 9B and test whether it can
interpret these decisions from public text, with reordered and paraphrased
action descriptions. Measure that baseline before collecting training data.
For the data expansion, reserve new mechanisms and fault combinations for
evaluation, include short-budget recovery problems, and add observations with
graded uncertainty so conditional probabilities are not almost always zero or
one. Then compare ordinary trajectory reward training with an explicit outcome
forecasting objective. A larger-model comparison becomes useful if the same
environment and data reveal a language-understanding bottleneck.
