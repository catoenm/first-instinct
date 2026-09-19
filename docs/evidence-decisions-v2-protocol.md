# Costly evidence and recoverable commands: prospective pilot

Keep Qwen3.5-9B and the original supervised adapter at step 2,742. Compare direct
forecast supervision (`outcome`), reward-only Proximal Policy Optimization
(`reward`), and the same reinforcement learning with forecast supervision
(`hybrid`). The forecast loss is binary cross-entropy, a proper scoring rule.
All arms also receive the same general replay schedule. Internal rank-16
language adapters change; the original foundation matrices remain frozen.

## Environments and data

Three authored families use real configuration files, SQLite databases and CSV
sales reports. Each bundle crosses two hidden file states with two opposite
priorities. Eight regimes cover missing, current and explicitly stale evidence;
cheap and costly queries; write locks; and a previous write that actually failed.
A lookup costs 0.04 or 1.2. A write costs 0.02; unlocking costs 0.03. A completed
correct write earns 1, an incorrect completed write earns -1, and abandoning or
exhausting four decisions earns zero before costs. The verifier checks the whole
requested change and preservation of unrelated files. Return code alone earns
no reward. Lock failures leave the episode open for recovery.

Current initial evidence is produced by an executed query. The stale condition
reads a marked synthetic cached file that is identical across paired hidden
worlds, independent of their current values; it is not a live external service.
Initial observations
and the failed-write prefix are counted as execution work but are supplied
before the actor's reward window. Both targets have equal prior probability
when current evidence is absent. Known locks make an immediate repair fail.

Every forecast specifies one exact command followed by immediate termination,
including after failure. There is no unspecified future policy in its label.
Each label comes from a separately executed branch and final-state verification.
The three forecast phases are initial, after inspection, and after inspection
plus unlocking. Keep opposite outcomes for identical incomplete observations.
All related worlds, goals and regimes stay in the same split. Four file-feature
combinations train; two validate; two test. This tests combinations in three
authored mechanisms, not arbitrary tool use or wholly new domains.

Local qualification runs in pinned Docker containers. Cloud collection uses the
same worker and immutable authored command catalog in disposable directories
under an unprivileged account. This backend is for these audited commands only;
it is not an arbitrary-code sandbox. No model-produced shell string executes.
Before any updates, 144 forecast branches must match Docker observations and
labels on the cloud backend. Harbor and an open-ended proposer are separate
future integrations.

## Training and comparison

Seeds 1507 and 1609; order outcome/reward/hybrid, then hybrid/reward/outcome.
Each arm reloads the same starting adapter. Maximum 40 updates, two optimizer
passes per update, batch size 8, language learning rate 0.000003, critic learning
rate 0.0001. Every reinforcement-learning update samples eight fresh episodes
from its current policy, one per regime. Exact public input, encoded tokens,
option ordering, sampled likelihood, observation, reward and terminal evidence
are retained. Old log probabilities and returns are detached. A value head sees
detached language features, so value fitting cannot silently train the actor.

The actor distribution is 80% native softmax and 20% uniform over legal actions
(a 4% floor with five actions). A separate eight-case local engineering probe
found that the starting model almost never considered abandonment, even when
inspection cost exceeded the reward. The mixture allows such actions to be
observed. Sampling, rescoring, entropy and the policy loss all use the exact
same mixture; this is not off-policy relabeling. Greedy choices are unchanged.
Outcome forecasts and general replay use the native logits without the floor.

One forecast group per update supplies all 24 questions crossing paired worlds,
opposite goals, three observation phases and two repair commands. Forecast-only
and hybrid arms use identical group schedules per seed. All arms sample the
same 16 general replay rows per update with replay weight 0.5. Reward-only does
not consume the extra forecast labels. Branch execution and forward counts
are reported; this is a component comparison, not equal total compute.

Before the first optimizer step, audit policy gradients into language adapters,
zero critic-to-language gradients, and unchanged parameter hashes. Rescore each
new rollout before updates to catch mismatched sampling/training likelihoods.
Clip ratios at 0.2. Stop an arm if mean full-distribution divergence exceeds
0.02 after a step. Forecast and replay gradients have their own reported checks.

Evaluate every ten updates. Include update zero in checkpoint selection.
Select by validation mean return minus 0.25 times forecast Brier score, requiring
general macro accuracy no worse than baseline minus 0.02 and general macro
log loss no worse than baseline plus 0.05. Also require return no worse than
baseline minus 0.02 and Brier no worse than baseline plus 0.02. Two missed
validation improvements stop an arm after at least twenty updates. Report
actual updates and early stops; planned budgets do not imply equal work.
If the baseline is within 0.02 of reference return and Brier is below 0.04,
stop before updates because the experiment lacks headroom.

Test data cannot affect checkpoint selection, stopping or hyperparameters.
After selection, evaluate held-out compositions and the frozen general transfer
set, including the original checkpoint as a separate final control. Report both
seeds, per-regime returns, success, Brier and log loss; do not
equate action probabilities with success forecasts or infer Jev's private recipe.
This pilot uses a small authored curriculum, not massive new pretraining data.

## Bounds

The user authorized this pilot within a $60 allocation and the existing $500
cumulative personal compute allowance. Prior tracked compute is $151.4214,
excluding storage. One H200 only, quoted at no more than $5.40/hour, with a
nine-hour rental ceiling: at most $48.60 compute and $11.40 reserved for storage
and recovery. Each arm has a 1.1-hour limit. Setup has a one-hour limit. A remote
provider stop guard operates independently of the laptop; verified artifact
collection precedes deletion. A failed qualification or training stage ends the
pipeline and preserves its evidence. No automatic extra rental or recipe retry.
