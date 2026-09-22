# New live-decision and forecast contracts pass CPU learning checks

The learning code now supports the revisioned database actor alongside existing
shell and retail actors. It explicitly distinguishes immediate goal status,
immediate command response, and outcomes under a displayed continuation.
Earlier frozen experiments and their results remain unchanged.

Seven tests used small randomly initialized CPU networks and real temporary
database executions. They verified:

- Sampling and learning use identical action probabilities across evaluation and
  training modes and batch partitions; existing shell/retail forwards are preserved.
- The collector executes sampled commands, stores their actual probabilities,
  and obtains rewards from database state. A nonzero critic value retains its
  declared units.
- Action and forecast losses reach language parameters; value loss reaches only
  the detached critic. Exploration does not alter forecast probabilities.
- Each of the three objective combinations consumes only its declared data and
  includes general-task replay.
- Incorrect forecast horizons, target-bearing actions, scripted trajectories,
  stale critic weights, and inconsistent reward units are rejected.
- An excessive update restores parameters, a populated optimizer, and random state.

The actual loss consumer also accepted all 3,216 qualified candidate distribution
rows: 120 previously staged questions plus 3,096 live-history candidates. Synthetic
logits were used for this contract check. It measures neither prediction quality
nor distinct task count, and it does not admit the new candidates into a mixture.

No foundation weights were loaded, no 9B optimizer updates occurred, and no paid
hardware was allocated. The original supervised step-2742 checkpoint stays selected.
The tests used about 360 MiB peak process-group memory; the separate candidate
contract check used about 240 MiB, with no additional swap in either job.

## Next bounded learning experiment

Package the qualified database examples with eligible existing mechanisms and
general replay, preserving whole-mechanism ownership. Avoid counting overlapping
initial-state forecast variants as new evidence. Audit full public tool contracts
before selecting data: the [paired diagnostic](report-contract-paired-v1-results.md)
showed that missing mechanics can substantially change forecasts.

Freeze a prospective exposure schedule and evaluation plan before renting
hardware. Compare forecast supervision, reward-only Proximal Policy Optimization,
and their combination from identical original 9B adapter weights. Check actual
CUDA likelihood parity and resource use before optimizer work. Bound each arm,
retain the existing decision/probability/retention stop checks and artifact recovery,
and account for prepared data, consumed questions, repeated presentations, and
checkpoint lineage separately.

An improvement on the database training mechanism alone cannot promote a release.
Broader execution-verified situations and unfamiliar-mechanism evaluation remain
necessary. Preserve the locked release pack until a prospectively eligible
candidate exists; no reserved-score checkpoint selection. This is a qualified
software component, not a newly qualified production training run.

[Protocol](decision-learning-v2-protocol.md) ·
[Aggregate receipt](../results/decision-learning-v2/summary.json)
