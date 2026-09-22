# Learning from immediate consequences and live database decisions

This successor preserves every earlier frozen learner. It adds the qualified
`revisioned_live_action` alongside shell and retail actions, and explicitly accepts
immediate goal-status and command-response distributions. Procedure forecasts
retain their displayed continuation. Unspecified learned-policy horizons and
acceptable-choice sets are not outcome distributions.

Every action uses the same one-row, gradient-enabled language forward during
sampling and updates. Disable dropout and keep exploration separate from native
forecast probabilities. Reject action inputs containing targets. The value head
receives detached language features: forecast and action losses can update language
adapters, while value loss updates only the critic.

The database collector samples its current policy, executes each selection through
the qualified adapter, and independently verifies rewards. Hash actor and critic
weights before and after collection. Only records with the exact current trainable
identity may enter a policy update; scripted path qualification is not training
experience. Keep database rewards divided by 100, retail by 20, and shell rewards
at their existing scale. Value predictions are already in the common units and
must not be scaled a second time.

Retain three objective combinations: forecast supervision with general replay,
reward-only Proximal Policy Optimization with replay, and both with replay. Guards
measure native and exploratory action distributions. A rejected or interrupted
optimizer transaction restores parameters, optimizer state, and random state;
no automatic retry or relaxed threshold follows a rejection.

Local qualification uses tiny randomly initialized CPU networks, real temporary
SQLite executions, and the already qualified candidate forecasts. Check unchanged
sampling likelihoods across modes and batch partitions, gradient separation,
objective accounting, stale-critic rejection, units, and full rollback. Then pass
all 120 staged and 3,096 newly prepared distribution rows through the actual loss
consumer using synthetic logits. That last check validates contract/schema
compatibility; it neither runs a language model nor admits the new rows to training.

No foundation model weights are loaded locally. This stage does not qualify CUDA
kernels, resource use of a 9B backward pass, a mixed-data exposure schedule,
probability quality, retention, or unfamiliar-mechanism transfer. A new bounded
comparison still needs those checks, full public tool contracts on its data,
identical starting weights, separate prepared/consumed accounting, and the existing
remaining budget. Preserve the locked release evaluation for an eligible candidate.
