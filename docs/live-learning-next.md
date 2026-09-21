# Before the next real-tool learning pilot

The history pilot improved exposed forecast and later-tool scores without
qualifying a replacement for the original supervised step-2,742 adapter. Use that
original parent for the next separately frozen diagnostic. The prepared independent
release evaluation remains unopened; it is not a checkpoint-selection aid.

The next comparison is forecast supervision, reward-only Proximal Policy
Optimization, and their combination, from identical adapter weights and matched
general replay. The existing training code and retail execution interface cannot
simply be connected without checking their contracts:

- `tool_lab/expanded_learning.py` applies exploration only to `shell_action`.
  Retail actor rows use `retail_live_action`. A new, separately tested policy
  adapter must declare both action types explicitly and preserve native forecast
  probabilities. Do not silently rename a forecast into an action to get it through
  the older loss or divergence guard.
- Retail terminal success pays 20, less every attempted operation's cost. Earlier
  shell objectives use a different scale. Declare reward/value units before
  collection and inspect actor, critic, forecast and replay gradients separately.
  A numerical rescaling must preserve the environment's underlying utility and
  reward ledger. It must not change labels or make the critic's estimate a reward.
- The critic uses detached language features. Verify that value loss changes only
  the critic, actor and forecast losses reach the internal language adapters, and
  each stored action likelihood matches the distribution that actually selected
  the executed command. Retain rollback of weights, optimizer and random state
  after an excessive update.
- Matched reset schedules are prepared inputs, not on-policy data. Each reward arm
  must collect fresh actions from its current policy, execute them in isolated
  workers and independently check terminal success. Matching worlds and costs
  across arms does not require their actions or resulting trajectories to match.
- Forecasts must retain their declared horizon. The existing retail history
  forecasts concern one specified command followed by stopping. Fixed-procedure
  forecasts describe a displayed continuation. Neither is automatically a label
  for success under an evolving learned actor. Any new on-policy counterfactual
  collector needs its own clone/replay and compatible-world weighting audit.

The matched retail qualification crosses six cost pairs with all existing
compatible states. It does not establish more than three goals, two mechanisms
and one connected training ownership group. Combine only role-eligible mechanisms;
whole new mechanisms must supply independent transfer evidence. Previously exposed
calendar/payment diagnostics are not fresh transfer, and telecom remains reserved.
Qualify any added runtime locally before renting hardware.

Freeze a bounded three-arm, two-seed experiment only after these interfaces, full
context limits, independent outcome labels, retention cohorts and mechanism
ownership qualify. Reconcile the existing budget and require remote setup/training
deadlines, exact checkpoint lineage and verified recovery. Preserve the prospective
joint return/forecast improvement requirements and general retention; a successful
training-family diagnostic alone cannot promote a release. No new rental or
production training run is scheduled by this document.
