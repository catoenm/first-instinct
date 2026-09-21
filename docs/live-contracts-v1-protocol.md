# Local learning-contract qualification

Qualify the bridge between the completed retail runtime and a future learning
comparison using tiny CPU networks and existing saved receipts. Keep every prior
model, environment, trainer, data split and freeze unchanged. No foundation model,
new world execution, reserved evaluation or GPU rental is part of this stage.

The new policy explicitly recognizes `shell_action` and `retail_live_action`.
For either action type, use a declared exploration mixture over only its offered
options. Record and optimize the actual behavior likelihood. Native language
probabilities remain separately accessible; forecast and general questions must
never receive the exploration mixture. Divergence guards check both native and
behavior distributions without relabeling the row's task.

The critic reads detached language features and predicts future reward in declared
learning units. For retail, one learning unit equals the 20-unit terminal success
payout. Divide every actual future reward and its discounted-free sum by 20;
subtract the critic's prediction in those same learning units. Preserve original
fees, payouts, raw returns and receipt hashes. Do not rescale the environment's
utility relative to costs, normalize successful and failed episodes differently,
or divide the already normalized critic prediction again. No advantage centering
or standardization is introduced by this bridge.

The existing collector remains unchanged. Its raw intermediate transition records
are not supplied directly to the new learner: rebuild records from audited full
traces, explicitly declaring the critic unit at collection. Require the independent
terminal verifier and attempt ledger to reproduce each raw return first. A stored
critic value or model probability never supplies outcome truth.

Bind the actor and critic's exact trainable-parameter identity before collection
and require it to remain unchanged afterward and before any learning attempt.
Rescore the actual sampled actions in gradient-enabled mode before an update:
their likelihood ratios must be inside the declared clipping interval and their
full distributions within 0.001 absolute probability of the recorded behavior.
This numerical check complements the parameter identity; it does not permit
reusing trajectories from changed weights. Historical scripted traces have no
trainable-policy identity and cannot enter an optimizer step.

Forecast loss accepts only independently admitted distributions with an explicit
`command_then_stop` or `displayed_fixed_continuation` contract. Reject action rows,
acceptable-answer labels, malformed distributions and an unspecified or changing
learned continuation. This field is necessary metadata, not proof of provenance;
the future data freeze must still bind its execution receipts and displayed
question. Do not admit decision ties as uncertain consequence targets.

CPU tests must show that both action types receive the declared mixture; padded
options receive zero mass; native and forecast probabilities are unchanged;
stored action likelihoods agree before an update; correctly earned returns are
converted without altering their order; actor/forecast gradients reach language
parameters while critic gradients do not; and an excessive optimizer update
restores both parameters and optimizer state. Guard scopes must preserve task
identities and include native distributions, so exploration cannot hide a jump.

Reconstruct the already qualified 252 matched-retail receipts without new
executions or model calls. Their scripted critics were all zero, allowing an
unambiguous diagnostic interpretation in either raw or normalized units. Refuse
nonzero historical critics without a declared unit. Report these reused receipts
separately from new synthetic unit tests and model-training presentations.

Passing establishes a local software contract, not a trained model, cloud runtime,
policy performance or transfer. Fresh learned actor/forecast branch collection,
broader mechanism coverage, general replay, exact training/selection freezes and
remote qualification remain necessary before a paid comparison.
