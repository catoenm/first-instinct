# Diagnose recorded actor choices after freezing the execution oracle

Apply the separately qualified revisioned-oracle-v1 values to every database
actor event from all four reward-bearing revisioned-pilot-v1 arms. Read only the
already recovered and audited trajectories. Make no model call, execute no tool
and change no completed-run selection. The forecast-only arms have no database
actor events and are not part of this analysis.

Before looking at the oracle-scored model choices, bind the oracle freeze, closed
pilot audit and recovered trajectory hashes. Require exact public-input identity,
matching offered commands, unlabeled actor rows, valid recorded likelihoods and
the declared 20% uniform exploration mixture. Normalize only the recorded
floating-point probability sum, within its existing tolerance; report the maximum
normalization adjustment. Never infer labels from model predictions.

For each visited public history report sampled and greedy optimal-action hits,
probability mass on optimal actions, and remaining-reward regret for the sampled,
greedy and probability-weighted first action. The latter means optimal value
minus the chosen first-action value assuming optimal public continuation. It is
not the value of the model's whole continuation. Report the regret imposed by
uniform exploration even if the native policy were optimal. Keep abstract raw
reward units explicit.

Report all visits and deduplicated public-history/update observations separately.
Repeated occurrences at fixed weights must have matching distributions. For
histories that recur across updates within an arm, compare the first and last
recorded distribution with exact same-input pairing. Label weights as before
the recorded optimizer update; an update-8 rollout uses weights after update7.
Report every qualifying pair, their support and update span, not a chosen subset.
This is a post-hoc, visitation-selected comparison, not a fixed-panel experiment,
causal learning curve or fresh transfer evaluation.

Aggregate by objective/seed, public goal/cost profile and optimal command. Include
support beside every score. Do not treat frequency-weighted visited contexts as
a representative distribution of tasks. The complete independently frozen panel
remains available for a later prospectively specified before/after model test.
Use the local memory/process guard; no additional compute allocation is needed.
