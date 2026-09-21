# Retail actor interface: local qualification

The language-policy adapter now uses the retail environment's actual reward
contract: 20 for verified terminal success, zero otherwise, less every tool-attempt
fee. This is a separate interface from the earlier shell actor, whose prompt
describes a different reward scale and irreversible-error penalty. Neither frozen
environment nor the active supervised trainer was changed.

The actor selects a fully specified command, sees the real response, and chooses
again. Its action probabilities and critic value never provide a ground-truth
label. The collector records the probability of the action actually executed and
computes returns from charged costs and an independent terminal goal check. It
rejects unexpected tool failures, incomplete episodes and overlength prompts.

All **222 pre-action observations** from the existing **120 runtime receipts**
(60 primary episodes, 60 independent replays) fit the pinned Qwen3.5-9B tokenizer:
**45 distinct public/tokenized inputs**, maximum **2,253 tokens**, zero exclusions
under the 4,096-token limit. The source runtime audit reproduced before this check.
These are repeated historical observations, not 222 new tasks or new training rows.

Six CPU checks passed for hidden-world equivalence, terminal payouts, six-turn
termination, refused-attempt fees, action likelihoods and failure handling. The
same visible input can correctly yield different terminal outcomes in different
hidden worlds. These checks use a small scripted policy and stub callbacks; they
are not learned-model results or new real-tool executions.

No new worlds, foundation-model calls, optimizer steps or admitted supervised
questions were produced. The saved receipts cover only their recorded paths.
Before a learning run, the next qualification must cover isolated real-tool workers,
the real six-turn cutoff, and the longest generated histories. It must also cross
the same cost schedule with every world in each declared prior: the historical
runtime fee cycle is a coverage test, not a matched training distribution.

The private format audit is bound to freeze
`5e69546366e80972c4ccc6f13d6b9c83b33489b8ec57e0ecc5f7ed8680d5bc81`.
The public aggregate report is in `results/retail-actor-v1/summary.json`.
