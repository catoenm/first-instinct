# First two completed arms: interim evidence

The first forecast-only and reward-only runs each completed 40 accepted updates
without a rejected transaction. Both started from the same original supervised
Qwen3.5-9B language weights. The combined method and second seed are still
running. This is not the final comparison.

On the reserved validation mechanism, the checkpoints selected by the frozen
rule produced these results:

| Checkpoint | Mean return | Forecast Brier, lower is better | General macro accuracy |
| --- | ---: | ---: | ---: |
| Original | 0.2622 | 0.6518 | 87.05% |
| Forecast-only, update 20 | 0.3189 | 0.5286 | 87.05% |
| Reward-only, update 40 | 0.3711 | 0.6529 | 87.15% |

These are 36 validation episodes and 504 forecast questions from one authored
mechanism, plus 622 general retention questions. The forecast-only checkpoint
was selected for combined decision return and forecast quality; update 40 had
lower forecast error but was not the selected checkpoint. Do not substitute it
after seeing the results.

The pattern so far is useful: consequence supervision improved forecasts, while
reward-only learning improved decision return more without improving forecasts.
Whether combining them improves both on unfamiliar situations remains open.
The original-model transfer control and both seeds must finish before applying
the advancement rule. The two available transfer scores are included in the raw
audit, including their negative mean returns; no baseline transfer comparison or
success claim is made yet, and they do not change the frozen training recipe.

The offline audit reconstructed public histories, checked terminal outcomes
against saved files and application databases, bound forecast targets to frozen
executed labels, recomputed probability and general-retention metrics, and
checked checkpoint selection against the original validation rule.

| Training arm | Executed training episodes | Unique training cases / roots | Completed objective presentations |
| --- | ---: | ---: | --- |
| Forecast-only | 0 | No live training rollouts | 1,034 forecasts; 640 general replay |
| Reward-only | 960 | 156 cases / 3 roots | 2,460 policy decisions; 640 general replay |

Forecast-only uses previously executed outcomes. Its 1,034 forecast identities
were distinct; 2,520 training forecasts were prepared. The reward arm's 2,460
policy presentations contained 1,197 distinct question identities and 1,263
repeated presentations. Each arm's 640 replay presentations contained the same
589 distinct identities. Evaluation executions are recorded separately. These
960 episodes are repeated visits to a small curriculum, not 960 distinct tasks.
Both arms covered the same 26 underlying world-and-goal tasks across those 156
context cases and three roots. The forecast labels referenced 1,034 distinct
previously executed branches; this training run did not execute new branches
to create forecast labels. This makes expanding the underlying mechanisms a
different next step from repeating more episodes of the current curriculum.

The reward arm's diagnostic measured a nonzero pure actor gradient into the
language adapters and zero critic gradient into them. This confirms that the
reward signal reached the language network independently of replay supervision.
The large foundation matrices remain frozen; internal adapter matrices change.

One wording correction belongs alongside the frozen protocol: its guard paragraph
says “native” action distributions. The implementation actually checks the same
80% model / 20% uniform behavior distribution used for sampling and likelihood
rescoring. That distribution is consistent across all arms; forecasts use the
native model probabilities. The guard bounds therefore apply to the behavior
distribution, not to unsmoothed action probabilities. The implementation and
active run are unchanged; the original frozen protocol is preserved.

The [interim receipts](../results/mixed-decisions-v1-interim) include 76 verified
files from the two completed arms and their audit. They exclude checkpoint
weights and are not full-run recovery. `tool_lab.mixed_results` is a separate,
post-upload reporting tool; it cannot update the model. Tests reject invented
targets, duplicated evidence, incomplete comparisons and a failed second seed.
The full six-arm study still has only five underlying roots and one transfer
root, so even a passing pilot cannot establish broad generalization or explain
Jev's private training method.
