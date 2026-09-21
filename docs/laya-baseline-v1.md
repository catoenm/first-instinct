# Prepare a fair external decision-model baseline

Keep the canonical-action comparison unchanged. This stage prepares an external
Laya baseline and measures input compatibility; it does not run another training
experiment, load foundation weights on the Mac, or open reserved evaluation scores.

The reviewed upstream source is
[`NandhaKishorM/laya@573e5b62696ba441230cd6be71d593331b5d23af`](https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af).
This is the current upstream formatter, not a claim of inference parity with the
separately pinned MLX port advertised on social media. The current upstream runtime
also clamps some shipped calibration temperatures. Runtime revision, checkpoint
revision, and applied temperatures must all accompany future probability results.

Before any predictions, fix `laya-typed-decisions` as the primary external model.
The English base and multilingual checkpoints are secondary controls, reported
separately; do not select the best checkpoint per question. Compare with the
original supervised Qwen3.5-9B parent and, only if the existing selection conditions
permit, a development-selected new checkpoint. The unchanged Qwen foundation is
an additional control if it fits a separately bounded scoring plan.

## Completed tokenizer-only qualification

The check used 2,510 existing training forecasts, 308 already exposed development
forecasts, and 41 action presentations from the first live-tools preflight. The
41 actions contain 40 distinct logical inputs and are not 41 independent tasks.
No new execution labels, model predictions, or optimizer updates were produced.

| Native checkpoint | Training forecasts preserved in full | Development forecasts preserved in full | Action presentations preserved in full |
| --- | ---: | ---: | ---: |
| English base, 512 total tokens | 1,580 / 2,510 | 144 / 308 | 0 / 41 |
| Multilingual, 1,024 total tokens | 2,285 / 2,510 | 308 / 308 | 9 / 41 |
| Typed decisions, 1,024 total tokens | 2,253 / 2,510 | 308 / 308 | 9 / 41 |

All 8,577 formatted presentations matched the pinned upstream sequence builder
exactly, including token IDs and option marker positions. Each tokenizer was loaded
from hashed local metadata without model weights. Aggregate counts are in
[`results/laya-input-v1/summary.json`](../results/laya-input-v1/summary.json).

The limits apply to several parts of an input. Native formatting caps each option's
text at 48 tokens, shares a separate budget between instructions and options, then
fits the state into the remaining context. Increasing total context alone does
not remove the option cap. In the typed-decisions action sample, 32 presentations
lost instruction tokens, 22 lost option text, and 10 lost state text; these sets
overlap. These are coverage measurements on our existing interface, not accuracy
measurements or evidence that one model is better.

The first audit attempt stopped because older execution traces do not uniformly
store a precomputed input digest. The corrected reader hashes the visible input
itself; the failed attempt remains recorded. No data or upstream formatting rule
was changed. The final qualification made zero model calls.

## Comparison contract

1. Supply exactly the same visible state, question and option descriptions to both
   models. Use neutral A/B/C labels and keep original tool IDs, targets, rewards,
   source identities and verifiers outside the prompt. Each model retains its native
   framing; this is equal information, not byte-identical token sequences.
2. Audit all input loss before scoring. Publish whole-cohort coverage and the
   complete-information subset separately. Do not turn silently shortened commands
   into a claim about decision quality. The 308 existing development forecasts
   can support an initial native typed-decisions baseline. An action comparison
   needs a shared compact interface, qualified without model scores, and a fixed
   cohort of executable tasks. Apply the same compact form to Qwen too.
3. Keep that initial smoke test on exposed development material. It cannot establish
   unfamiliar-task transfer. A later comparison must reserve entire new mechanisms,
   keep related worlds and trajectories together, and freeze all model identities
   and selection decisions before the first score. The existing reserved tool and
   telecom pack remains governed by its own unopened evaluation protocol.
4. For interactive tasks, allow each policy to produce its own trajectory from
   matched initial worlds, goals, costs and stochastic seeds. For consequence
   forecasts, use identical visible histories and the same specified immediate or
   continuation horizon, with labels from independently executed branches. Do not
   compare policies only on the histories one policy happened to visit.
5. Report verified task completion, utility after costs, costly mistakes, observation
   use and stopping. Report expected outcome Brier error and log loss separately
   from action-choice accuracy. A probability of choosing an action is not its
   probability of succeeding. Laya's choice `confidence` is an entropy-derived
   statistic; use the actual option distribution for probability metrics.
6. Qualify unrounded model probabilities for primary metrics. The public Laya
   interface rounds to four decimals and can slightly change total mass or produce
   zeros; preserve and report those public values separately. Do not silently
   renormalize or fit temperatures on evaluation examples. Applied native calibration
   and any separately fitted development calibration are distinct variants.
7. Measure latency and memory with both models on the same device, serially,
   including tokenization and synchronized execution. Separate loading, first-use
   overhead, warm latency and environment execution. Published M3 Max timings remain
   external reference figures, not our own comparison measurements.

The code in `release_lab/laya_compatibility.py` qualifies input framing and output
identity mapping. It is not a complete model-serving adapter. Before paid scoring,
still qualify the real runtime, probability extraction, checkpoint bytes, and the
shared compact action interface. Use a bounded pilot only after the current rental
is recovered and the remaining original authorization is reconciled. There is no
new budget or additional active rental for this work.
