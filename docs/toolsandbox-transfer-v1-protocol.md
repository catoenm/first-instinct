# ToolSandbox transfer diagnostic v1

This is a prospective supplementary model evaluation of the **already completed**
[partially observed ToolSandbox collection](toolsandbox-partial-v1-results.md).
The collection, its rational truths and implementation have been inspected;
model performance on these questions has not been measured for this protocol.
Freeze the evaluator, this protocol, exported prompts, all source artifacts,
model-question mapping and runner/checkpoint provenance before predictions.
No new training, data generation, tool execution or model selection is authorized
by this diagnostic. Never modify the frozen GPU experiment's inputs.

## Cohort and holdout

The predeclared reference is the independently selected supervised Qwen3.5-9B
checkpoint, step 2,742. The eligible later comparisons are **all six arms** of
outcome-v2: outcome, reward and hybrid training, each at seeds 77 and 83.
Independently selected checkpoints and latest checkpoints are distinct reported
roles; identify the exact weights and original selection/termination receipts.
They must be chosen by the existing experiment's rules before this corpus is
evaluated. Report missing, failed or unaffordable cohort members transparently.
Do not choose arms, checkpoint roles, epochs or additional attempts from their
ToolSandbox scores. Comparisons reuse this same evaluation version, without
editing questions or scoring after seeing model results.

All 48 root contexts are evaluation-only, including the 16 create roots called
`train` in the original collection. The original split labels remain useful
operation strata: 16 create/train, 16 update/validation, 16 delete/test. None of
these roots, prompts, labels or derivatives may enter training, temperature
fitting, prompt tuning or checkpoint selection. This holdout is relative to our
current training pipeline; public third-party tool code may have been present
in foundation pretraining. Authored priors and tool semantics are explicit in
prompts. This is no claim that the underlying APIs are unseen by the foundation.

## Complete workload and information boundary

Use the published collection byte-for-byte, all 864 exported marginal questions
in their existing order. **720 require model predictions**: 432 outcome and
288 cost forecasts. The 144 singleton stop-cost distributions are known to be
zero and bypass prediction. They never enter learned-forecast denominators.
No result-dependent subsampling, new wording, truncation, menu pruning,
probability calibration, caching or retries based on outputs.

The callback receives only each exported question's `state`, `question` and
`options`, preserving option order. Strip the `deterministic_bypass` transport
flag; it is not part of the existing serializer. Root/split/history IDs,
exact rational distributions, hidden-world support, execution receipts and
verifier summaries remain scorer-only. The tokenizer audit covers all inputs:
1,078–1,458 tokens against 1,536; at most 22 options against 36. A model runner
must reproduce its serialization and reject changed model identity, source,
corpus, prompt audit or checkpoint provenance. Full responses and timing records
must survive partial failure. Do not represent incomplete runs as comparable
complete evaluations or silently resume/replace them.

`model_questions(corpus)` maps raw prediction indices 0..719 to the original
question index 0..863, a stable question ID and public input. Freeze this mapping.
`run(corpus, callback)` uses the same map; `score(corpus, ordered_predictions)`
reproduces the report from the saved 720 probability maps. Callback batching is
an orchestration detail; a serial HTTP wrapper still makes 720 independent
single-question calls. The evaluator itself loads no model, ToolSandbox runtime,
network client or new dependency and never executes a tool.

## Primary decision endpoint

For each root's **initial state**, derive each offered action's predicted value:

`sum(predicted outcome probability × stated terminal utility) − sum(predicted cost probability × future cost)`.

Linearity of expectation makes the two marginals sufficient; do not assume
outcome and cost are independent. Select the highest predicted value, with ties
broken by the existing public action order. Score that action using its exact
expected value reconstructed from weighted actual executions. Primary endpoints
are mean expected value and mean expected regret, **averaged equally over the
48 initial root states**. Also report the fraction of choices attaining an
exactly tied optimum. These are implied one-step choices followed by the
explicit `fast_then_act/v1` continuation, not executed learned-policy returns.

The comparator is the **maximum expected value among the same three offered
actions with that same fixed continuation**. It is not an omniscient oracle that
observes the realized hidden world, nor an unconstrained optimal planner.
Predeclared simple baselines are stop, always complete time query, cheap
continuation (contact lookup initially / fast query after lookup), and a uniform
random offered action. Baseline expectations use the same executed action
values. Report absolute errors in predicted terminal utility, future cost and
net action value, averaging across offered actions within each context.

## Conditioned and forecast endpoints

Keep phone-conditioned results separate. Within each root, weight the A/B
phone histories by their **declared probability of being observed**, obtained
by summing initial world weights whose actual recorded phone return matches
that history. Then average roots equally. Prefix costs are already paid and
excluded from these conditional values. This is not a combined policy that
replans after choosing a lookup. An equal-context macro across initial/A/B
histories may be reported only as an explicit secondary diagnostic.

Keep outcome and cost forecasts separate. Report exact expected clipped log
score, exact expected Brier score, oracle expected losses, total variation,
maximum absolute error, per-option mean squared error and **summed squared
probability error / excess expected Brier**. The summed error avoids making a
larger cost menu appear better merely by dividing by more classes. Expected
losses integrate the specified finite target distribution; positive oracle
loss is irreducible uncertainty. Probability floors are fixed at 1e-12 for log
scores. Modal accuracy is secondary and cannot establish accurate probabilities.

For each marginal and stratum, average offered questions within a history,
then histories using the weights above, then roots equally. Known singleton
costs participate in expected action utility but never in forecast-score means.
Report create/update/delete and original split strata, all per-root results,
all 144 context decisions, and every learned question's probabilities/errors.
No confidence interval treats 144 histories or 720 questions as independent.
The 48 root IDs themselves vary costs/priors within one shared authored mechanism;
report descriptive paired comparisons without presenting those variants as
48 independent mechanisms or the two training seeds as a population estimate.

## Claims and provenance

A difference from the supervised reference can provide narrow evidence about
transfer from prior outcome/reward training to explicitly described third-party
query semantics and cost-sensitive choices. It cannot establish broad arbitrary
question calibration, real-user priors, internal Jev architecture, or general
Jev-like capability. Tool priors and mechanism are visible, and an exact program
can solve this finite authored task. Performance is on these templates only.

The reader validates the collection freeze, reviewed file hashes and prompt
mapping, and recomputes all 432 posterior/marginal/joint distributions and action
values from retained execution labels and actual recorded visible histories.
The earlier independent full-state audit supports the labels; this reader does
not rerun third-party tools or claim a new independent execution verification.
A separate model runner must freeze and verify checkpoint/server/runtime
provenance before and after inference. `score` alone proves arithmetic
reproducibility from supplied predictions, not that a particular model produced
them. Keep a run's source/checkpoint provenance with every published report.
