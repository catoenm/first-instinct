# Inspecting evidence before choosing a command

The next priority is useful observations and trustworthy outcome forecasts with
the existing 9B model. The previous contextual dataset improved its authored
tasks, while general probability scores worsened. A larger model would not
resolve that measurement problem.

We built and executed a small sequential environment locally. The original
supervised model completed **11 of 12 tasks**, yet its separate success forecasts
were only **13 of 24 correct after diagnostic evidence**. This opened engineering
probe identifies a gap between selecting an action and forecasting its outcome.
It is not a held-out benchmark or a new training result.

## The executable episode

Each case starts with a goal and hidden configuration files, a database, or
sales records. The selector can read raw records, request calculated measurements,
read an irrelevant reference note, apply one of two repairs, or finish. Commands
execute in a persistent Docker container; actual output becomes the next
observation. An update ends the attempt immediately. A host-side verifier checks
the resulting files and protected state. Command exit status alone earns no
success reward. Success earns 1, each command costs 0.02, and the horizon is three
decisions. Failed-write recovery and longer sequences are future work.

One actual database query returned:

| Invoice | Stored subtotal | Sum of line items | Discrepancy |
| --- | ---: | ---: | ---: |
| 786 | 329 | 321 | 8 |
| 789 | 40 | 0 | 40 |

A goal requesting the larger discrepancy selects invoice 789; the opposite goal
selects 786. The two offered repair commands stay identical. The twelve contexts
cross two file states with two priorities in each of three authored families,
using feature combination 111 and seed 4201 from the contextual generator.
These are engineering fixtures, not twelve independent mechanisms.

Before inspection, each pair of hidden worlds produces **identical public
inputs and menus**, but the correct repair reverses. The two worlds are equally
likely, and that prior is stated. Either repair therefore has a 50% success
probability before evidence. After the diagnostic query, the facts distinguish
the worlds and permit a 0% or 100% forecast for these deterministic commands.

The forecast contract is explicit: execute this specified repair now, end the
attempt, and verify. It does not concern an unspecified continuation policy.
Action-selection probabilities are recorded separately from success forecasts.

## Qualification and original-model results

All 72 scripted control episodes completed, executing 84 real commands:

| Control policy | Success | Mean reward |
| --- | ---: | ---: |
| Read measurements, then follow the goal | 12/12 | 0.96 |
| Blindly choose the first target | 6/12 | 0.48 |
| Blindly choose the second target | 6/12 | 0.48 |
| Finish without a repair | 0/12 | 0.00 |
| Read irrelevant information, then finish | 0/12 | -0.02 |
| Inspect, then repair the wrong target | 0/12 | -0.04 |

The blind executions also label twelve pairs of indistinguishable initial
forecast inputs, each containing one success and one failure. Hidden file values,
expected answers and private verifier evidence are excluded from model inputs.
Tests replace those private fields and confirm that the public menus do not change.

The unchanged resident Qwen3.5-9B supervised checkpoint at step 2,742 then made
24 action selections across twelve fresh episodes on the Mac:

| Family | Success | First command |
| --- | ---: | --- |
| Configuration | 3/4 | Read raw records |
| Database | 4/4 | Read calculated measurements |
| Sales report | 4/4 | Read calculated measurements |

Mean reward was 0.8767. The contextual checkpoints were not loaded for this
probe; this is not a comparison against them.

Separately, 48 binary questions forecast both repairs before and after the
diagnostic query. After-query prompts use the same measurement output as the
reference policy, independently of the model's own choice of inspection.
Each label is bound to an executed branch with the identical public forecast input.

| Forecast setting | Accuracy | Brier score, lower is better |
| --- | ---: | ---: |
| Before inspection | 12/24 (50.0%) | 0.2831 |
| After diagnostic inspection | 13/24 (54.2%) | 0.2672 |

Brier score is the average squared difference between reported probability and
the actual zero-or-one outcome. An initial forecast of 50% scores 0.25; a correct
fully informed deterministic forecast scores zero. Initial predictions deviated
from the known 50% by 14.0 percentage points on average. The model often favored
both opposing repairs after inspection. Good action selection does not establish
calibrated outcome forecasts.

The probe used 48 local service requests containing 72 typed questions. There
were no retries, model updates, paid model API calls, or GPU rentals. Model service
metadata stayed fixed; resident parameter bytes were not independently attested.
Prompt sensitivity and this tiny opened set limit the conclusions.

## Recommended next experiment

Keep the 9B foundation. The current environment leaves little room for improving
task success, so strengthen the data before another paid run:

1. Vary existing evidence, informative queries, inspection costs and the costs
   of wrong actions. Always inspecting first should cease to be sufficient.
2. Add failed commands, stale observations and recovery paths with real state
   transitions and independent terminal verification. Reserve new combinations
   and mechanisms before collecting model results.
3. Pair action and outcome questions on the same observed state. Opposite outcomes
   for identical partial observations represent uncertainty; preserve them instead
   of deduplicating into one supposedly certain label. Keep related worlds together
   in data splits.
4. Compare **forecast supervision alone**, **reward-only Proximal Policy
   Optimization**, and **reinforcement learning plus a proper forecast loss**.
   The supervised comparison tests whether reinforcement learning adds anything
   beyond the extra executed outcome labels.

Use common starting weights, splits and declared rollout/update budgets. Count
branch executions and additional forecast computation explicitly. Retain general
replay and probability-quality checks; report all arms and seeds. A first paid
pilot should have roughly a $50–$60 allocation within the existing authorization,
after the richer environment and collection path pass their gates. No rental
has been started for this work.

The training collector must sample from its current policy and retain exact
encoded inputs, menus and sampled likelihoods. These greedy probe traces are not
current on-policy training data. This prototype uses direct Docker and a fixed
authored candidate catalog; the earlier Harbor adapter is a separate integration.
Live Harbor training and an unattended language-model proposer remain unfinished.

The direction is consistent with the public interface's emphasis on small
judgments from supplied context. It is our proposed experiment, not a claim about
Jev's undisclosed recipe. Sources: [TypeSafe primitives](https://docs.typesafe.ai/primitives)
and [Harbor custom agents](https://docs.harborframework.com/core-concepts/agents/custom-agents).

## Reproduce

With the repository Python environment and Docker running:

```sh
python -m unittest test_evidence_shell -v
python -m tool_lab.evidence_shell --output output/my-evidence-qualification
# Requires the original supervised model service on port 8766:
python -m tool_lab.evidence_shell_probe --data output/my-evidence-qualification \
  --output output/my-evidence-probe
```

Output directories must be new; failures are preserved without automatic retries.
Containers use a pinned Python image, an unprivileged user, a read-only root,
temporary working storage and no network. No host directory or credentials are
mounted. Public [receipts](../results/evidence-shell-v1/) record bounds, source
identities and aggregate outcomes; full fixtures and trajectories remain local.
