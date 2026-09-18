# Learning to inspect software before making a forecast

We now have a working software-inspection environment, an execution-backed data
factory, and 24 small trained models. The strongest result is a useful boundary:
learning when to buy evidence works, but a simple empirical planner still beats
our neural policies. Extra forecast practice is not a consistent improvement.

This is an exploratory study with development corrections disclosed below.
It does not establish TypeSafe's private Reinforcement Learning for Calibrated
Decisions recipe. The new neural policies use hashed visible text and evidence
counts, with no pretrained language encoder. A separate export connects these
records to our [four-billion-parameter model](larger-model.md).

## What the data actually contains

We extract functions and human-written doctests from the MIT-licensed
[TheAlgorithms/Python](https://github.com/TheAlgorithms/Python/tree/a381578994d545e44f26afabbd2303746a2dc358)
repository. Each candidate is the original function or a syntax-tree mutation.
Mutations are selected before seeing their test outcomes. Unchanged and mutated
programs can both pass: mutation intent is never a correctness label.

| Unit | Count |
| --- | ---: |
| Eligible functions before verification | 431 |
| Accepted source functions | 369 |
| Source modules | 285 |
| Connected source groups | 254 |
| Source categories | 25 |
| Accepted program variants after audit | 7,793 |
| Passing program variants | 2,317 |
| Evidence views per variant | 7 |
| Total related evidence views | 54,551 |

These are **369 functions from one repository**, with correlated mutations and
evidence views. Replaying millions of episodes does not create millions of
independent programs. Source categories are mostly educational algorithms,
not 25 production software environments.

The initial accepted pool required **506,550 candidate-check executions**, using
two distinct Python hash seeds. There were 190 quarantined candidate proposals;
a subsequent audit excluded one additional candidate with a memory-limit error.
Original-source checks and rejected proposals add work beyond that count.
An independent replay of 128 candidates matched all 8,716 checked executions.

Each function has at least four distinct literal-input, literal-output upstream
doctests. The original implementation must satisfy those tests. Additional
perturbed inputs use the original program as a differential reference. That
verifies compatibility with a pinned implementation; it does not independently
prove that the implementation or its mathematics is correct.

Execution workers have no network or host mounts, a read-only root, a non-root
user, and bounded processor time, memory and process count. Harmless containment
probes passed. This is containment for this corpus, not a proof against all
hostile code. Programs are never executed on the accelerator machine that
holds provider credentials.

## What an episode means

The model sees a contract, a candidate function and one executed check. It can:

1. Buy more upstream example checks.
2. Buy checks on perturbed inputs.
3. Buy an explicitly announced copy of the original check.
4. Stop and report a probability that the **complete fixed suite** passes.

At most two purchases are allowed. Prices vary independently of the outcome.
Every actual report is on a grid from zero to one in increments of 0.05. The
terminal score is `1 - (probability - outcome)²`; inspection costs are subtracted
separately. The outcome is zero or one, determined by execution.

The complete suite includes initial, purchasable and permanently hidden checks.
One observed failure establishes failure of that event. Passing a subset does
not establish success. Inspection changes the information available, not the
candidate or the event being forecast. That distinction keeps the forecast's
meaning stable while the policy chooses actions.

The environment replays cached, verified executions. Purchases have simulated
costs; millions of training interactions are not millions of new container runs.
Hidden results, source paths, mutation ancestry and labels are excluded from
policy inputs. The controller receives the outcome to compute rewards.

For example, a faulty harmonic-series implementation passes an initial check
on a negative input, which returns an empty list. A later check on six terms
reveals that it returns `1, 2, 3, ...` instead of `1, 1/2, 1/3, ...`. The policy
must decide whether another check is worth its cost before seeing that result.

## The evidence is deliberately incomplete

| Evaluation set | Initial check passes, but full suite fails | All purchasable checks pass, but hidden checks fail |
| --- | ---: | ---: |
| Held-out source groups | 131 / 294 | 31 / 194 |
| Held-out strings and ciphers | 289 / 415 | 35 / 161 |

The denominators are candidates passing the stated visible checks. This is a
designed mutation population, not an estimate of a real repository's bug rate.
It gives us plausible-looking failures and cases where further tests change a
forecast. A copied result supplies no new evidence.

All functions and variants from one module stay together. Before training, we
also joined modules sharing an identical function body and alternative solutions
to the same Project Euler problem. Training has 173 connected groups; validation
has 25; the ordinary test has 24; the family holdout has 32. This does not exclude
all semantic near-duplicates or contamination in pretrained language models.

## Reward-only forecasts were not enough

The first experiment uses Proximal Policy Optimization to choose both inspections
and probability reports. It compares ordinary terminal rewards, equally sized
extra report exercises on visited terminal states, and exercises on broader
evidence/price states. A direct outcome-label reference sees the broader exercise
stream. Each condition has three training seeds.

| Method | Held-out-group Brier score | Held-out-family Brier score |
| --- | ---: | ---: |
| Terminal rewards only | 0.1888 | 0.1552 |
| Extra report practice on visited states | 0.1701 | 0.1463 |
| Extra report practice across broader states | 0.1943 | 0.1595 |
| Direct outcome-label training | 0.1182 | 0.1004 |
| Simple evidence-frequency reference | **0.0830** | **0.0968** |

Lower Brier score is better. It measures squared forecast error against actual
outcomes, combining calibration and informativeness. These are means across
three seeds and seven common evidence states per candidate, not evaluations
only on the states a policy chose to visit. Full calibration bins and log losses
are included with each result.

Several report policies converged to nearly constant probabilities. A proper
scoring reward has the right optimum in expectation; it does not guarantee that
a finite, randomly initialized policy will learn to use evidence. We cannot
claim that broader reward exercises helped in this experiment.

## A practical hybrid does learn useful inspection

The follow-up separates forecast learning from acquisition learning. Every
condition starts with the same supervised predictor for its seed. A separate
policy learns which evidence to buy using Proximal Policy Optimization. Forecasts
are frozen or updated with direct outcome labels. The two extra-practice
conditions have equal additional label presentations.

| Forecast updates during interaction | Group-test reward | Family-test reward | Group-test common-state Brier score |
| --- | ---: | ---: | ---: |
| Frozen predictor | 0.8648 | 0.8906 | 0.1182 |
| Train on terminal states | **0.8890** | **0.8998** | 0.1114 |
| Repeat terminal-state practice | 0.8848 | 0.8909 | **0.1052** |
| Additional broader-state practice | 0.8818 | 0.8942 | 0.1058 |
| Empirical transition planner | **0.9174** | **0.9192** | 0.0830 |

Higher reward is better. It includes forecast quality and paid inspection costs.
The planner estimates evidence transitions from training candidates and plans
over the small action space. It does not receive held-out labels while choosing
actions. It is an important, stronger baseline.

With the predictor frozen, learned acquisition improves group-test reward by
**0.0334** over stopping immediately with that same predictor; the improvement is
0.0299 on the family holdout. The broader-practice hybrid almost never buys the
announced duplicate, but it does not consistently beat matched terminal-state
practice. These experiments support useful information acquisition; they do not
show that our calibration training outperforms simple alternatives.

Both studies use 614,400 root episodes per reward-trained model, sampled from
the same 5,544 training candidates. All initializations, streams, costs, selected
checkpoints and per-candidate predictions are retained. Checkpoints in the first
study are selected by validation common-state Brier score; the hybrid uses
validation workflow reward. The hybrid is an exploratory follow-up, not a
controlled comparison changing only one factor from the first study.

## Reproduce and inspect

```bash
python -m pip install -r requirements-calibration.txt
python -m inspection_lab.download
python -m inspection_lab.demo
```

The download includes source snapshots, data, execution receipts, small trained
checkpoints, traces, predictions and checksums. It does not include large-model
foundation weights. The demo prints the code, revealed evidence, forecast, action
probabilities and eventual outcome. It takes the most likely action; published
evaluation integrates all possible action paths exactly.

```bash
python -m inspection_lab.verify \
  --data output/pretrained/first-instinct-software-inspection-v1/curated \
  --run output/pretrained/first-instinct-software-inspection-v1/reward
python -m inspection_lab.verify \
  --data output/pretrained/first-instinct-software-inspection-v1/curated \
  --run output/pretrained/first-instinct-software-inspection-v1/hybrid --hybrid
```

All 24 selected models reconstructed their predictions and metrics exactly in
the local verification. The verifier also checks matched candidate/price streams,
shared hybrid initializations, source snapshots and group separation.

To rebuild the data with a local Docker engine, then train from scratch:

```bash
python -m inspection_lab.fetch
python -m inspection_lab.check_isolation --output output/isolation.json
python -m inspection_lab.build --out output/inspection-raw --workers 6
python -m inspection_lab.curate --raw output/inspection-raw --out output/inspection-data
python -m inspection_lab.train --data output/inspection-data --out output/inspection-reward
python -m inspection_lab.hybrid --data output/inspection-data \
  --init output/inspection-reward --out output/inspection-hybrid
```

The raw build's original source snapshot omitted the imported mutation helper.
A lineage addendum records its hash and verifies that it was unchanged from
the pre-experiment repository revision. Future builds snapshot that helper too.
The initial trainer pilot was stopped to condition the value network on available
actions, which differ between interaction and forecast-only exercises. The
reported replacement run preserves the same data and splits. See the full
[development protocol](software-inspection-protocol.md).

## What this makes worth investigating next

The [completed language-model pilot](software-outcome-model.md) now improves
forecasts over a simple evidence-frequency reference and provides an interactive
local demo. Previously trained small inspectors also improve reward when paired
with those forecasts, although the empirical planner still chooses evidence
better. The next question is whether training acquisition with the large
forecaster, and adding independent source repositories, improves that result.

The export has 36,189 training views after length checks, containing 34.6 million
input tokens. The pilot's selected checkpoint processes 4.57 million of those
tokens. Overlong inputs are excluded, never silently truncated. Seven views of
one program still represent one program variant.

The next data expansion should add repositories, dependency-bearing tasks,
independently authored tests and actual bug fixes. Keep a representative audit
stream alongside selectively collected difficult cases. A model that chooses
which labels to acquire also changes the data on which its confidence is judged.

[SWE-smith](https://swesmith.com/) is relevant for repository environments and
execution-verified mutations. [TOUCAN](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)
offers a much larger pool of tool trajectories, with Apache 2.0 licensing on its
dataset card. Those trajectories would need their own schema, success and source
audits before becoming calibrated-outcome targets. They have not been imported
into this experiment.

We subsequently [audited one TOUCAN shard](toucan-data-audit.md): 12,963
trajectories across 425 servers. The audit found concrete declaration, identifier
and split-design issues to resolve before scaling training; it did not execute
those tools or turn model-judge scores into verified rewards.
