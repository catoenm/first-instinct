# Evolved rewards: first foraging experiment

**Small neural agents learned to survive in the world. Genetic search did not
establish an advantage over random-search or hand-designed rewards.** Both stages
are complete and audited. All work ran locally, with no paid compute or
foundation-model calls. This is separate from the 9B decision-model experiments.

The [protocol](evolved-values-v1.md) describes the fixed first pilot.
[Pilot receipts](../results/evolved-values-v1/summary.json) and the separate
[learning-dose diagnostic](../results/evolved-values-v1/dose-diagnostic.json)
contain full-precision metrics, counts, checkpoint identities, and source hashes.

## What was built

The agent lives on a connected 12×12 map with renewable food, walls, hazards, and
depleting energy. It sees a 5×5 patch and its energy. A 9,478-parameter network
learns action probabilities and a prediction of future **internal reward** through
Proximal Policy Optimization. It has no recurrent memory.

The inherited genome determines seven reward weights: survival, food, energy
change, exploration, hazard penalty, death penalty, and collision penalty.
Evolution selects, crosses, and mutates these weights. Neural weights start fresh
for each candidate and learner seed. Learning continues across episodes within
that candidate's training; policy weights are not inherited between generations.

Selection uses world-computed lifespan, with mean energy as a small tiebreaker.
It never reads the agent's reward prediction. This evolves motivations inside a
restricted space of authored features; it does not evolve arbitrary value functions.

## Initial controlled pilot

Six candidates over three generations were compared with 18 random-search
candidates. Each candidate learned for 32,768 transitions under each of two learner
seeds. Selection used 16 development maps. Final evaluation used 64 separate maps
under both seeds: 128 episodes, **64 unique maps**, per method.

| Reward / control | Transitions per evaluated learner | Mean lifespan / 128 | Survived full episode |
| --- | ---: | ---: | ---: |
| Evolved reward | 32,768 | 35.87 | 0.0% |
| Random-search reward | 32,768 | 35.70 | 1.6% |
| Survival reward, early checkpoint | 32,768 | 31.10 | 0.0% |
| Hand-designed reward, early checkpoint | 32,768 | 33.95 | 0.8% |
| Survival reward, longer learning | 589,824 | 107.67 | 68.8% |
| Hand-designed reward, longer learning | 589,824 | 111.32 | 74.2% |
| Untrained network | 0 | 30.18 | 0.8% |
| Uniform random actions | 0 | 30.62 | 0.8% |
| Always wait | 0 | 24.00 | 0.0% |

The two long fixed-reward conditions each consumed the same **total** learning
budget as an entire search: 1,179,648 transitions across two learner seeds. They
put that budget into two agents, rather than training 36 agents briefly. The
early checkpoints supply the separate equal-per-agent-duration comparison.

The predefined gate required an eight-step evolved advantage over random search
and both early fixed controls, without a loss to random search on either learner
seed. It **failed**: the gains were 0.16, 4.77, and 1.91 steps respectively, with
a loss on one seed. We did not launch a larger genetic search.

## Frozen rewards, longer learning

After seeing that result, a separately recorded diagnostic held the already
selected evolved and random-search reward functions fixed. Each received 589,824
transitions per learner seed. Fresh learners used the original initialization and
training streams; their early checkpoint parameters matched the original short
candidate exactly. No new genomes were proposed or selected.

All four long-trained conditions were evaluated on **64 new maps**, under the
same two learner seeds. The fixed controls reuse their original long checkpoints.

| Frozen reward | Mean lifespan / 128 | Survived full episode | Mean lifespan by learner seed: 17 / 29 |
| --- | ---: | ---: | ---: |
| Evolved | 109.95 | 72.7% | 114.81 / 105.09 |
| Random-search | 110.01 | 71.1% | 113.48 / 106.53 |
| Survival only | 105.38 | 67.2% | 105.72 / 105.03 |
| Hand-designed | 110.23 | 68.0% | 113.03 / 107.44 |

The selected evolved motivations can support competent learned behavior after
enough experience. Random-search and hand-designed motivations can too. This
post-hoc diagnostic does not change the failed original search gate. Its maps
differ from the initial final cohort, so the two tables are not a paired-map
estimate of the effect of longer learning.

## Accounting and verification

| Quantity | Initial pilot | Additional diagnostic |
| --- | ---: | ---: |
| Search candidate slots | 36 | 0 |
| Distinct searched reward vectors | 32 | 0 new |
| New trained neural models | 76 | 4 |
| Executed training transitions | 4,718,592 | 2,359,296 |
| Repeated optimizer sample presentations | 14,155,776 | 7,077,888 |
| Optimizer minibatch steps | 55,296 | 27,648 |
| Completed training episode presentations | 102,937 | 26,421 |
| Executed evaluation transitions | 92,475 | 55,753 |
| Development + final evaluation episodes | 2,304 | 512 |
| Pipeline elapsed seconds | 88.82 | 37.20 |

The initial pilot stepped 16,902 distinct training-world seed identities across
two learner streams, reused across candidates. The diagnostic uses subsets of
those same streams. These are randomized maps of **one mechanism**, not 16,902
different kinds of task. Elites retrained under identical seeds count as repeated
genomes. Optimizer sample presentations repeat each collected transition for
three learning epochs; they are not additional experience.

An independent scalar transition implementation replayed all 2,816 evaluation
episodes and 148,228 actions across both stages. It checked energy, food, hazards,
survival, and fitness. Audits also checked saved model identities, fresh initial
weights, selection, counts, and the original failed gate. All eight diagnostic
animation traces and their probabilities were checked against saved models.
Gradients were not independently replayed.

The suite completed 1,208 tests with 29 skips, including ten focused environment
and learning tests. Training peaked below 330 MB resident memory with no added
swap. Both stages used source revision
`e3f6a12efd109f53e667383c4a50e264133d1e73`; immutable source copies and raw outputs
remain under the ignored `output/evolved-values-v1-*` directories. Full hashes
are in the linked aggregate receipts. No training remains running.

## Next experiment

Change **when evolution judges a learner**, before enlarging the search. A
follow-up should compare rewards after enough learning, track early and late
survival, and repeat entire searches under new search and learner seeds. Keep an
equal-total-budget random-search control. Reserve world-rule changes such as
different food scarcity or regeneration times for a separate transfer test.

This pilot demonstrates a working two-level learning loop and a learning-budget
bottleneck. It has not demonstrated an evolutionary advantage, transfer to new
mechanisms, or open-ended continuous self-improvement.

Implementation: [`games_lab/foraging.py`](../games_lab/foraging.py) and
[`games_lab/evolve_rewards.py`](../games_lab/evolve_rewards.py), reusing the existing
small policy, advantage calculation, and clipped policy loss. The original pilot
entry point is `python -m games_lab.evolve_rewards --output <new-output-directory>`;
on this Mac, run it through `release_lab.local_guard` as in the protocol. The dose
diagnostic reused `learn(..., updates=576, short_checkpoint=True)` with frozen
selected genomes and evaluation role `long-dose-diagnostic`.
