# Curiosity without game rewards

**Completed:** [audited results](curiosity-foraging-v1-results.md). The fixed
schedule finished; the prediction-improvement gate failed. No extension or
larger search was launched. The run's original protocol/source copies are preserved.

The user chose **no game scores in training or evolutionary selection, while
allowing internally generated curiosity**. This replaces survival-based selection
for the next experiment. The completed [reward-evolution pilot](evolved-values-v1-results.md)
and all its source/receipt hashes remain unchanged.

## Information boundary

The agent receives the same 76 visible inputs: a local 5×5 food/wall/hazard patch
and its energy. Energy is a visible resource, not a payoff. It receives no food
bonus, death penalty, survival bonus, score, absolute position, seed, or global
map. Game-authored reward features are discarded by an observation-only adapter.
It also receives no explicit end-of-episode signal: after death or the fixed
horizon the adapter resets the world, and the next observation reflects that
reset. This observable discontinuity is unavoidable in this setup and can itself
attract curiosity. It is not claimed to remove all information about survival.

Three independent small world predictors learn the next observation from the
current observation and chosen action. They use bootstrap samples of actual
observed transitions. The policy's internal reward is a bounded mixture of:

1. **Observation novelty:** inverse square root of the count of the visible
   observation, with energy quantized to eight intervals. No hidden coordinates.
2. **Predictor disagreement:** variance among the three next-observation predictions.
3. **Learning progress:** positive reduction in next-observation prediction error
   on the collected batch after the predictor update. This is in-sample progress;
   it can overfit, which is why selection uses separate fixed probes.

Each signal is divided by a running root-mean-square scale (decay 0.99), clipped
to five and divided by five. Three nonnegative genome weights sum to one, then
the reward receives a common 0.1 scale. Internal rewards are therefore still
present; **this is not learning without an objective**. The actor and value critic
learn with the existing clipped policy update, discount 0.99, trace decay 0.95,
three epochs, learning rate 0.0003, entropy weight 0.02, and value-loss weight 0.5.
The critic forecasts internal curiosity returns, not game success.

## Selection and controls

Evolution minimizes mean next-observation squared error on a fixed bank of
8,192 transitions collected by a uniform random policy on separate development
worlds. The probes never train the predictors or policy. All candidates see the
same probe bank. This measures predictive generalization to a particular
distribution; a lower loss does not imply better gameplay.

Two independent search seeds (4101, 9107), each with four candidates over three
generations, are compared with the same number of random-search candidates.
Each search uses two fresh learner seeds (137, 241). The initial four genomes are
identical across search methods within a repeat; later candidates differ.
Evolution keeps two elites, uses size-three tournaments, convex crossover, and
log-weight mutation with standard deviation 0.65. Only reward weights are inherited.

Every candidate gets **262,144 transitions per learner**: 256 updates of 32
steps in 32 worlds. This is eight times the per-candidate dose in the first pilot.
Predictors train once per rollout with independent bootstrap samples, using
512-row minibatches and learning rate 0.001. They have three 64-unit hidden-layer
networks. The actor/critic is the existing 9,478-parameter network.

Fixed controls use novelty only, disagreement only, progress only, an equal
mixture, and uniform random actions. All train world predictors on the same
number of interactions; random actions do not train the actor. These are equal
**per-learner experience** controls. Each search spends twelve times as much
training as one fixed condition; only evolution versus random search is matched
for total search budget. This cost difference must be reported, not hidden.

All selected identities and final-update checkpoints are sealed before any game
performance is evaluated. There is no task-based early stopping, ranking,
checkpoint choice, reward design revision, or hyperparameter adjustment mid-run.

## Frozen evaluation and continued learning

Use a separate final prediction bank and 64 familiar game maps per learner.
Also evaluate on a previously untrained scarcity/danger condition: four food
plants, sixteen hazard cells, food regeneration after 32 steps, and five extra
energy lost per hazardous step. Original values were eight plants, eight
hazards, 20 steps, and three energy. This is changed parameters of one mechanism,
not transfer to unrelated games.

For each of the four search winners and five fixed conditions, continue learning
for exactly **65,536 transitions per learner** in the changed condition. Preserve
policy, predictors, both optimizer states, novelty counts, normalization, and
random-generator states. Re-evaluate the same familiar and changed map banks
with the same evaluation randomness. The pre-adaptation checkpoint is the
no-further-learning comparator. No measured task outcome changes this schedule.
Prediction probes are fixed and never consumed as training examples.

The prospective advancement gate is at least **5% lower final familiar prediction
error** than both random search and the best fixed control in each search repeat,
with no learner-seed loss against random search. This is solely an intrinsic
learning gate. Report survival, food, adaptation changes, and forgetting after
the run as independent outcomes, including failures. A gameplay benefit requires
its own honest empirical evidence; passing the prediction gate is insufficient.

## Bounds and verification

The full schedule contains 48 search candidate slots, 96 candidate learners, ten
fixed-control learners, and eighteen continuations. It consumes **28,966,912
training transitions**: 27,787,264 before selection and 1,179,648 afterward.
Three probe banks add 24,576 random-policy transitions. Final gameplay has
4,608 episodes on 128 unique maps (64 familiar and 64 changed), reused across
methods, learner seeds, and before/after measurements. Repeated optimizer
presentations, predictor updates, unique observation keys, world seeds, and
training interactions are distinct units.

Qualify the score boundary, genuine predictor/policy updates, continuing-value
targets, retained learning state, and the complete stage order locally. Use a
separate-namespace smoke run. Then run one guarded CPU process: 25-minute internal
deadline, 30-minute outer deadline, 4 GiB resident limit, 512 MiB maximum added
swap. No foundation model, GPU rental, paid model calls, or subagents. Stop on
limits, numerical failures, or a failed gate; do not extend or silently retune.
Save exact source copies, protocol, input probe banks, selection, checkpoints,
training counts, and evaluation actions. Independently audit final dynamics,
selection from prediction losses, source/checkpoint lineage, and consumption.

```sh
python -m release_lab.local_guard --output output/curiosity-foraging-v1-guard -- \
  python -m games_lab.curiosity --output output/curiosity-foraging-v1
```

Related work: [curiosity from prediction](https://proceedings.mlr.press/v70/pathak17a.html),
[the 54-environment curiosity study](https://pathak22.github.io/large-scale-curiosity/),
and [exploration by disagreement](https://pathak22.github.io/exploration-by-disagreement/).
This pilot is an explicit small comparison, not a claim to invent reward-free learning.
