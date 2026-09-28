# Evolving values in a game world

The user wants a different experiment: a genetic algorithm determines a value
function, an agent lives and learns in a game world using that function, and
evolution selects functions that help the agent thrive. The previous database
policy-iteration proposal is deferred. The user approved the reward-evolution
design. The first implementation and local pilot use small neural networks;
no foundation model or rented GPU is required.

## Two timescales

**Between lifetimes:** a population of inherited functions is selected, combined,
and mutated according to the success of the agents that learned with them.

**Within a lifetime:** each agent observes the world, acts, and learns a behavior
policy using its inherited function as part of the learning process.

An important design choice remains explicit:

| Evolved component | What it means | What the agent learns during life |
| --- | --- | --- |
| Internal reward function | Which experiences feel good or bad | A policy and predictions of future internal reward |
| Value predictor or its initialization | How promising a situation/action is under a fixed objective | A policy and, optionally, further value updates from experience |

The approved first version evolves internal rewards: priorities concerning
energy, food, danger, and novelty. It must not be described as a calibrated
prediction of future return. If the intended subject is a literal value
predictor, keep the environment reward fixed and evolve the predictor instead;
do not silently substitute one experiment for the other.

## A first world

Use a small two-dimensional foraging world:

- Moving and waiting both consume energy; reaching zero ends the life.
- Eating replenishes energy. Food is finite and replenishes over time.
- Obstacles and hazardous terrain create choices between short risky routes
  and longer safer routes.
- The agent observes nearby terrain and its own energy. World randomness and
  any unseen resources remain private to the environment.

Start with one agent per world. Multiple competing agents, reproduction inside
the simulation, tools, and changing seasons are later experiments. The initial
question is whether evolution discovers functions that help a small learning
agent acquire useful survival behavior.

The current pinned Puffer game adapters cover Lights Out and 2048, which lack
this resource/survival mechanism. Keep the game interface compatible with fast
batched execution; first qualify a simple world and its invariants locally.
Reuse the existing small-policy learning components where appropriate. A large
language model is not required for this first test.

## The evolutionary loop

1. Initialize a small population of bounded reward-function parameter vectors.
   Begin with interpretable weights rather than unrestricted generated code.
2. For each genome, start fresh small policy networks from the same set of
   initialization seeds. Give each the same learning-step budget.
3. Let the agent learn over several lifetimes using the genome's internal reward.
4. Freeze learning and evaluate behavior on separate world seeds, measuring
   actual survival and resource sustainability.
5. Retain successful genomes, recombine and mutate them, and repeat for a fixed
   number of generations.

Initially inherit only the genome, not trained policy weights. That isolates
whether evolution found better learning incentives. Inheriting learned policy
weights is a separate comparison, since it adds cross-generation skill transfer.
Use common development worlds and learner initialization seeds across genomes;
keep final worlds unopened until the population is selected. Reusing development
worlds permits selection overfitting, which the final worlds must expose.

## Fitness is separate from internal reward

The outer genetic algorithm needs a fixed definition of "thriving," such as
survival duration with an independently measured energy/resource condition.
The world computes this fitness from its state. A candidate cannot change its
fitness formula or earn selection merely by outputting larger internal rewards.

Evolution therefore discovers useful internal incentives under a chosen external
criterion; it does not eliminate the need to define success. Cap and normalize
candidate rewards so reward magnitude alone is not the experimental treatment.
Learning capacity, observation access, steps, and evaluation worlds must match.

Compare evolved rewards with a fixed survival reward, a hand-designed reward,
and a random-search population using the same total agent-learning budget.
Also evaluate agents without within-lifetime learning. These controls distinguish
evolutionary benefit, ordinary learning, and simply trying many candidates.

Report survival distributions across worlds and learner seeds, not just the best
agent or summed internal reward. Count genomes, learning steps, lifetimes, and
held-out evaluations separately. Stop/revise if survival stays at random-agent
levels or apparent gains disappear on separate worlds.

## Fixed first pilot

The world is a connected 12×12 map with 12 wall cells, eight hazardous cells,
and eight food plants. The agent sees a 5×5 patch in three channels (food, walls,
hazards), plus its current energy. It begins with 24 energy, capped at 40.
Every step costs one energy, including waits and collisions. Hazard occupancy
costs three more. Food restores 14, regrowing 20 steps after consumption. Eating
on arrival precedes the starvation check. A lifetime ends at zero energy or
128 steps. Observations do not expose seeds, distant cells, or regrowth timers.

The inherited genome is seven nonnegative weights summing to one, applied to
bounded features: alive, eating, energy change, visit novelty, avoided hazard,
avoided death, and avoided collision. Penalty features are negative when incurred.
Novelty declines as the inverse square root of visits and pays nothing for
waiting or bumping a wall. All internal rewards receive the same 0.1 scale.
This is a restricted, interpretable search space, not discovery of arbitrary
new motivations.

The outer fitness is **steps lived + mean energy / maximum energy**. Energy
contributes at most one extra unit, so survival dominates. It is computed from
environment state and never reads the genome or the agent's reward prediction.

The policy and learned state-value critic share two 64-unit hidden layers,
reusing the small-policy implementation. Both learn through Proximal Policy
Optimization. Each genome starts fresh under learner seeds 17 and 29 and
receives 32,768 transitions per seed: 32 environments × 32 steps × 32 updates.
Training uses three epochs, 256-sample minibatches, learning rate 0.0003,
clipping 0.2, entropy weight 0.02, value-loss weight 0.5, discount 1, and
advantage trace decay 0.95. These weights and optimizer states are not inherited.

Evolution uses six genomes per generation for three generations. It retains
the best two, selects parents through size-three tournaments, blends their
weights, and mutates log weights with standard deviation 0.65 before normalizing.
An independent random search tries the same 18 candidate slots. Repeated elites
are retrained and counted as repeated genomes, not newly discovered functions.

Each candidate is selected on 16 common development maps for each learner seed.
Selection is written before opening 64 final maps per seed. Actions are sampled
from the learned policy with repeatable, role-specific randomness. Final maps
share world rules with training; they do not measure transfer to new mechanisms.

Fixed survival rewards and a hand-designed mixture each receive the same total
training budget as a whole search: **1,179,648 transitions per method** across two
seeds. Each fixed policy is therefore trained 18 times longer than a candidate.
Their early 32,768-transition checkpoints also provide matched-lifetime controls.
Report both comparisons. Untrained, uniform-random, and wait controls use no
learning. The full pilot totals **4,718,592 training transitions**, plus bounded
development and final evaluation.

The pilot earns a larger follow-up only if its final evolved policy beats random
search and both matched-lifetime fixed controls by at least **eight survival
steps**, with no learner-seed loss against random search. Report the much longer
fixed-policy controls regardless of this gate. Two learner seeds and one search
seed provide a pilot result, not a robust claim of evolutionary advantage.

The process has a 25-minute internal deadline and runs inside the existing
4-GiB/30-minute/512-MiB-swap-growth local guard. Tests and a separate-seed smoke
run must pass first. Stop and report incomplete work if a limit is reached;
do not silently extend or use a GPU. Save source identities, genomes, ancestry,
training counts, checkpoints, and action traces. No paid compute is allocated.

```sh
python -m games_lab.evolve_rewards --output output/evolved-values-v1-pilot
```

On this Mac, invoke that command through `release_lab.local_guard`.

This direction has prior research in evolved intrinsic rewards, including
[Singh, Lewis, Barto, and Sorg (2010)](https://www.ece.uvic.ca/~bctill/papers/ememcog/Singh_etal_2010.pdf).
The interesting empirical question here is which inherited incentives produce
robust learned behavior under our world dynamics and learning budget.
