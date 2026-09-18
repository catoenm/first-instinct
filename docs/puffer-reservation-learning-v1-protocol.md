# Small-policy learning check, before another language-model run

Declared after the execution qualification passed and before training. The C
core and SQLite qualification remain frozen. This study asks whether an agent
can learn useful behavior in that one environment. It does not measure language
understanding, arbitrary commands, Jev parity, or a change to our 9B checkpoint.

Use a **local PyTorch implementation of Proximal Policy Optimization** over the
same C core as the PufferLib adapter. The native CUDA PuffeRL trainer is not run
in this check: this Mac has no NVIDIA processor, and no rental is necessary to
test the learning signal. Preserve this distinction in every result.

## Fixed training

- Two independent seeds: 41 and 73. No replacement runs or hyperparameter sweep.
- Public observations: the exact 39-dimensional adapter vector. The sampled
  hidden-world ID is unavailable to the network. Nine fixed semantic actions;
  all available in every world. This is a small environment policy, not an
  open-vocabulary language model.
- Two shared hidden layers of 64 tanh units, with action and value outputs.
- 64 concurrent episode states; 32 rollout steps; 64 updates: 131,072 sampled
  transitions per seed. Four optimization epochs, minibatch 256, learning rate
  0.0003, clipping 0.2, entropy coefficient 0.02, value coefficient 0.5, maximum
  gradient norm 0.5. Discount 1, generalized-advantage factor 0.95. Stop remaining
  optimization epochs in an update if mean approximate divergence exceeds 0.05.
- CPU only, at most two compute threads; total time cap 15 minutes per seed,
  including evaluation. Maximum 160,000 environment transitions per seed,
  including evaluation. Numerical failure ends that seed and remains reported.

Training samples 32 contexts uniformly: inspection cost 0.25 or 2 credits;
atomic cost 1 or 4; horizon 3 or 6; priors over the first four worlds proportional
to `[1,1,1,1]`, `[7,1,1,1]`, `[1,7,1,1]`, or `[1,1,1,7]`. Combined faults have
zero training probability. Other action prices and rewards remain unchanged.

## Selection and diagnostics

Validation uses eight contexts: inspection costs 0.5 or 1.5 credits; atomic
costs 1.25 or 3.5; horizons 3 or 6; uniform prior over the first four worlds.
Enumerate supported worlds and weight by the public prior. Evaluate greedily at
update zero and after every eight updates. Select the largest mean validation
return, keeping the earlier checkpoint on a tie. Save every selection result.

After selection, evaluate the initial and selected policy once on:

1. Eight familiar context combinations (training costs/horizons, uniform prior
   over the first four worlds).
2. Eight combined-fault contexts with a uniform prior over the two previously
   absent stock-plus-account worlds. This changes both fault composition and
   the supplied prior; it is one diagnostic, not an isolated causal factor.

Compare finish-only, the declared public continuation, and uniform-random
actions (16 repetitions with fixed seed 2026). These baselines use identical
goals and horizon rules. They are not equivalent compute budgets for training.
There are no policy/parameter changes after these diagnostics are opened.

Retain all sampled action probabilities/log likelihoods, encoded observations,
rewards, terminal flags, value estimates and computed targets by update, along
with actual counts, gradients, losses, checkpoint hashes and selection history.
Report each seed separately and failures explicitly. A positive result supports
learning within this mechanism; it does not establish that its data improves
the language model or transfers to Harbor.
