# A proposer, a selector, and an executable verifier

This prototype implements the user's proposed loop: a language model proposes
concrete next commands, First Instinct selects one, Harbor executes it, and the
next proposal is conditioned on the actual command output. A separate verifier
checks the final files/database. No model decides its own reward.

This is new work after the frozen outcome-v2 study. It changes none of that
study's code, data, checkpoints or reported results. It is not the separate
SQLite hidden-world qualification proposed in `outcome-v3-data-design.md`.

## Components

- `tool_lab.generate`: three small authored mechanisms: scoped nested JSON
  configuration changes, conjunctive SQLite updates, and filtered CSV sums.
  Entity/number seeds produce variants, not independent skills. The broader
  proposals in `harbor-task-families.md` are not all implemented here.
- `tool_lab.harbor_agent.ProposalChoiceAgent`: a Harbor external agent. The
  proposer gets only the instruction, previous chosen commands and their actual
  outputs, plus remaining steps. It returns 2–5 description/command pairs.
  The exact command is included in every selector option; menu order is shuffled
  independently. A finish action is always added by the harness.
- `tool_lab.protocol`: binds proposals to the exact observed history and checks
  candidate uniqueness and probability distributions.
- `tool_lab.report`: joins Harbor's independently verified terminal outcome to
  the saved selection probabilities and computes returns after each decision.

The proposer mailbox can be served by a Codex subagent in the user's current
OpenAI account. This adapter does not invoke a paid model API or secretly use a
scripted oracle as the proposer. Each observed-state request and model-authored
response is retained. No response means a timed-out/error trial, not invented
candidates. A mailbox by itself is not an unattended language-model service;
a proposer process must service it.

The resident supervised 9B service supplies selector probabilities. `argmax`
collects deterministic evaluation trajectories; `sample` draws from the exact
returned distribution for exploratory collection. No weights are updated here.
Future Proximal Policy Optimization must collect with its current training
policy, retain the exact encoded option order and sampled likelihoods, and
optimize this same selection behavior. Historical greedy demonstration traces
must not be relabeled as current on-policy training data.

## Qualification bounds, declared before Harbor execution

Generate seeds 101 and 102 in each family: six engineering fixtures. Run each
with Harbor's reference-solution agent and no-op agent (12 trials). Run at most
three live proposer/selector trials, one per family. Allow one additional
infrastructure diagnostic trial: **16 Harbor trials maximum**, no automatic
retries. Each live trial permits at most six decisions, one local model request
per decision: **18 selector predictions maximum**. Reference/no-op runs use no
model. This phase performs no training and rents no compute.

Each command has a 15-second timeout; proposer requests have a 180-second
timeout and the whole agent a 900-second timeout. Containers use one CPU and
512 MB, no network, and an unprivileged command user. Verification uses a fresh
separate container, receiving only `/app/data` artifacts. Model/proposer code
runs outside the container. No user repository, Docker socket, credentials,
solution, or verifier is mounted into the agent's working directory. Harbor's
ordinary log/artifact mounts still exist; the separate verifier does not trust
agent-written reward files. Docker artifacts are deleted after each trial.

The observation policy records up to 1,200 stdout and 600 stderr characters
per command, with explicit truncation flags. Entire task instructions, menus,
and the retained history are supplied to the selector. An overlength model
request fails rather than silently truncating the decision input.

## Reward and checks

Terminal reward is 1 for verified success and 0 otherwise. Each selected,
executed command costs 0.01; finish costs zero. Total reward is terminal success
minus command costs. Intermediate command exit status is an observation, not
a success label. Infrastructure errors are separate from ordinary task failure.

The verifier checks the whole declared data directory, protected file hashes,
exact JSON values (including types), or database schema and all rows. SQLite
integrity is checked. Unexpected files and symbolic links fail verification.
The report keeps failed attempts, entire menus, selected commands, probabilities,
model metadata and actual stdout/stderr. It does not claim tensors resident in
the demo process have been independently attested.

Reference solutions must pass; no-op controls must fail. These establish that
fixtures are solvable and tests actually require a change. Unit controls also
check wrong-target changes, protected-state damage, type confusion, stale
proposals, invalid probabilities, and attempted reward-file spoofing.

## What scaling would require

Many seeded tasks can be generated, but useful diversity requires different
tools, dependency structures, failure modes and hidden information—not merely
renamed files. Group splits by mechanism before expansion. Keep a genuinely
untouched final set, a starting-selector baseline, and a fixed proposer version.

Measure candidate coverage separately from selector success: a missing useful
command is a proposer failure. Compare the selector with random selection and
a fixed/simple selector on matched task/proposer settings. Branch evaluations
from identical state snapshots can establish which offered actions lead to
success under an explicitly fixed future proposer/policy. Branching is an
additional measurement budget, not free ground truth for unexecuted actions.

Future calibrated forecasts need those controlled branches/repeated outcomes.
One successful trajectory alone does not supply a probability of success. A
proposal generator that adapts during training also changes the environment;
record and control its version, prompt, observations, randomness and work.

## Reproduction

Use a separate Python environment with `requirements-harbor.txt`. The frozen
model runtime should not be changed. Docker must be running.

```sh
python -m unittest test_harbor_selection -v
python -m tool_lab.generate --output /new/task-directory --seeds 101 102
harbor run --config /path/to/job.json
python -m tool_lab.report --job /path/to/completed-job --output /new/report.json
```

Harbor 0.23.0 is pinned. The generator pins the Python base-image digest.
References: [Harbor task format](https://docs.harborframework.com/core-concepts/tasks/overview),
[external agents](https://docs.harborframework.com/core-concepts/agents/custom-agents),
and [separate verifiers](https://docs.harborframework.com/core-concepts/tasks/verifier).
