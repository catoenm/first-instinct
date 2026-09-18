# Better outcome data before a longer run

This is a development diagnosis and a plan for a separate experiment. The
current [frozen comparison](general-training-v1-protocol.md) stays unchanged.
Nothing described below has produced an additional trained checkpoint.

## What the current validation data says

After 100 reinforcement updates, better decisions are not a consistent result.
The extra forecast objective reduces probability drift relative to reward-only
training, but neither hybrid run improves on the supervised starting model.

| Training arm | Expected validation reward ↑ | Squared error against exact event probability ↓ |
| --- | ---: | ---: |
| Supervised start | 0.19771 | 0.00206 |
| Reward, seed 47 | 0.17194 | 0.00859 |
| Reward + forecast, seed 47 | 0.17793 | 0.00293 |
| Reward, seed 53 | 0.20193 | 0.00607 |
| Reward + forecast, seed 53 | 0.18367 | 0.00245 |

These are **latest-update validation diagnostics**, not selected-checkpoint
test results. Each evaluation has 64 root worlds and four correlated evidence
states per world. The exact planner's validation reward is 0.23567. The single
small reward gain does not establish a reliable improvement across training
seeds. Retaining update zero is an intended outcome of checkpoint selection.

There are no nonfinite training metrics. Every optimizer step clips the combined
gradient norm, which motivates checking the relative actor, value, forecast and
replay gradients before changing objective weights. Clipping by itself is not
proof that a particular component dominates or that the trainer is broken.
The diagnostic values and source hashes are in
[the validation receipt](../results/general-rl-data-audit-v1/validation-diagnostic.json).

## What is missing from the data

Reconstructing all 12,800 training worlds across the two seeds gives 6,979 cases
where the exact best first action buys evidence, 2,321 that deny the event,
2,319 that affirm it, and 1,181 that abstain. All eight domain wordings are
represented roughly evenly. Only 861 worlds have a gap below 0.02 reward units
between the best two first actions. See the
[training-only coverage audit](../results/general-rl-data-audit-v1/training-coverage.json).
Reproduce it without loading a model:

```sh
python -m general_lab.audit_training_coverage --output output/training-coverage.json
```

The obvious gap is therefore not a missing majority of action types. It is that
all these worlds have the same mechanism: one fixed hidden event and two
conditionally independent readings. Actions buy information or end the episode;
they never change the underlying event.

The next data should include actions that change the world, delayed responses,
duplicate side effects, dependent evidence, missing outcomes, and sequences
where an early choice changes which later choices remain possible. Oversample
some close decision boundaries while retaining the original broad distribution
as a separately measured slice. Count independent worlds and mechanisms, not
only question rows or paraphrases.

For a probability target, specify an event and a deadline. “Will this retry
leave exactly one completed job by the deadline?” is different from “Which
action should I take?” Several actions can each have high success probability;
those probabilities do not form a single distribution over the action menu.
An outcome forecast that includes later actions must also name the continuation
policy. Otherwise its target changes when the policy changes.

## External environments worth integrating

These are integration candidates, not data used in the current model. Pin and
audit their versions before collecting training examples.

| Candidate | New capability | Outcome and integration caveat |
| --- | --- | --- |
| [TextWorld](https://github.com/microsoft/TextWorld) | Local language quests, inventory, prerequisites and irreversible state changes | Best inexpensive first external mechanism. Use executable facts or win/loss predicates. Candidate commands must come from visible information; hidden facts and state-dependent admissible-command lists can leak the answer. |
| [BrowserGym / MiniWoB++](https://github.com/ServiceNow/BrowserGym) | Local form, selection and navigation workflows | A bounded menu of visible targets fits our model. The current adapter maps positive raw reward to success; audit exact completion predicates before treating partial credit as a binary outcome. |
| [ToolSandbox](https://github.com/apple/ToolSandbox) | Stateful tool dependencies, clarification and missing information | Use exact database assertions, not an aggregate similarity score. An offline subset needs scripted user responses and exclusion of external-search dependencies. |
| [Current τ benchmark](https://github.com/sierra-research/tau2-bench) | Policy-constrained service workflows and database mutations | The repository now contains τ³-bench; older tasks have changed. Restrict outcome labels to database/environment assertions, rather than model-judged natural-language checks. Full runs also require user simulation. |

TextWorld has an MIT project license, but its documented dependencies include
separately licensed components; review its [setup and notices](https://github.com/microsoft/TextWorld).
Its supported Python range and native dependencies also differ from this
project's current environment, so use an isolated runtime. BrowserGym is
Apache-2.0 and MiniWoB++ is MIT. ToolSandbox uses Apple's own permissive license
text and separate notices. The current τ package declares MIT. These statements
describe the upstream projects, not blanket licensing for every bundled asset
or dependency.

Primary implementation references: TextWorld's
[information interface](https://textworld.readthedocs.io/en/stable/textworld.html),
BrowserGym's [MiniWoB validator](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/src/browsergym/miniwob/base.py),
ToolSandbox's [evaluation design](https://github.com/apple/ToolSandbox), and the
[τ evaluator](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/evaluator/evaluator.py).

## Gates before more paid training

1. Build a small executable pilot. Log legal actions, visible inputs, chosen
   actions and their sampling probabilities, actual transitions, costs, and
   verified outcomes. Include failed trajectories. Separate private verifier
   state from every model-visible field.
2. Reproduce trajectories and labels from pinned code and seeds. Group all
   variants, retries and counterfactuals of a world in one split. Keep whole
   mechanism combinations or quest structures for a new holdout.
3. Collect an independent forecast-audit stream, including states the policy
   avoids. A lack of observed failure is not evidence that an untried action
   succeeds. Repeated stochastic trials need independent outcome draws.
4. Freeze a new protocol before querying its heldout models: supervised control,
   reward-only control and reward-plus-forecast treatment, with equal interaction
   budgets and explicit labeling of the additional forecast data. Compare native
   actions with a controller using predicted outcomes and declared costs.
5. Only then consider a short, bounded training pilot. Set a validation-based
   stopping rule and check retained general abilities. A larger model, more
   updates and spending the rest of the budget are not automatic next steps.

The separately implemented retry pilot is described in
[retry-environment-pilot.md](retry-environment-pilot.md). It is a local simulator,
not external customer traffic or evidence of broad real-world reliability.
