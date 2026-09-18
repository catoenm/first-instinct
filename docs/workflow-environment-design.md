# Executable workshop workflows

This environment adds a different kind of decision problem to First Instinct:
complete a small resource-dependent job within four or five decisions. It is a
finite, authored research task, not evidence of unrestricted real-world ability
or a reconstruction of Jev's private training recipe.

The implementation is [`general_lab/workflow_environment.py`](../general_lab/workflow_environment.py).
It runs entirely on the processor using the Python standard library. No model,
external service, network request, or paid call generates its ground truth.

## What actually happens

A workshop starts with one part of unknown condition and one assembly casing.
The part can be ready, need preparation, or be broken. A spare may or may not be
available. These two initial facts are independent draws from probabilities
given in the public state; there are at most six hidden worlds per configuration.

The agent can inspect, acquire a spare, prepare, assemble, submit, or abandon the
job. Each action consumes a tick. Acquiring a stocked spare replaces and discards
the old part. Spares need preparation. Assembly and submission are separate
actions: an assembled but unsubmitted unit misses the deadline.

Three mechanism choices interact:

- Preparation may require a recorded inspection of the current part.
- Preparation applies either to the station, surviving replacement, or to the
  current part, becoming invalid when it is replaced.
- Failed assembly either leaves the casing reusable or destroys it permanently.

An unknown prerequisite does not silently remove a menu option. Attempting that
action fails visibly and consumes its stated cost and time. A broken part cannot
be repaired by preparation. All allowed-action menus use only visible state.

Executable state assertions determine the mutually exclusive terminal outcomes:
`completed`, `damaged`, and `unfinished`, worth 100, -100, and -40 reward cents
before costs. Costs are integer multiples of a public unit price; acquisition
costs two to five units, assembly two, inspection/preparation/submission one,
and abandonment zero. There are no language-model judges or text-match rewards.

## Forecasts have explicit continuations

For each offered action, a question specifies the deadline and a fixed adaptive
continuation: submit an assembled unit; inspect an unknown part; replace a known
broken part if acquisition remains; satisfy required inspection and preparation;
otherwise assemble. If replacement has already failed, abandon a broken part.
The continuation uses visible observations and never extends the deadline.

One fresh hidden-world draw, conditioned on the observed prefix, is executed
through that continuation. Its terminal category labels an outcome question;
its total future action cost labels a cost question. The same execution supplies
both labels. Future costs include the offered action and exclude costs already
paid. The full hidden tape and final state are stored separately in replay
receipts. They never enter the model's public input.

The cost menu is a public superset constructed from bounded action sequences and
documented possible responses. It does not enumerate hidden tapes or expose
their probabilities. Some cost options can have zero probability under the
fixed continuation. A terminal action or action on the last tick has one known
cost and can bypass the model.

Exact conditional outcome and cost distributions are available only to the
verifier. A controller can calculate expected terminal utility minus expected
future cost from the two predicted marginals; it does not need to assume they
are independent. These are values under the stated continuation, not promises
about every possible future policy. Replanning after each action is a separate
controller that must be evaluated by executing its own trajectories.

## Holdouts are whole mechanism combinations

The eight three-factor combinations are assigned before model use. Four even
parity combinations train the model; two others are reserved for validation and
two for testing. Every pair of factor values occurs in training. All costs,
initial probabilities, deadlines, seeds, hidden draws and explored paths within
one combination retain the same group and split. Generated indices cycle evenly
through the split's mechanism combinations.

| Split | Inspection required | Preparation scope | Failed assembly |
|---|---|---|---|
| Train | No | Station | Reusable casing |
| Train | No | Part | Casing destroyed |
| Train | Yes | Station | Casing destroyed |
| Train | Yes | Part | Reusable casing |
| Validation | No | Station | Casing destroyed |
| Validation | No | Part | Reusable casing |
| Test | Yes | Station | Reusable casing |
| Test | Yes | Part | Casing destroyed |

Every split admits all three outcome classes and both casing rules. The split
has a deliberate asymmetry: validation has optional inspection and testing has
required inspection. Both are represented during training, but conclusions
should report these held-out interactions separately. Thousands of generated
rows still represent a small finite set of authored mechanisms.

## Interface and checks

`EpisodeAdapter` supports `reset`, `observe`, `legal_actions`, `input`, `step`,
and `close`. `forecast_inputs` is public-only; `forecast_bundle` returns paired
empirical targets plus a separate private replay object. `exact_forecast` is a
verifier function, never a training label generator.

Run the semantic checks with:

```sh
python -m unittest test_workflow_environment -v
```

The checks cover prerequisite failures and recovery, irreversible damage, actual
resource replacement, deadline accounting, hidden-state leakage, conditional
replay, paired outcome/cost labels, known-cost bypasses, cost-vocabulary coverage,
and complete mechanism separation between splits.
