# Consequence-learning pilot

The prospective [protocol](../../docs/consequence-training-v1-protocol.md) and
[freeze](freeze.json) were published before candidate inference. This experiment
updates the released Qwen3.5-9B language adapters using exact consequence targets
from executed database commands, mixed with retained general training examples.
It is supervised consequence learning; it is not a new reinforcement-learning run.

The data contains 2,592 training questions, 864 public-history validation questions,
2,512 general replay examples and 622 general retention questions. All reservation
questions concern one previously studied mechanism. New parameter combinations
are diagnostic, not evidence of transfer to independent mechanisms.

One H100 was rented at $3.49/hour with a six-hour provider stop deadline. The
prospective limit is 180 optimizer steps, with early stopping and retention gates.
The starting checkpoint remains eligible. A completed run will be reported whether
it helps, fails, or selects the unchanged starting checkpoint.

The [reporter](../../puffer_lab/consequence_report.py) is separate from the frozen
training code. It recomputes probability metrics, checks prediction identities,
and replays every before/after decision trajectory in the qualified environment.
Its checks reject state corruption and malformed probabilities. There are no
completed training results in this directory yet.
