# Reservation environments and a small learning check

This lab connects a fast C simulator to independently executed SQLite
transactions. It also implements the PufferLib 5.0 environment interface and
includes a separate local PyTorch trainer using Proximal Policy Optimization.

**The published experiment trains a 7,370-parameter numerical policy, not Qwen.**
The 9B checkpoint is unchanged. Native CUDA PuffeRL training has not been run.
Read the [results and limitations](../docs/puffer-reservation-results.md),
[environment protocol](../docs/puffer-reservation-v1-protocol.md), and
[learning protocol](../docs/puffer-reservation-learning-v1-protocol.md).

## What the agent does

Reserve one unit each of two inventory items for a customer. The customer may
not exist, either item may be out of stock, and commands consume a public cost
and a limited number of turns. The nine actions inspect, transact, replenish,
create the customer, undo this request, or finish. A sequential transaction can
leave partial writes; an atomic transaction rolls back on failure.

The policy receives 39 numerical features containing only public observations,
the disclosed prior, prices, remaining turns, and committed-write receipts.
It cannot see the sampled hidden world. All nine actions remain available.
This fixed action interface is an environment test, not the arbitrary
question-and-choice interface of the main language model.

## Reproduce locally

Run from the repository root with a C compiler and the repository's Python
dependencies installed. The qualification itself uses only the Python standard
library and C compiler; the learning check and full receipt audit need NumPy
and PyTorch. Recorded training used PyTorch 2.14.0 on CPU with two threads.

Verify the published evidence without training or executing new SQLite tasks:

```sh
python -m puffer_lab.audit --manifest-only
python -m puffer_lab.audit
python -m unittest test_puffer_reservation test_puffer_evidence -v
```

The full audit checks frozen source hashes, recomputes the conditional targets,
reconstructs all 262,144 saved training transitions and all saved evaluation
actions, checks rewards and terminal flags, recalculates advantage targets, and
checks initial/selected checkpoint probabilities against saved rollouts. Replay
counts are separate from the original training and evaluation budgets.

To execute a **new qualification**, choose a fresh output directory:

```sh
python -m puffer_lab.qualify \
  --output output/reservation-reproduction \
  --references results/puffer-reservation-v1/reference
```

This compiles the simulator, checks the real pinned PufferLib header, and
executes the primary SQLite branches and eight guards. The published run also
contains six earlier debug attempts and a separately recorded extension of 64
random guards. The reproduction command does not manufacture those receipts.
The output directory must not already exist.

To run the two fixed small-policy training seeds on the newly compiled core:

```sh
python -m puffer_lab.train_small \
  --library output/reservation-reproduction/libreservation.so \
  --output output/reservation-learning-reproduction
```

The `.so` name is also used for the compiled dynamic library on macOS. Training
is bounded to 64 updates and 131,072 sampled transitions per seed, with a
160,000-transition total cap including evaluation and a 15-minute time cap.
These commands do not call a model service, rent hardware, or change Qwen.

## Files and integration boundary

| File | Role |
| --- | --- |
| `reservation_core.h` | Fast state transitions, public features, rewards |
| `sql_oracle.py` | Actual independent SQLite commands and protected-state verifier |
| `qualify.py` | Differential execution, replay, conditional outcome/cost labels |
| `reservation.h` | PufferLib environment lifecycle adapter |
| `check_adapter.c` | Headless check against the real pinned upstream header |
| `native.py`, `native_bridge.c` | Local Python driver for the same C core |
| `train_small.py` | Separate CPU Proximal Policy Optimization implementation |
| `audit.py` | Frozen evidence verification and recorded-action replay |

The adapter targets PufferLib commit
`6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`. Its four profile presets match the
qualification. The Python trainer additionally varies costs and priors through
the C bridge; that training-context sampler is not implemented in the native
PufferLib adapter. Integrating the adapter into a full PufferLib checkout and
running its CUDA trainer remains untested. The checked-in upstream headers
serve the headless interface check only, with their original licenses.

The SQLite executor uses fixed authored commands. It does not generate shell
commands or use Harbor. The separate [Harbor pilot](../tool_lab/README.md)
covers actual sandbox command selection. Transfer from this environment to
language decisions or Harbor has not been measured.
