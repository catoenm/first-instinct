# First Instinct

**A small language model for structured decisions, tool selection, and consequence forecasts.**

Give it a situation, a question, and the allowed answers. It returns a choice and
probabilities. Questions and answer definitions come from the caller; they are
not a fixed list of categories.

[Demo setup](docs/general-demo.md) ·
[Download the model](https://github.com/catoenm/first-instinct/releases/tag/general-decisions-v1) ·
[Experiments and results](docs/README.md) ·
[MIT license](LICENSE)

## What works today

The released model adapts **Qwen3.5-9B** with small trainable matrices throughout
the language network. One supervised pass over **350,857 examples** improved
accuracy from **63.3% to 78.1%** on **17,277 held-out questions** under the same
constrained-answer interface. See the [full measurements](docs/general-supervised-results.md),
including tasks that did not improve.

It supports user-defined choices, yes/no questions, and ordered scores. It scores
the supplied answers directly, without generating an explanation. Applications
execute the chosen action; the model does not execute a tool by itself.

The original supervised checkpoint remains the selected model. Later
reinforcement-learning experiments have not qualified a replacement. Current
work pairs execution-verified decisions with outcome forecasts and tests longer
training. See the [latest experiment](docs/paired-capacity-v1-launch.md) and
[completed decision/forecast comparison](docs/oracle-capacity-completion-v1-results.md).

This is an open research project inspired by Jev. It does not establish Jev
parity, reveal its training recipe, or guarantee calibrated probabilities on
unfamiliar questions. An action's selection probability is different from its
probability of succeeding.

## Try it

Download the archive and checksum manifest from the
[model release](https://github.com/catoenm/first-instinct/releases/tag/general-decisions-v1).
Verify the checksum, extract it, and run its `verify.py`. Foundation weights
download separately on first use. The 9B model needs suitable hardware; check
[the setup and memory notes](docs/general-demo.md#run-a-saved-checkpoint) first.

From this checkout, with your Python environment active:

```sh
python -m pip install -r requirements-scale.txt
python -m general_lab.serve --run /path/to/extracted-release/runs/supervised --device auto
```

Open [localhost:8766](http://127.0.0.1:8766). For your own questions, use the
[supplied input example](examples/general-decisions.json):

```sh
python -m general_lab.interface \
  --run /path/to/extracted-release/runs/supervised \
  --input examples/general-decisions.json --device auto
```

A minimal request looks like this:

```json
{
  "state": "Expedite parcels that are at least two days late. This parcel is three days late.",
  "questions": {
    "next_action": {
      "type": "choice",
      "instructions": "Which action follows the policy?",
      "criteria": {"expedite": "Expedite the parcel", "wait": "Wait"}
    }
  }
}
```

## Work on the project

| Location | Purpose |
| --- | --- |
| [`general_lab/`](general_lab/) | General decision model, interface, and demo |
| [`scale_lab/`](scale_lab/) | Foundation loading, adapters, scoring, and inference |
| [`tool_lab/`](tool_lab/) | Executable environments, verified data, training, and evaluation |
| [`release_lab/`](release_lab/) | Data admission and release checks |
| [`tests/`](tests/) | Offline regression tests |
| [`docs/`](docs/README.md), [`results/`](results/) | Experiment guides and published evidence |

Earlier encoder, calibration, inspection, and game experiments remain available
through the [research index](docs/README.md). Saved source snapshots in `results/`
are historical evidence, not the place to edit current implementations.

Install `requirements-release-data.txt` to run the offline suite:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

Code is MIT-licensed. Dataset and upstream project terms are recorded in
[third-party notices](THIRD_PARTY_NOTICES.md).
