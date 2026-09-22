# First Instinct

**A language model that chooses from the answers you define.**

Give it a situation, a question, and the allowed answers. First Instinct scores
those answers and returns a probability distribution. You supply the question
and answer definitions on each request.

[Download the model](https://github.com/catoenm/first-instinct/releases/tag/general-decisions-v1) ·
[Run the demo](#run-the-demo) ·
[Results](docs/general-supervised-results.md) ·
[Documentation](docs/README.md)

## The interface

```json
{
  "state": "Expedite parcels that are at least two days late. This parcel is three days late.",
  "questions": {
    "next_action": {
      "type": "choice",
      "instructions": "Which action follows the policy?",
      "criteria": {
        "expedite": "Expedite the parcel",
        "wait": "Wait"
      }
    }
  }
}
```

The response contains the selected answer and probabilities for each option.
The same model supports **choices**, **yes/no questions**, and **ordered scores**;
see [examples of all three](examples/general-decisions.json).

Each question takes one model pass, with no generated explanation. Your application
executes any chosen action. Selection probabilities describe the offered answers;
they are not guarantees that an action will succeed.

## Run the demo

The browser demo is **Snake, played by the model**. Each turn supplies the board
and three directions, executes the selected move, and displays its probabilities.
You can pause, advance one move, or inspect the exact request. The game supplies
immediate collision observations; it does not override the model's choices.

Download and verify the archive from the
[model release](https://github.com/catoenm/first-instinct/releases/tag/general-decisions-v1),
then extract it and run its `verify.py`. Use the **current repository checkout**
for Snake; the release archive contains an earlier interface.

With your Python environment active, run from this checkout:

```sh
python -m pip install -r requirements-scale.txt
python -m general_lab.serve \
  --run /path/to/extracted-release/runs/supervised --device auto
```

Open [localhost:8766](http://127.0.0.1:8766/). Foundation weights download on
first use. Check the [hardware requirements and setup guide](docs/general-demo.md)
before loading the 9B model.

To ask your own questions, edit the [example file](examples/general-decisions.json)
and run:

```sh
python -m general_lab.interface \
  --run /path/to/extracted-release/runs/supervised \
  --input examples/general-decisions.json --device auto
```

## Model and results

The released checkpoint adapts **Qwen3.5-9B** on **350,857 examples** using
low-rank adapters throughout the language network. Training updates 43.3 million
adapter parameters while keeping the original foundation matrices frozen.

| Evaluation | Questions | Original Qwen | First Instinct |
| --- | ---: | ---: | ---: |
| Full test mixture | 17,277 | 63.3% | 78.1% |
| Public sources held out from adaptation | 13,077 | 66.2% | 73.5% |
| Reserved executable task combinations | 4,000 | 50.4% | 71.7% |

Values are question-weighted accuracy using the same constrained-answer interface
for both models, without generated reasoning. The public-source rows are a subset
of the full test; reserved combinations belong to a separate challenge set.
Some tasks regress. The [full report](docs/general-supervised-results.md) includes
task breakdowns, probability scores, uncertainty, and known failures.

The demo uses this supervised checkpoint. Reinforcement-learning experiments
have not yet qualified a replacement. Ongoing work tests whether learning from
executed actions and verified outcomes improves decisions and consequence
forecasts. See the [experiment guide](docs/README.md#active-decision-and-forecast-work)
for protocols and results.

This is an independent, Jev-inspired research project. General calibration and
equivalence to Jev have not been established.

---

[Project guide](docs/README.md) ·
[Environments and data](docs/README.md#environments-and-data) ·
[Reproduction and tests](docs/README.md#reproducing-and-extending-experiments)

Code: [MIT](LICENSE). Data and upstream dependencies:
[third-party notices](THIRD_PARTY_NOTICES.md).
