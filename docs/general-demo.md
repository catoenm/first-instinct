# General decision demo

The local browser demo presents one editable parcel example and a Run button,
followed by a short explanation of the experiment. It uses the trained language
model to score two supplied answers: expedite or wait.

The nine-billion-parameter experiment is still running. A training preview is
available in the author's workspace; it is a snapshot of supervised training,
not a completed release or a reinforcement-trained selection. Held-out results
and reinforcement-learning comparisons are pending.

## Run a saved checkpoint

With the project environment active, install `requirements-scale.txt`, then:

```bash
python -m general_lab.serve --run /path/to/saved/run --device auto
```

Open [the local decision lab](http://127.0.0.1:8766/). The saved run must contain
`run.json` and its selected adapter in `best/`. Its receipt identifies the exact
foundation model revision, which downloads separately on first use. On an Apple
Silicon Mac, `--device mps` explicitly selects the graphics processor. This
version uses full-precision weights on the Mac; the 9-billion-parameter model
needs roughly 36 GB for weights alone, plus working memory. The development
machine has 128 GB of unified memory.

The server keeps one model loaded and accepts one inference request at a time.
It binds to localhost and does not save submitted questions or state. It is a
local development demo, not a public hosting setup. The earlier software
inspection demo can continue on port 8765.

## Try it

1. Read the parcel policy and situation in the text box.
2. Select **Run** to see probabilities for expediting or waiting.
3. Change “three days late” to “one day late” and run again. You can also edit
   the policy. Editing the example clears the earlier result.

The note below the explanation identifies the loaded model and its training
status. Unfinished runs are labeled **Training preview**. If a reinforcement
run selects update zero, the demo identifies it as the supervised starting
checkpoint: completed reinforcement updates elsewhere in the run do not change
the selected model.

## Limits of this experiment

The browser deliberately shows only one simple choice question. The underlying
interface also supports user-defined choice, yes/no, and ordered-level
questions. It scores supplied answers directly, without generating an
explanation first. Use the command below and edit its input file to explore
those capabilities.

Each question independently includes the shared state and its own answer
definitions. The interface accepts up to 32 questions with 2–36 options each
and checks the 1,536-token limit for every complete prompt before inference.
An ordered score averages zero-based level numbers using model probabilities;
that average is a convenience for the supplied scale, not an independently
learned unit.

Probabilities are conditional on the state and supplied choices. High
probability is not a guarantee of correctness, and calibration on arbitrary
new questions has not been established. Action-choice probabilities are also
different from forecasts of whether an event will happen. This interface is
inspired by typed decision models; it does not establish Jev parity or reveal
Jev's private training method.

The broader interface is available without a browser:

```bash
python -m general_lab.interface \
  --run /path/to/saved/run \
  --input examples/general-decisions.json \
  --device auto
```

See the [training protocol](general-training-v1-protocol.md),
[data sources](general-data-sources.md), and
[TensorBoard guide](https://github.com/catoenm/first-instinct/blob/main/docs/training-monitor.md)
for the experiment and its monitoring.
