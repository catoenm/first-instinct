# General decision demo

The browser demo is a compact retro game of **Snake**. On each turn, the game
sends its visible board, recent moves and three candidate directions to the
trained language model. It executes the model's returned choice and displays
its probabilities. A wall or body collision ends the game.

The released demo uses the supervised checkpoint selected at step 2,742.
Start a local server with the instructions below. See the
[measured supervised results](https://github.com/catoenm/first-instinct/blob/main/docs/general-supervised-results.md). The
[four completed reinforcement-learning runs](https://github.com/catoenm/first-instinct/blob/main/docs/general-reinforcement-results.md)
did not reliably improve held-out decisions, so this demo keeps the supervised
checkpoint. The [combined model release](https://github.com/catoenm/first-instinct/releases/tag/general-decisions-v1)
includes all selected and latest adapters, predictions, and experiment receipts.

Download the archive and its checksum manifest from that release, check the
archive's SHA-256, then extract it and run `python verify.py` inside the extracted
directory. Use the current repository checkout for the Snake interface; the
older release archive contains an earlier demo. Install `requirements-scale.txt`
in the checkout and pass the extracted `runs/supervised` directory to `--run`
below. No frontend build is needed.

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
local development demo, not a public hosting setup.

### Keep the page available while using a remote model

Run the model server on port 8767 on the GPU machine and forward that port to
your computer with SSH. Then serve the page locally without loading any weights:

```bash
python -m general_lab.serve --upstream http://127.0.0.1:8767
```

The page stays accessible if the model disconnects, shows **Offline**, and returns
no predictions until the connection is restored. Press **Start model** to retry.
Both the page and the forwarded model port must remain bound to localhost.

## Try it

1. Press **Start model** to watch it choose successive moves.
2. **Pause** stops automatic play after any move already in flight.
3. **One move** asks the model for a single decision; **New game** resets the board.
4. Expand **View model input** to inspect the request that produced the displayed
   move probabilities, or the starting request before the first move.

The game removes reverse turns, supplies mechanically computed immediate
collision and food-distance observations, and executes the returned choice.
There is no heuristic fallback or collision veto. These observations are not
learned consequence forecasts. This is an interactive example, not an independent
benchmark of game-playing or general decision competence.

The footer identifies the loaded model and its training
status. Unfinished runs are labeled **Training preview**. If a reinforcement
run selects update zero, the demo identifies it as the supervised starting
checkpoint: completed reinforcement updates elsewhere in the run do not change
the selected model.

## Limits of this experiment

The browser asks a new choice question after every move. The underlying
interface also supports arbitrary caller-defined choices, yes/no propositions
and ordered levels. It scores supplied answers without generating an explanation.
Use the command below and edit its input file to supply your own questions and
answer definitions.

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
