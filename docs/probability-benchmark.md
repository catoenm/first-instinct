# A controlled test of decision probabilities

This benchmark asks whether probability estimates respond to information in the
right way. Each situation has a fully specified random mechanism and an exact
answer. We do not use another language model's opinion as the target.

The [training study](forecast-audit.md) uses small numeric models. The same facts
are also available as short text requests for Jev or another language-capable
model. Numeric and language models receive different representations; this is
not a broad capability or speed leaderboard.

## One example

A switch is equally likely to start on or off. A sensor that is correct 75% of
the time reads “on.” The probability the switch is on is now 75%.

- A photocopy of that reading supplies no new information: still 75%.
- Another sensor, also 75% reliable and independent given the switch's state,
  reads “on”: the probability becomes 90%.
- Changing the price of looking at another reading does not change what the
  current evidence says. It can change whether looking is worthwhile.

A constant 50% answer would pass the copy and price invariance checks. That is
why the benchmark also measures absolute probability error and how correctly
forecasts respond to fresh evidence.

## What's included

There are 512 base groups: 256 ordinary-source groups and 256 reversed-source
groups. Each has six situations, giving **3,072 unique numeric cases**:

| Situation | Expected behavior |
| :--- | :--- |
| Initial observation, additional source unseen | Report from currently visible evidence |
| Fresh source reads zero | Update in the direction and amount implied by its reliability |
| Fresh source reads one | Update in the direction and amount implied by its reliability |
| An unseen offered source is an announced copy | Offered-source metadata adds no observed evidence |
| That copy is revealed | No probability change |
| Inspection price increases | No probability change; inspection can become less attractive |

A reversed source announces that it is more often wrong than right. Its reading
still supplies information, interpreted in reverse. The benchmark does not
test deception about source reliability.

Two text views of each case—prose and a record with explanatory prose—produce
**6,144 requests**. Their numeric parameters round-trip to the same float32
values, and they describe the same visible evidence. The numeric models receive
no wording-invariance score because they do not read either text view.

Related variants and wordings share a base case. They are not 6,144 independent
problems. The generated prices and reversed sensors also include conditions
outside the training range; report failures by family rather than only a pooled
average.

## Request and answer files

The files are in
[results/forecast-audit-v1/evaluation/benchmark](../results/forecast-audit-v1/evaluation/benchmark).

- `requests.jsonl.gz`: identifiers, grouping metadata, visible numeric inputs,
  text state, and question instructions. No target probability or hidden label.
- `answers.jsonl.gz`: expected probability and optimal inspection decision.
- `manifest.json`: counts and checksums.

The runner constructs requests from only the text state and instructions. It
does not read the answer file until scoring saved responses. Question identifiers
and benchmark metadata are not included in the model input.

## Try Jev when access is configured

The runner uses TypeSafe's documented
[evaluation endpoint](https://docs.typesafe.ai/api) and
[Noul probability primitive](https://docs.typesafe.ai/primitives/noul).
It requires no extra client package beyond the laboratory's existing dependencies.

```bash
python -m pip install -r requirements-calibration.txt

# Local preview: no key needed, zero network requests.
python -m calibration_lab.jev_benchmark

# Configure TYPESAFE_API_KEY in your local environment without committing it.
# This sends at most 24 new requests, keeping related cases together.
python -m calibration_lab.jev_benchmark --execute --limit 24

# Re-run to continue: completed requests are skipped, errors remain recorded.
python -m calibration_lab.jev_benchmark --execute --limit 24

# Evaluate already collected responses; no new network requests.
python -m calibration_lab.jev_benchmark --summarize
```

The default model is `jev-latest`; `--model` can name another available version.
Results record the requested model, returned model identifier, raw successful
response, token usage, request digest, time and latency. An alias is not an
immutable model version. Keep collection dates and model identifiers attached
to conclusions.

Execution stops on the first failed request so an access or service error does
not trigger thousands of calls. A sanitized failure record is retained, without
credentials or arbitrary server error bodies. Use another `--output` path when
changing the dataset or requested model; incompatible responses are not reused.
The default files stay under the repository's ignored `output/` directory.

Partial results explicitly count missing coverage and complete pairs. Missing
or failed requests are never scored as zero-probability answers. Mock responses
are used only in offline tests; they are not Jev measurements.

## What the result can establish

This test can expose inconsistent probabilities, insensitivity to useful
evidence, and sensitivity to prices or wording that supply no new evidence.
It also requires reading a specified probabilistic mechanism and reasoning about
it. A failure may involve arithmetic or language interpretation as well as
calibration. It cannot establish broad performance on ordinary routing or
classification workloads.

Passing the test would demonstrate the measured behavior on these cases. It
would not identify the model's architecture, training data or reinforcement
learning algorithm. We should measure Jev before making claims about Jev.
