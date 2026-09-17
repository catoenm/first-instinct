# Decision data

Source: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE),
declared Apache-2.0. [Paper](https://arxiv.org/abs/2409.00920).
See [third-party notices](../../THIRD_PARTY_NOTICES.md).

## Pinned training source

`decision_dataset.py` downloads `data.json` from revision
`6bda777c88d21e5a204703c1ee45597a8fa4f734` and verifies its SHA-256 checksum:

```text
ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e
```

The 11,300-row source is cached under `raw/`, outside version control. The
reported experiment samples 5,000 rows with seed 7, excluding source rows 0–99
because those were used for introductory inspection and hardware experiments.

## Conversion and modifications

We retain first-turn requests followed by a parsable single assistant tool call
and one matching tool result. Each example must offer at least two uniquely
named tools. Unsupported formats, ambiguous call syntax, and inputs exceeding
512 tokens per request/question/option pair are explicitly excluded. The
converter parses source text without executing it.

The model input contains the request as `state`, a first-tool question, and
options described by each tool's description and parameter schema. Identifiers,
reference labels, subsequent assistant text, tool results, and provenance never
become encoder input. Labels remain synthetic reference annotations; the project
does not independently verify them by executing tools.

Connected groups keep duplicate or near-duplicate requests and shared normalized
option descriptions in the same partition. All options count, including
incorrect alternatives. This guards against those specific overlaps; it does
not prove semantic tool-family separation or absence from encoder pretraining.

The first build retained 268 examples. A question-wording revision retained 266,
with 166 training, 44 validation, 22 development, and 34 fresh test examples.
The old test became development data after it informed the wording correction.
Its filename is still `calibration.jsonl`; it is **not** an untouched calibration
set. [Full history and reproduction commands](../../docs/experiment.md).

## Bundled introductory snapshot

The small `raw/8657de928d9509729defe15832a1e804c6342c9b3c68837839e49c3f391d5cf5.json`
file is the original public dataset-viewer response for rows 0–99. Its filename
is the checksum of the exact received bytes. It is not a random sample and does
not itself identify an upstream commit. Those rows were subsequently checked
against the pinned source and matched. The response URL is recorded in
`decision_data.py` and its output manifest.

Run `python decision_data.py` to convert this inspection snapshot into 84
examples under `output/decisions/`. It exists to make the schema and tiny-batch
training exercise easy to inspect; none of these 100 source rows enter the
reported held-out experiment.
