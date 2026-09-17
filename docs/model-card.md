# First Instinct v0.1.0 model card

## Intended use

An educational model for choosing the first tool to call, given a request and
at least two tool descriptions. Intended for local experiments in data quality,
supervised fine-tuning, evaluation, and dynamic option scoring.

This is not a general assistant, code generator, or tested production router.
It does not execute tools or generate their arguments. It has no trained refusal,
clarification, or “none of these” option.

## Architecture and provenance

- Base: [microsoft/deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small), revision `a36c739020e01763fe789b4b85e2df55d6180012`.
- Six encoder layers, 768-dimensional hidden representations; token vectors are averaged with padding excluded and then normalized.
- One shared linear scoring layer, 768 weights, no bias; softmax over candidate scores.
- **141,305,088 total trainable parameters** during full fine-tuning, including the embedding matrix.
- 32-bit floating point weights; saved in Safetensors format.
- Inputs: `state`, `question`, and `options` with identifiers and descriptions. Only state, question, and descriptions enter the encoder.
- Context limit: 512 tokens for each complete request/question/option pair. Longer input is rejected.

## Training and selection

The source is [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE),
a public synthetic tool-use dataset, pinned at revision
`6bda777c88d21e5a204703c1ee45597a8fa4f734`. Source rows are filtered and converted
to first-call decisions. The final experiment has 166 training examples and 44
validation examples, separated by the overlap rules in the
[experiment report](experiment.md).

Supervised fine-tuning used AdamW for three passes, with seed 7, encoder learning
rate 0.00002 and scorer learning rate 0.001. The checkpoint from pass two had the
lowest validation log loss and is the released model. Training ran on a personal
Apple M5 Max; no cloud training service was used.

## Evaluation

The selected checkpoint matched **29 / 34 synthetic reference labels (85.3%)**
on the fresh test partition. Word overlap matched 24 / 34; the frozen-encoder
baseline matched 18 / 34. Log loss was 0.2494 and the Brier score was 0.1767.
These are results from one small experiment, not a broad capability benchmark.

The initial exploratory test informed a correction to question wording and was
retired to development data. The final test came from a previously unused
partition. The full history is in [the report](experiment.md#the-question-wording-mistake).

## Limitations

- All training questions concern first-tool selection. Other question types are untested.
- Synthetic references may be wrong or may omit other acceptable first actions.
- Exact normalized schema and near-duplicate request isolation do not establish semantic task-family isolation.
- Pretraining contamination was not measured.
- Candidate descriptions are trusted input to a small experimental scorer; robustness against misleading descriptions was not evaluated.
- Probabilities are **uncalibrated**, and the model must choose an available option even when all options are unsuitable.
- Minimum memory, throughput, tail latency, multilingual performance, and production reliability were not evaluated.

## Download and verification

[Release v0.1.0](https://github.com/catoenm/first-instinct/releases/tag/v0.1.0)
contains the selected full checkpoint, tokenizer, original training manifest,
traces, and licenses. The uncompressed encoder and tokenizer occupy approximately
574 MB; allow about 1.2 GB of free space for download and extraction.

Run `python download_checkpoint.py` from the repository root. The downloader
checks the pinned archive digest in [releases/v0.1.0.json](../releases/v0.1.0.json)
and every artifact listed in the original model manifest before installing it.
The release is loaded locally with remote model code disabled.

## License

First Instinct code and new model contributions: [MIT](../LICENSE).
Microsoft's encoder and tokenizer: MIT. ToolACE data: declared Apache-2.0.
The release includes upstream notices and licenses; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
