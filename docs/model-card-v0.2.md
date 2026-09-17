# First Instinct v0.2.0 model card

## Intended use

An educational model that chooses among described options for three known task
families: first-tool selection, sentence relationships, and emotion recognition.
The latter two support categorical and yes/no questions about the same text.

Inputs are `state`, `question`, and `options` containing identifiers and
descriptions. Outputs are a choice and uncalibrated probabilities over the
available options. The model does not generate text, execute a tool, or produce
its arguments. It must choose an available option, even when none is suitable.

## Architecture and provenance

- Base encoder: [Microsoft DeBERTa-v3-small](https://huggingface.co/microsoft/deberta-v3-small), revision `a36c739020e01763fe789b4b85e2df55d6180012`.
- Starting checkpoint: the published [First Instinct v0.1.0](model-card.md), verified against its release and artifact hashes.
- Six encoder layers, 768-dimensional hidden representations; mean pooling excludes padding, followed by layer normalization.
- One shared linear scoring layer with 768 weights and no bias. Softmax converts option scores to probabilities.
- **141,305,088 total parameters**, stored as 32-bit floating point values in Safetensors files.
- Only state, question, and option descriptions reach the encoder. Option identifiers remain outside model inputs.
- Maximum **512 tokens per complete state/question/option pair**. Longer inputs are rejected.
- Each option is scored separately by the same network. The model does not jointly compare option contents; adding an option changes the softmax normalization. More options require more encoder computation.

The interface supports changing the supplied options. That does not establish
reliable interpretation of arbitrary new tasks or arbitrary option descriptions.

## Training and data

The continuation uses **3,567 questions from 1,407 source examples**, with 637
validation questions from 277 sources. These include synthetic ToolACE references,
human-annotated Stanford Natural Language Inference sentence pairs, and six
single-label GoEmotions categories. Every selected human source yields a category
question and a pair of yes/no questions with opposite answers. All versions of a
source remain in the same partition.

See the [experiment report](multitask-experiment.md) for quotas, duplicate and
partition rules, source versions, negative-category balancing, and the limitations
of deriving binary questions from a single annotation.

Each of three seeds receives six training passes, 672 optimizer updates, and
21,402 question visits. Effective batches contain 32 questions; eight-question
microbatches accumulate gradients before each update. Training uses AdamW with
encoder learning rate 0.00002, scoring-layer learning rate 0.001, weight decay
0.01, and gradient norm clipped at 1. Tasks receive equal expected loss weight.

All training runs use the same released starting weights. A control updates only
the 768 scoring weights while keeping that encoder fixed. Per-run checkpoints
are selected by lowest mean validation log loss across the five tasks, including
the initial checkpoint. The released full-model checkpoint is chosen by that
same validation criterion across the three seeds, before final test predictions.

Training ran on a personal Apple M5 Max with 128 GB of unified memory. No cloud
compute, model service, reinforcement learning, or probability calibration was
used. Minimum training memory and cross-machine reproducibility of numerical
results have not been established.

## Evaluation

The final evaluation contains **1,144 questions from 424 source examples** and
**360 opposite-answer pairs**. Reported uncertainty resamples whole source
examples, with shared resamples across compared models. Questions derived from
one source are correlated and are not counted as independent samples.

Canonical wording, predeclared paraphrases, and a removed-question control are
all evaluated. These are new examples within three known task families;
paraphrases are new wording of those same tasks. The report includes every seed,
per-task accuracy, probability loss, and how often both paired answers are right.

The released checkpoint is **seed 17, pass 5**, chosen by validation loss.
Its mean accuracy across the five tasks is **81.1%**; it answers both questions
correctly on **230 of 360 pairs (63.9%)**. With the predeclared question
paraphrases these become **77.7%** and **176 of 360 pairs (48.9%)**.

Across all three full-training seeds, mean task accuracy is **79.5%**, versus
**57.5%** for the fixed-encoder comparison. Paired accuracy averages **61.8%**
versus **20.7%**. These are the main experimental comparison; the released
checkpoint's figures are not a substitute for reporting all seeds.

## Limitations

- No evaluation of previously unseen task families, long contexts, code writing, continuous ratings, multi-step agents, or real tool execution.
- Synthetic tool labels can be wrong or omit other acceptable actions. Human labels can be ambiguous; single-label emotion negatives are an adaptation, not a new human audit.
- Public benchmarks may have appeared in the base encoder's pretraining; contamination was not measured.
- Tool grouping excludes specified exact schema and request overlaps, but does not establish semantic family isolation.
- Probabilities are uncalibrated. A score of 0.9 is not demonstrated to mean 90% reliability.
- No learned abstention, clarification, or refusal behavior. Misleading candidate descriptions and adversarial inputs were not evaluated.
- All five tasks share one encoder and scorer; improvements in their mean can conceal individual task regressions.
- The selected seed is a deployment convenience chosen by validation loss; the primary experimental comparison reports all three seeds.
- This is an independent conventional described-option classifier, not a reproduction of Jev's architecture or proprietary training method.

## Download and verification

[Release v0.2.0](https://github.com/catoenm/first-instinct/releases/tag/v0.2.0)
contains the selected full model, tokenizer, original manifest, training traces,
evaluations, and license notices. From the repository root:

```bash
python download_checkpoint.py --version v0.2.0
python decision_model.py \
  --run output/pretrained/first-instinct-v0.2.0 \
  --input examples/emotion.json
```

The downloader verifies the archive checksum and each artifact listed in the
model manifest before installation. Inference uses local files with remote
model code disabled. Allow approximately 1.2 GB for the download and extraction.
The default downloader version remains v0.1.0 for reproducing the continuation's
starting point; request v0.2.0 explicitly as above.

## Credits and license

Original project code: [MIT](../LICENSE). Microsoft's encoder and tokenizer:
MIT. ToolACE and GoEmotions declare Apache-2.0. Stanford Natural Language
Inference source and adapted question data retain Creative Commons
Attribution-ShareAlike 4.0 terms and are not relicensed as project code.
The release preserves those notices and license texts; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
