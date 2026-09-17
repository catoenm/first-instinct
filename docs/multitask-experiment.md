# Several decisions, one shared model

This experiment continues the published First Instinct v0.1.0 checkpoint on
three task families: first-tool selection, emotion recognition, and the
relationship between a premise and a hypothesis. The architecture stays the
same: a shared text encoder, pooled token representations, and one shared
scoring layer for described answer options.

The experiment asks whether this small model can use the question to decide
which judgment to make about the same text. It does not test arbitrary new task
families, general agent behavior, or equivalence to Jev.

## What changed from the first experiment

The first comparison trained on 166 tool-selection examples and tested 34.
Here, **3,567 training questions come from 1,407 source examples**. The final
**1,144 test questions come from 424 source examples**. Questions derived from
the same source are correlated; they are not 1,144 independent observations.

| Task | Training questions | Validation questions | Test questions |
| :--- | ---: | ---: | ---: |
| First tool to call | 327 | 97 | 64 |
| Sentence relationship, three choices | 600 | 90 | 180 |
| Sentence relationship, yes/no | 1,200 | 180 | 360 |
| Emotion, six choices | 480 | 90 | 180 |
| Emotion, yes/no | 960 | 180 | 360 |
| **Total** | **3,567** | **637** | **1,144** |

The six emotion labels are anger, disgust, fear, joy, sadness, and surprise.
Sentence relationships distinguish a hypothesis supported by the premise, a
hypothesis contradicted by it, and a hypothesis whose truth is undetermined.
These are classification and yes/no tasks, not learned continuous ratings.

## The test that requires the question

Every selected human-labeled source produces three examples:

1. Choose its annotated category from the described alternatives.
2. Ask whether the text expresses or satisfies that category. The reference is yes.
3. Ask about a different category. The reference is no.

The two binary examples have **identical state and answer options**, but
different questions and opposite targets. All versions of a source stay in the
same partition. Every individual binary question has equally many yes and no
labels because categories are sampled equally. A balanced schedule pairs each
reference category with **every other category equally often**, within every
partition. Binary negatives therefore cover all alternative classes.

The final test contains **360 opposite-answer pairs**. We report how often a
model gets **both** answers right. A deterministic model that ignores the
question scores zero on this paired criterion by construction. Independently
flipping a coin for each answer would average 25%.

A model looking only at the question, ignoring the text, achieves 50% individual
binary accuracy on these balanced samples. Paired accuracy alone is not a full
semantic-understanding test; ordinary per-task accuracy and the frozen-model
comparison must also be considered.

## Sources and partitions

- **ToolACE:** the same pinned public synthetic source as v0.1.0. The complete source yields 586 eligible first-call examples under our supported-format and length rules. Connected groups sharing requests or normalized tool schemas stay together. Groups touching an old test/development partition are retired, and groups merging old boundaries are excluded. Existing training and validation groups keep their roles. Entirely new groups receive a fixed hash assignment. All 64 new tool test rows were absent from the earlier experiment, and their specified schema/request overlaps are excluded from training.
- **Stanford Natural Language Inference:** [human-written, human-labeled sentence pairs](https://nlp.stanford.edu/projects/snli/), under Creative Commons Attribution-ShareAlike 4.0. We use equal numbers of the three classes, at most one selected pair per normalized premise, and preserve official train/validation/test boundaries.
- **GoEmotions:** [Google's human-annotated emotion data](https://huggingface.co/datasets/google-research-datasets/go_emotions), declared Apache-2.0. We select single-label examples from six categories, with equal class counts, and preserve official partitions.

All official human validation/test source groups are excluded from our training
sample, including held-out rows we did not select. Selected exact duplicates and
same-family states with word-trigram Jaccard similarity at least 0.8 are excluded.
All derived questions share a group. Fixed yes/no or category descriptions may
appear across human-data partitions: these are known-label tasks, so sharing the
answer vocabulary is intentional. Tool-schema isolation applies to ToolACE.

[Source revisions and checksums](../data/multitask/sources.json) pin each download.
The derived split files were independently rebuilt byte for byte. Original
source labels can still be wrong; converting a single emotion label into a
negative answer for another emotion adds an assumption. This is public annotation
agreement, not an independent human audit of our generated questions.

## Training protocol

All learned comparisons start from the **same released v0.1.0 weights**. The
frozen condition updates only the 768 scoring weights. The full condition updates
all 141,305,088 parameters. Frozen features are cached once; every full-model
update recomputes its encoder representations.

- Three training seeds: **7, 17, 29**. They change shuffling, option ordering, and full-model dropout. The data split remains fixed.
- Six passes over the training questions; batches of 32 questions. Nearby lengths are grouped to limit padding.
- AdamW; encoder learning rate 0.00002, scoring-layer learning rate 0.001, weight decay 0.01, gradient norm clipped at 1.
- Each task receives equal expected weight in the loss, using inverse task-frequency weights.
- Select the checkpoint with the lowest mean validation log loss across the five tasks, including the initial checkpoint.
- Select a primary full-model run by validation loss only. Report every seed, rather than picking one by test accuracy.
- Keep all final test predictions closed until every frozen and full training run and checkpoint selection is complete.

These are two fixed training recipes under the same data and pass budget.
They have not been separately tuned to establish the best possible performance
of either approach. No probability calibration or reinforcement learning is used.

Before the final run, a data audit found that an initial pairing rule always used
the same negative category for each positive category. We replaced it with the
balanced all-alternatives schedule above. The exploratory training run was
stopped before any final test predictions. The source states and sample counts
stay the same; the negative questions change. Training-batch profiling, using
training examples only, led to the larger batch and six-pass budget. All
checkpoints in the reported comparison use this corrected recipe.

## Three final evaluations

For the original model and each selected trained checkpoint, evaluate:

1. **Canonical questions:** the same question templates used during training, applied to new source examples.
2. **Unseen wording:** fixed paraphrases declared in the data builder before training. They express the same tasks, not new task families.
3. **Question removed:** replace every question with the same generic instruction. Identical inputs are evaluated once and their scores reused, preventing tiny numerical differences from making opposite-answer pairs appear correct.

The main measurements are reference agreement by task, the mean of the five task
accuracies, probability loss, and both-answers-correct accuracy on the 360 pairs.
A higher mean does not imply that every individual task improved.

## Reproduce

From a clone of the repository with Python 3.14 and an activated environment:

```bash
python -m pip install -r requirements-multitask.txt
python download_checkpoint.py
python multitask_data.py
python multitask_train.py \
  --init-run output/pretrained/first-instinct-v0.1.0
```

Skip the download command if that verified checkpoint already exists. On Linux,
install the processor-only PyTorch wheel first as described in the main README.
Training automatically uses Apple's Metal Performance Shaders backend when
available, or the central processor otherwise. Training on other machines can
produce numerically different results.

The training preset targets the measured M5 Max with 128 GB of unified memory.
The largest-batch preflight fit on that machine. Smaller-memory systems can use
`--batch-size 8` or less; this changes the optimization trajectory and runtime.

Every run records a protocol, the exact source snapshot, update traces, task
weights, validation histories, selected model hashes, and complete final test
predictions. The original v0.1.0 checkpoint and report remain available.

## What would remain unproven

The experiment can establish behavior on these three known task families and
sensitivity to the question on unseen source examples. It cannot establish
transfer to arbitrary new decision problems, robust interpretation of arbitrary
policies, trustworthy probabilities, or production reliability. Public benchmark
examples may have appeared in the base encoder's pretraining; that was not measured.
