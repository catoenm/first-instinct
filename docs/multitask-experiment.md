# Several decisions, one shared model

This experiment continues the published First Instinct v0.1.0 checkpoint on
three task families: first-tool selection, emotion recognition, and the
relationship between a premise and a hypothesis. The architecture stays the
same: a shared text encoder, pooled token representations, and one shared
scoring layer for described answer options.

The experiment asks whether this small model can use the question to decide
which judgment to make about the same text. It does not test arbitrary new task
families, general agent behavior, or equivalence to Jev.

## Measured results

Across three seeds, updating the full network reached **79.5% mean accuracy
across the five tasks**, compared with **57.5%** when only the scoring layer was
trained. The average improvement was **22.0 percentage points**. Both methods
continue the same released v0.1.0 model on the same data and six-pass budget.

| Model | Mean task accuracy | Both answers correct, 360 pairs |
| :--- | ---: | ---: |
| Original v0.1.0, no additional training | 50.3% | 3.3% |
| Fixed encoder, trained scorer; three-seed mean | 57.5% | 20.7% |
| **Full fine-tuning; three-seed mean** | **79.5%** | **61.8%** |

Full-model task accuracy ranged from **78.6% to 81.1%** across seeds. The 95%
source-resampling interval for the mean improvement over the fixed encoder is
**18.9 to 25.1 percentage points**. These intervals use 2,000 bootstrap resamples
of whole source examples, stratified by source family, with identical resamples
for all models. They describe this benchmark's sampled examples, not uncertainty
over arbitrary tasks or every possible training seed.

### Accuracy by task

The following are means over the same three seeds. The first-tool test contains
only 64 examples, so its high accuracy has a small denominator.

| Task | Test questions | Fixed encoder | Full fine-tuning |
| :--- | ---: | ---: | ---: |
| First tool to call | 64 | 92.2% | 94.3% |
| Sentence relationship, three choices | 180 | 32.2% | 70.0% |
| Sentence relationship, yes/no | 360 | 50.4% | 72.8% |
| Emotion, six choices | 180 | 54.8% | 76.5% |
| Emotion, yes/no | 360 | 58.1% | 84.1% |

Every task improved on average, although the sentence-relationship tasks remain
the weakest. This is agreement with source annotations, not verified execution
of tools or independent adjudication of every human label.

### Wording sensitivity

| Evaluation, full-model three-seed mean | Mean task accuracy | Both paired answers correct |
| :--- | ---: | ---: |
| Training question templates, new examples | 79.5% | 61.8% |
| Predeclared unseen question wording | 76.0% | 46.0% |
| Question replaced with a generic instruction | 68.6% | 0.0% |

Paraphrases reduce mean accuracy by **3.5 percentage points** and paired accuracy
by **15.7 points**. The model learned useful question-dependent behavior, but its
interpretation is still sensitive to wording. The removed-question condition's
zero paired score follows from identical inputs having opposite reference
answers; it is a structural check, not additional evidence of broad reasoning.
Categorical option descriptions still reveal which categorical task to perform,
which helps explain that condition's much higher ordinary accuracy.

### Every seed and checkpoint

| Condition | Seed | Selected pass | Mean task accuracy | Both paired answers correct | Run seconds |
| :--- | ---: | ---: | ---: | ---: | ---: |
| frozen | 7 | 6 | 57.3% | 80 / 360 | 0.8 |
| frozen | 17 | 6 | 57.4% | 70 / 360 | 0.8 |
| frozen | 29 | 6 | 57.9% | 74 / 360 | 0.9 |
| full | 7 | 5 | 78.6% | 224 / 360 | 580.1 |
| full | 17 | 5 | 81.1% | 230 / 360 | 624.4 |
| full | 29 | 4 | 78.9% | 213 / 360 | 632.1 |

Full training took about **10 minutes per seed** on the Apple M5 Max, including
validation and checkpoint writes. Frozen scoring-layer times exclude the
one-time shared encoder feature cache, so they are not end-to-end comparisons.
Downloads, initial shared setup, and final test evaluations are outside the
per-run timings. Every run completed six passes even when an earlier checkpoint
was selected.

The released model is **seed 17, pass 5**, selected by validation loss before
final testing. Its 81.1% test accuracy is listed for reproducibility; the headline
comparison uses the three-seed mean. For example, seed 29 performs best on tools,
while seed 17 performs best on sentence relationships. There is no single seed
that wins on every measure.

[Complete predictions, traces, manifests, and checks](../results/multitask-v1)
include all 21 model/wording evaluations. Every run visited each training question
exactly once per pass, with 672 updates and 21,402 question visits. Checkpoint
hashes matched; reloaded validation probabilities differed by less than 0.000006.
The source splits reproduced byte for byte. The 25 offline code checks pass on
macOS and Linux.

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
- Six passes over the training questions; effective batches of 32 questions. Full training processes eight-question microbatches and accumulates gradients before each update. Nearby lengths are grouped to limit padding.
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
checkpoints in the reported comparison use this corrected recipe. A subsequent
32-question execution preflight fit, but sustained training suffered a severe
memory-related slowdown and was stopped before final testing. Processing eight
questions at a time with gradient accumulation keeps the effective batch at 32
while reducing memory use. Gradients are divided by the whole update-batch size,
including a shorter final batch; a numerical test checks this accounting.

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
Smaller-memory systems can reduce `--microbatch-size`; `--batch-size` controls
the number of questions contributing to each optimizer update. Dropout and
floating point execution can still make runs differ when microbatch size changes.

Every run records a protocol, the exact source snapshot, update traces, task
weights, validation histories, selected model hashes, and complete final test
predictions. The original v0.1.0 checkpoint and report remain available.

## What would remain unproven

The experiment can establish behavior on these three known task families and
sensitivity to the question on unseen source examples. It cannot establish
transfer to arbitrary new decision problems, robust interpretation of arbitrary
policies, trustworthy probabilities, or production reliability. Public benchmark
examples may have appeared in the base encoder's pretraining; that was not measured.
