# A decision model in 144 seconds of local training

We fine-tuned a pretrained text encoder and a shared scoring layer to select the
first tool for a request. All **141,305,088 parameters** were trainable. On 34
held-out cases, it matched 29 synthetic reference labels, compared with 18 for a
frozen encoder with a trained scorer and 24 for word overlap.

The complete recorded evidence is in [results/mac-v1](../results/mac-v1).

## Experimental setup

| Setting | Value |
| :--- | :--- |
| Starting model | Microsoft DeBERTa-v3-small, pinned revision |
| Hardware | Apple M5 Max, 40 graphics cores, 128 GB unified memory |
| Runtime | Python 3.14.6; PyTorch 2.14.0; Metal Performance Shaders backend |
| Training / validation / test | 166 / 44 / 34 examples |
| Additional development partition | 22 examples, unused by the corrected comparison |
| Training | Three passes; 498 updates; one question per update |
| Optimizer | AdamW; weight decay 0.01; gradient norm clipped at 1 |
| Learning rates | Encoder 0.00002; scorer 0.001 |
| Seed | 7 |
| Checkpoint selection | Lowest validation log loss, including the initial model |
| Full training elapsed | 144.1 seconds, including validation and checkpoint writes |
| Frozen baseline elapsed | 10.2 seconds, including feature extraction |

The encoder and scorer use 32-bit floating point weights. Both modes use the
same seeded example and option order and three-pass budget. The frozen baseline
caches deterministic encoder representations; full fine-tuning enables encoder
dropout during training. These are two training recipes under a shared small
budget, not separately optimized baselines. Setup and downloads are outside the
reported timings. Memory use and steady-state inference latency were not measured.

## Results

| Method | Reference matches | Match rate | Log loss ↓ | Brier score ↓ |
| :--- | ---: | ---: | ---: | ---: |
| Uniform random choice, expected | — | 37.1% | — | — |
| Word overlap | 24 / 34 | 70.6% | — | — |
| Frozen encoder + trained scorer | 18 / 34 | 52.9% | 0.9350 | 0.5709 |
| Fine-tuned encoder + scorer | 29 / 34 | 85.3% | 0.2494 | 0.1767 |

Word overlap is Jaccard similarity between the request's word set and each full
option description's word set. The random baseline averages `1 / option_count`
across cases. The Brier score is the mean sum of squared probability errors
across options. Neither a lower Brier score nor a lower log loss establishes
calibration on its own.

The full model corrected 13 frozen-model errors and introduced two. Each case
changes the match rate by about 2.9 percentage points, so these percentages should
not be read as precise population estimates. No tools were executed to determine
correctness. The outcome is agreement with a single synthetic reference call.

Full-model validation history explains the selected checkpoint:

| Completed passes | Reference matches | Log loss |
| ---: | ---: | ---: |
| 1 | 31 / 44 | 0.8150 |
| **2, selected** | **37 / 44** | **0.7856** |
| 3 | 37 / 44 | 0.8851 |

The best checkpoint came from pass two. Both training runs completed and their
checkpoints were selected before either selected model was evaluated on the
corrected test set.

## The question-wording mistake

The first exploratory comparison asked:

> Which available tool best fulfills the user's request?

The source label instead names the **first tool call**. A request might ask to
check travel restrictions and then create a packing list. Either tool relates to
the overall request, but the recorded sequence starts with restrictions.

We inspected errors on the original 22-case test and corrected the question to:

> Which available tool should be called first to begin fulfilling the user's request?

Because that inspection informed the change, the old test is now development
data. The previously unused 35-case calibration partition became the new test.
One training case and one new test case exceeded the 512-token limit with the
longer question and were explicitly excluded. No other examples changed their
request, options, label, or established group boundary.

The corrected partitions contain 166 training, 44 validation, 22 development,
and 34 test cases. For compatibility, development is still named
`calibration.jsonl`. It is **not a clean calibration set**, and no probability
calibration was fitted. The run's original `protocol.json` calls this a reserved
split; that means it was unused by this comparison, not that it was never inspected.

This history matters: we cannot treat the original exploratory test as untouched
validation of the corrected task. The results in this report use only the new
34-case test. Further iteration should reserve another new final test.

## Data boundaries

The pinned ToolACE source contains 11,300 rows. We sampled 5,000 outside the
first 100 inspection rows. The strict converter initially retained 268; the
wording revision left 266. Most exclusions did not match the supported
first-turn single-call conversation format or offered fewer than two tools.
The filter counts are in the [source manifest](../results/mac-v1/dataset-v1/manifest.json).

Before partitioning, connected groups join examples sharing an exactly
normalized request or option description and parameter schema. All candidates
count, including distractors. Requests with word-trigram Jaccard similarity of
at least 0.8 are also joined. Independent boundary checks enforce those rules
across every pair of partitions.

This prevents those specific overlaps. It does not guarantee separation of
semantic tool families, paraphrased descriptions, or the encoder's original
pretraining data. Some requests also permit several valid first actions; a
single synthetic reference label cannot capture that ambiguity.

## Verification and reproduction

These commands rebuild every split in both revisions byte for byte:

```bash
python decision_dataset.py \
  --task legacy-whole-request --output output/decision_dataset_v1
python decision_dataset.py \
  --revise-from output/decision_dataset_v1 --output output/decision_dataset_v2
python finetune_decisions.py --data output/decision_dataset_v2
```

The first build downloads the pinned 37 MB ToolACE source; training downloads
the pinned base encoder if it is not cached. Existing output directories are
never overwritten. A direct build with the corrected default question is useful
for a new experiment, but differs from this historical revision's membership.

Original source snapshots and hashes are
preserved alongside the results. Dataset manifests change when current source
code changes; split-file hashes are the stable data-identity check.

Each of the 498 recorded updates per model uses a training row. Encoder
gradients were nonzero during full training, and a checked encoder weight matrix
changed. Saved artifact hashes matched their manifests. Reloading the selected
full checkpoint changed validation probabilities by at most approximately
0.0000024; the frozen scorer reloaded exactly.

Training need not be bit-for-bit identical on another processor or dependency
version. The repository includes the exact dependency lock used for this run.
Tests check the training mechanics and split rules without downloading models.

## What this establishes

The pipeline can convert public tool-use data into described choices, keep
specified overlaps out of the test set, change a pretrained encoder through
supervised learning, and reload a working decision model. Full fine-tuning
improved reference agreement over the two baselines in this small experiment.

It does **not** establish general instruction following, strong performance on
unseen semantic task families, calibrated uncertainty, tool argument generation,
reliable abstention, or a latency/cost advantage over a well-engineered alternative.
Those require new experiments. The input contract resembles decision-oriented
interfaces, but the architecture is a conventional encoder and shared scorer.
