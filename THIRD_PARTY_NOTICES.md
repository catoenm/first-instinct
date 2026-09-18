# Third-party materials

Original First Instinct code is released under the [MIT license](LICENSE).

## Pretrained encoder and tokenizer

The released model is a fine-tuned derivative of
[`microsoft/deberta-v3-small`](https://huggingface.co/microsoft/deberta-v3-small),
revision `a36c739020e01763fe789b4b85e2df55d6180012`. Its model card declares the MIT
license. Microsoft's copyright and license are preserved in
[licenses/DeBERTa-MIT.txt](licenses/DeBERTa-MIT.txt) and in the checkpoint release.
First Instinct adds a learned scoring layer and fine-tunes the encoder on tool
selection examples. It does not train a language model from scratch.

## ToolACE data

Source: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), by the
ToolACE authors. [Paper and author list](https://arxiv.org/abs/2409.00920).
The source dataset declares Apache-2.0; that license is included in
[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

The repository includes a 100-row dataset-viewer response for an introductory
exercise. The larger training source is downloaded from pinned revision
`6bda777c88d21e5a204703c1ee45597a8fa4f734` and checked against its recorded checksum.
Derived data changes are documented in [data/decisions/README.md](data/decisions/README.md):
filtering, conversion to described answer options, question wording, grouping,
and partition assignment. Recorded reference labels, example inputs, and option
names in the experiment artifacts come from this public source. Their presence
does not imply that the authors of this project independently verified them.

The release contains a model trained on this derived data. It includes these
notices and both upstream license texts. Dependencies retain their own licenses;
they are installed separately, not bundled in the checkpoint.

## Additional multi-task training data

The [Stanford Natural Language Inference corpus](https://nlp.stanford.edu/projects/snli/)
was created by Samuel R. Bowman, Gabor Angeli, Christopher Potts, and
Christopher D. Manning (2015), and incorporates caption material from earlier
datasets acknowledged on the corpus page. It is licensed under Creative Commons
Attribution-ShareAlike 4.0. The license is preserved in
[licenses/CC-BY-SA-4.0.txt](licenses/CC-BY-SA-4.0.txt). Downloaded source material
and our adapted question data retain those terms; they are not relicensed as MIT.

[GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
is Google's human-annotated emotion dataset. Its source card declares Apache-2.0.
The dataset and research are credited to the GoEmotions authors; see the
[original project](https://github.com/google-research/google-research/tree/master/goemotions)
for attribution and annotation details.

Exact versions and checksums are listed in
[data/multitask/sources.json](data/multitask/sources.json). Changes to the data
include sampling, filtering, grouped partition isolation, and adaptation into
categorical and yes/no questions. These modifications and assumptions are
documented in [the multi-task report](docs/multitask-experiment.md).

## Larger decision-model experiments

The optional `scale_lab` pipeline uses [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B)
and [Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) checkpoints published
by the Qwen team under Apache-2.0. Exact revisions are pinned in
`scale_lab/common.py`. These are separate from the released DeBERTa model.

[Glaive function calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
is synthetic data published by Glaive AI under Apache-2.0. Revision
`e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` and the original file checksum are
recorded in `scale_lab/glaive.py`. Our conversion extracts the first visible user
request and the first assistant action from single-function conversations,
groups connected function names and repeated requests, and derives described
options for calling a tool, asking a follow-up question, or responding.
Question-mark detection supplies the follow-up label; it is not a human audit.
Synthetic reference actions are not independently verified optimal actions.

All source licenses continue to apply to adapted data. Original authored
execution tasks are project code under MIT. Data manifests distinguish the two.
# Software inspection corpus

The `inspection_lab` experiment extracts and mutates functions from
[TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) at commit
`a381578994d545e44f26afabbd2303746a2dc358`. The upstream project is MIT licensed;
its notice is retained in [licenses/TheAlgorithms-Python-MIT.txt](licenses/TheAlgorithms-Python-MIT.txt).
Generated corpus archives contain derived source code, test excerpts and the
upstream notice. They are separate from our own experiment code. Source paths,
the pinned revision, input origins and execution receipts are preserved.

The local execution image is the official Python image pinned by manifest digest
in `inspection_lab/sandbox.py`. It is downloaded from Docker Hub and is not
redistributed in the repository or experiment archive.

## General instruction-conditioned decisions

The `general_lab` experiment uses selected tasks from
[Super-NaturalInstructions](https://github.com/allenai/natural-instructions),
revision `55a365637381ce7f3748fa2eac7aef1a113bbb82`, by Yizhong Wang,
Swaroop Mishra, and the contributors credited in the
[paper](https://arxiv.org/abs/2204.07705) and task metadata. Instructions and
metadata are Apache-2.0; individual task instances retain their original
licenses. They are not relicensed as project-authored MIT data.

The reproducible curation audit records each included task's contributors,
source URLs, pinned task URL, checksum, and instance-license declaration.
It accompanies the prepared data and must accompany derived artifacts.
Selected instance licenses in this run are MIT, Apache-2.0, CC-BY-4.0,
BSD-2-Clause, and CC0. See [the data documentation](docs/general-data-sources.md)
for transformations, source ancestry, exclusions, and attribution details.
Original sources remain authoritative for their copyright notices and terms.
The repository provides download/conversion code; it does not include the full
upstream instance corpus in Git.

The nine-billion-parameter foundation is
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, published by the Qwen team
under Apache-2.0. Generated executable scenarios, environment code, and
original question templates are project-authored materials under MIT.
