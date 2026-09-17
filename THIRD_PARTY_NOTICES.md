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
