# From text to one choice

Imagine a request and three available tools: weather, calendar, and calculator.
The model must pick one. The options can change from request to request.

## 1. Make one text pair per option

Each pair contains the same **question and request**, followed by one option's
**description**. The tokenizer turns text pieces into integer identifiers.
Those identifiers select learned token vectors; the integers are not themselves
probabilities or measurements of meaning.

The model accepts up to 512 tokens per pair, including special tokens. Long
examples are rejected, not silently cut short. The request is repeated for every
option, so more options mean more computation.

## 2. Read each pair with the same encoder

A pretrained DeBERTa encoder lets the words in the request and option description
influence each other's representations. This is often called a cross-encoder:
both texts are processed together. Each token ends up represented by 768 numbers.
We average the real token vectors, excluding padding, then normalize the result.

The encoder already learned useful language representations during Microsoft's
pretraining. Our small training set adapts those representations to this task.

## 3. Turn each representation into a score

A shared layer with 768 weights takes a weighted sum of each option's vector:

```text
score = weight[0] × vector[0] + ... + weight[767] × vector[767]
```

The **same weights** score weather, calendar, calculator, and any newly described
option. Identifiers such as `weather` stay outside the encoder input. They let
the caller connect each result back to the supplied option.

Softmax converts the scores into positive probabilities that sum to one across
available options. The highest probability determines the returned choice. The
model must select from the provided list; this version has no learned “none of
these” answer.

## 4. Learn from the reference answer

Training supplies the reference option separately. The loss is the negative
logarithm of that option's probability: a low reference probability incurs a
large penalty. Backpropagation computes how each weight affects that loss, and
an optimizer updates the weights to reduce it.

The frozen experiment changes only the 768 scoring weights. Full fine-tuning
also changes the encoder's 141 million weights. That is the main experimental
comparison. The starting encoder already knows language; the experiment teaches
it a particular selection task.

The reference tells us which answer to reinforce. It does not prove that answer
is uniquely correct. This is why data definition and label inspection matter as
much as the training loop.

## Why this is still a classifier

It is a classifier whose possible answers are supplied as descriptions at
inference time. A traditional fixed-label classifier might have permanent output
positions for “weather,” “calendar,” and “calculator.” Here, one scoring function
reads each supplied description. This supports changing candidate lists without
creating a new output layer for every tool.

That interface does not make arbitrary questions work automatically. All our
training examples ask one kind of question: which tool should be called first?
Learning more question types requires appropriate data and evaluation.

No text is generated during selection, and no reinforcement learning is used.
Returning probabilities also does not make them calibrated. Calibration asks
whether predictions assigned a given probability succeed at the corresponding
rate on independent examples.

## Learning path

Run these from the repository root after installing the dependencies:

1. `python pipeline.py` — inspect normalization, exact hashes, duplicate decisions, and text similarity on five tiny documents.
2. `python lsh.py` — explore locality-sensitive hashing as a way to find similar documents without comparing every pair.
3. `python decision_data.py` — turn the bundled public inspection sample into explicit inputs and separate labels.
4. `python train_decisions.py` — memorize eight examples while recording the first weight update and every loss. This has no held-out evaluation.
5. Follow the [reproduction commands](../README.md#reproduce-the-experiment) — build isolated partitions and compare frozen versus fully fine-tuned encoders.
6. Inspect [the recorded errors](../results/mac-v1/full/test_results.json) and [experimental history](experiment.md) before deciding what data to collect next.

The useful milestone is understanding what each number measures and what it
cannot tell you. A small working model makes that easier to inspect.
