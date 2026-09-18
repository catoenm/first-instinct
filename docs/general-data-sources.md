# Public instruction data for the general decision model

The public portion supplies varied natural language instructions and human/source annotations. Executable synthetic worlds supply a separate kind of evidence: outcomes we can recompute. The public rows must not be described as verified environment rewards or calibrated probabilities.

We use [Super-NaturalInstructions](https://github.com/allenai/natural-instructions) at revision `55a365637381ce7f3748fa2eac7aef1a113bbb82`. Its [paper](https://arxiv.org/abs/2204.07705) describes a broad collection of tasks with natural language instructions. Our pinned snapshot contains 1,613 task files; only a conservative subset enters this experiment.

## Reproduce the public portion

```sh
.venv/bin/python -m general_lab.sources \
  --output output/general-data-v1/sources/natural-instructions
.venv/bin/python -m general_lab.curate \
  --source output/general-data-v1/sources/natural-instructions \
  --output output/general-public-v1 \
  --seed 41 --max-per-task 4000
.venv/bin/python -m unittest test_general_curate
```

The downloader verifies each file against its pinned Git blob checksum. Curation verifies selected task files against the inventory's SHA-256 hashes and refuses an incomplete inventory. The audit records task checksums, revision, original source names and URLs, contributors, categories, per-instance license metadata, exclusions, split hashes, source aliases, ancestry links, and row counts.

The default output is:

| Partition | Rows | Instruction tasks | Source components | Document groups |
| --- | ---: | ---: | ---: | ---: |
| Training | 170,935 | 123 | 40 | 156,906 |
| Validation | 9,962 | 123 | 40 | 9,192 |
| Test | 13,077 | 46 | 12 | 12,448 |
| Challenge | 2,048 | 4 | 2 | 2,048 |

There are 173 included tasks altogether. Training covers 20 categories, including scientific and mathematical question answering, inference from rules, negotiation strategies, intent, dialogue state, sentiment, answer verification, and table/query decisions. Rows and task names are not independent domains: several tasks can come from one underlying dataset.

The complete audit is `output/general-public-v1/natural-instructions-audit.json`; this run's SHA-256 is `88833defe9b97d6cd18e4c37264a0687b314780c90b75829a1de9179eedcf823`.

## Selection and label handling

Each example supplies the upstream input as its state and the upstream definition as its question. The offered choices come from that task's observed output vocabulary. Targets retain the annotated answer or acceptable-answer set. Source identifiers, labels, audit fields, and targets are never copied into the model prompt. Option ordering is randomized in the later preparation stage.

Tasks need English input/instruction metadata and 2–36 distinct, short output labels. Classification, detection, verification, and explicitly multiple-choice tasks are eligible. Small observed vocabularies alone do not justify turning a generative task into a classification task: sampled question generation and free-form word analogy are excluded. Capitalization, whitespace, and final-period variants of the same label are merged; meanings are not rewritten. A letter choice such as `A` is described as the corresponding answer in the supplied state, so it is distinguishable from the model interface's own selection letters.

We excluded all 738 officially excluded tasks, 528 additional tasks with instance licenses outside the allowlist, 150 tasks without bounded output vocabularies, and 23 generative or insufficiently specified cases. One task whose English metadata conflicts with its Malayalam scope is explicitly excluded. These counts follow the ordered selection rules, so they are not independent totals by defect. We also removed empty inputs, duplicate task/state rows, and conflicting duplicate annotations.

Upstream annotations remain fallible. A binary sentiment or commonsense label is not an executed fact. Public-data evaluation measures agreement with those annotations.

## Source and document separation

The source graph is built from **all** task metadata before filtering, including generation tasks and excluded tasks. Every source component touching an official test task is reserved for test. Known aliases are normalized, including e-SNLI/SNLI and Winograd/WSC variants; semicolon-separated source fields are split.

We also record known dataset ancestry. In particular, [Defeasible-NLI's authors](https://github.com/rudinger/defeasible-nli) describe its ATOMIC, SNLI, and Social Chemistry source datasets. Natural Instructions gives all three portions the same `defeasible_nli_atomic` source string. Accounting for this ancestry moved ATOMIC's 21 instruction variants out of training. Counting those variants as a novel source would have overstated the holdout.

The challenge additionally reserves the entire **Word Relation Classification** and **Coherence Classification** categories and their connected sources. Training and validation may share sources, but normalized state/document groups are disjoint. Repeated explicitly marked contexts, paragraphs, passages, reviews, and premises are grouped across question variants. If a document connects a development example to a reserved source, the development example is dropped. Quotas are applied only after grouping and partition assignment.

Training has a cap of 4,000 rows per task and 40,000 per source component. Validation is capped at 256 per task and 2,048 per source; test/challenge at 512 per task and 4,096 per source. Whole document groups are retained or dropped at each cap. This prevents multiplying instructions over one source from overwhelming the mix.

These checks detect normalized exact text and explicitly marked document reuse. They do not prove the absence of paraphrases, latent source relationships, or semantic duplicates. The source graph records known metadata and ancestry, not a complete genealogy of every public dataset.

## Licenses and benchmark claims

The repository's metadata license does not automatically license every dataset's instances. We filter using each task's `Instance License`. The current selected tasks declare MIT, Apache 2.0, CC BY 4.0, BSD 2-Clause, or CC0; the code contains the full explicit allowlist. Unknown, unspecified-version CC BY, noncommercial, share-alike, and custom terms are excluded from this run. Preserve the original source attributions and task-level audit with derived data. This records upstream declarations rather than independently certifying every license claim.

The language foundation may have seen these public benchmarks during pretraining. Natural Instructions also does not preserve every original benchmark's train/test boundary in its exported task instances. Some well-known benchmark tasks, including its multiple-subject knowledge questions, appear in our **training** mixture. Do not present scores on those source benchmarks as uncontaminated generalization results.

Our defensible claims are narrower: adaptation across many instruction/option schemas, source-held-out performance within this explicit curation protocol, and separate results on newly generated executable environments. This dataset does not establish general intelligence, reproduce Jev's private data recipe, or demonstrate calibrated outcomes by itself.
