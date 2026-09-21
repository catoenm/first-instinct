# Broad tool behavior admitted for the release dataset

The three previously cached TOUCAN shards now yield **22,434 qualified first-tool
questions**, including **17,786 training questions across 283 server groups**.
Every exported prompt, target and complete token sequence was checked against
its pinned source row. No model has consumed these new examples yet.

| Usage | Questions | Distinct normalized requests | Server groups | Input tokens | Fits 1,536 tokens |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Training | 17,786 | 17,072 | 283 | 19,729,286 | 13,504 |
| Development | 2,217 | 2,071 | 27 | 2,913,684 | 1,379 |
| Reserved transfer | 2,431 | 2,330 | 38 | 3,138,141 | 1,783 |

A question is one rendered training/evaluation input. Repeated requests with
different actual tool declarations can produce distinct inputs in the same
partition. The request count normalizes whitespace; it does not establish
independent real-world tasks or detect semantic paraphrases. A server group is
a source attribution boundary, not a claim of a unique reasoning mechanism.

## What the labels mean

These are **observed first-action imitation** labels from synthetic trajectories
in [TOUCAN](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M), pinned to revision
`0df3cf37f2abefb380370cfb02eabea2a35ae782`. Inputs contain the preceding user request
and actual offered function names, descriptions and parameter schemas. They do
not contain the teacher's thoughts, demonstrated arguments, future tool results
or judge scores.

An accepted call satisfies structural argument/schema checks. That does not make
it an optimal first action or prove task success. None of the recorded tools was
executed by this conversion. These examples teach broader tool selection; the
separate execution-verified pools supply consequence forecasts. No success
probability was manufactured from teacher confidence or a dataset judge score.

## Filtering and separation

The source contains 35,227 trajectories and 33,961 normalized requests. Fixed
server ownership was assigned before labels were inspected. Exact shared schemas
removed 42 lower-priority server groups in their entirety, affecting 3,265 rows;
another 1,685 rows composed servers across partitions and were rejected.

The remaining exclusions include 4,205 invalid menu sizes, 717 undeclared first
calls, 307 argument/schema violations, 213 unsupported schema references, 311
unsupported schema keywords, 190 conflicting teacher choices and 1,649 inputs
over 4,096 tokens. Smaller structural exclusions are itemized in the saved
[aggregate report](../results/release-tool-data-v1/summary.json). Conflicts were
rejected rather than majority-voted into certain labels; long inputs were
rejected rather than truncated.

The earlier-corpus overlap index scanned 558,390 visible row presentations and
484,891 prepared-token row presentations. Repeated copies reduce to 346,596
request keys, 16,493 tool-schema keys and 403,804 token-input keys. These are
overlap-index counts, not additional training data. No admitted request/schema
or token input overlaps that exact index. Protected application records stayed
local. This is not a semantic contamination audit or evidence about the
foundation model's pretraining exposure.

The saved-export audit accounts for all source trajectories, checks every
admitted source witness, reconstructs prompts independently of the converter,
retokenizes them and checks disjoint ownership. Admission is bound to the
candidate manifest digest. The loader checks data hashes and rejects a changed
row or a requested usage role inconsistent with its manifest.

All 15 converter/audit fixture tests pass. Six additional CPU tensor checks pass
for the release loss: categorical distributions retain uncertainty, acceptable
sets retain their different meaning, padding contributes no gradient, and
malformed targets fail. These are local objective checks, not a model training
run or a reinforcement-learning result.

## Scope and next stage

The source is three contiguous cached shards, not a representative sample of the
full advertised dataset. Exact separation survived with useful coverage, so no
random row split or weakened ownership rule was needed. All new model calls,
optimizer steps, executed branches and training presentations remain **zero**.

This completes the first admission stage in the
[release plan](release-candidate-v1-plan.md). Retail paired-question admission,
the combined hard/distribution-target pack, fixed evaluation cohorts and the
bounded pilot remain separate gates. The original 9B checkpoint and live demo
remain unchanged. The release target of 4,096 tokens is not yet a qualified
serving capability.

Code: [converter](../release_lab/toucan.py),
[saved-export audit and usage guard](../release_lab/toucan_audit.py),
[mixed objectives](../release_lab/objectives.py).
The [prospective protocol](release-tool-data-v1-protocol.md) and
[admission receipt](../results/release-tool-data-v1/admission.json) preserve scope
and lineage. Candidate rows, source witnesses and the full freeze remain private.
