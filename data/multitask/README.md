# Multi-task source data

The [source manifest](sources.json) records exact upstream revisions, source-file
checksums, download URLs, and declared licenses. Raw downloads are kept in
`raw/` and excluded from version control. `multitask_data.py` verifies every file
before conversion.

The human-labeled sources are the Stanford Natural Language Inference corpus
and GoEmotions. The builder additionally reuses the pinned ToolACE snapshot from
the first experiment. Derived data modifications include class balancing,
selection of single-label emotion examples, grouped sampling, conversion to
categorical and paired binary questions, fixed alternative question wording,
and removal of overlong inputs and duplicate or near-duplicate source states.

Each source state contributes one categorical example and two binary examples
with different questions and opposite answers. The latter two keep their state
and options identical. All versions stay in one partition.

See [the experiment protocol](../../docs/multitask-experiment.md) for sample sizes,
partition isolation, label assumptions, and reproduction commands. The code is
MIT-licensed; downloaded and derived source data retain their upstream license
terms. Stanford Natural Language Inference material and adaptations are under
Creative Commons Attribution-ShareAlike 4.0; GoEmotions and ToolACE declare
Apache-2.0. See [third-party notices](../../THIRD_PARTY_NOTICES.md).
