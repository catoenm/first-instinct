# Database questions admitted to a separate staging index

All 138 previously qualified canonical questions passed exact token and ownership
checks against 842,299 stored presentations in 18 frozen training, development,
and evaluation files. There were no candidate token collisions or existing group
identifier collisions. Reserved evaluation was used only for token fingerprints;
no reserved model scores were opened or generated.

The staged index preserves 120 outcome distributions and 18 acceptable-choice
sets, their original prompts, and their executed-branch provenance. The outcome
contracts remain separate: 36 immediate goal-status questions, 36 immediate return
codes, and 48 outcomes under a displayed procedure. Reversed menus were checked
but were not added as extra training examples.

This is one training ownership group, four existing world/goal tasks, and one
mechanism. Exact input checks do not establish semantic novelty across all prior
language data. No existing training pack was changed, and none of these staged
questions was consumed by a model.

The streaming check took about 25 seconds under the local guard, with approximately
102 MiB peak process-group memory and no additional swap. Both scanner tests
passed. The [new loss consumer](decision-learning-v2-results.md) subsequently
qualified the distinct forecast contracts on CPU; that does not start a run or
admit the later live-history candidates automatically.

[Protocol](revisioned-admission-v1-protocol.md) ·
[Aggregate receipt](../results/revisioned-admission-v1/summary.json)
