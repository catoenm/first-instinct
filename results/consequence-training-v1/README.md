# Consequence-learning pilot

The prospective [protocol](../../docs/consequence-training-v1-protocol.md) and
[freeze](freeze.json) were published before candidate inference. This experiment
updates the released Qwen3.5-9B language adapters using exact consequence targets
from executed database commands, mixed with retained general training examples.
It is supervised consequence learning; it is not a new reinforcement-learning run.

The data contains 2,592 training questions, 864 public-history validation questions,
2,512 general replay examples and 622 general retention questions. All reservation
questions concern one previously studied mechanism. New parameter combinations
are diagnostic, not evidence of transfer to independent mechanisms.

One H100 completed all 180 optimizer steps. The prospective validation and
retention rule selected step 120. Every one of the 496 internal language-adapter
tensors changed. Estimated compute cost was $1.78, excluding storage. The full
archive was verified locally, then the rental was stopped and deleted.

Immediate consequence modal-answer accuracy rose from 81.25% to 99.65%; categorical
probability error fell 90.9%. Executed decisions had mixed gains and regressions,
with substantial sensitivity to wording and answer order. Read the
[full results](../../docs/consequence-training-v1-results.md) before interpreting
the [computed report](report.json). This does not establish general calibration,
independent-mechanism transfer, or a reinforcement-learning benefit.

The [post-hoc overlap audit](overlap-audit.json) found that 216/864 validation
prompts exactly matched training prompts despite disjoint history identifiers.
The remaining 648 improved from 83.33% to 99.54% modal accuracy. That filtered
result does not repair the original checkpoint-selection split or establish
independent generalization. The prospective freeze and selected step are preserved.

The [reporter](../../puffer_lab/consequence_report.py) is separate from the frozen
training code. It recomputes probability metrics, checks prediction identities,
and replays every before/after decision trajectory in the qualified environment.
Its checks reject state corruption and malformed probabilities. All 802 before/after
action transitions were replayed. The [saved-tensor audit](tensor-audit.json)
compares actual starting and selected adapter files; the
[collection receipt](collection.json) records cost and verified deletion.

The `evidence` directory preserves 31 original files byte-for-byte, including
every candidate validation, training sample identity, probability prediction,
public decision prompt and resulting state. The
[research release](https://github.com/catoenm/first-instinct/releases/tag/consequence-learning-v1)
contains the complete original 47-file archive with selected/latest adapters,
tokenizer and prepared data, plus its required attribution companion and asset
hashes. Foundation weights are not included. The existing demo retains its
general supervised checkpoint.
