# Prospective typed-question robustness audit

This is a supplementary diagnostic for a model that answers supplied questions
using supplied choices. It is **not part of the current frozen training run or
checkpoint selection**, and no trained model has been evaluated on it here.
Freeze its content and evaluation plan before using it to make model comparisons.

The implementation is [`general_lab/robustness.py`](../general_lab/robustness.py).
It creates 48 distinct root configurations and 456 public questions. These are
authored, finite examples; hundreds of variants do not establish broad reasoning
ability or calibration on arbitrary user questions. They also do not identify
Jev's architecture or training procedure.
Before any model predictions, review found unexercised routing, credit and
safety-alarm clauses. The final corpus includes evidence pairs where those
clauses alone change the answer; tests verify that coverage.

## What the questions test

There are twelve roots in each family:

| Family | Supplied answer type | Executable target |
|---|---|---|
| Compositional routing | Three described choices | Ordered Boolean and numeric rules select Review, Express, or Standard |
| Partial-knowledge entailment | Binary proposition | Enumerate all permitted unknown-fact completions and check whether eligibility holds in every one |
| Urgency | Three ordered levels | Apply explicit threshold and priority rules to Routine, Elevated, and Urgent |
| Finite random experiment | Binary forecast | Enumerate every ball after a fully specified weighted bag selection |

Unknown facts in the entailment family have **no assigned probabilities**. “No”
means the claim that eligibility is guaranteed is false; it does not claim that
eligibility itself is false. A supported alternative can prove eligibility even
when another fact is unknown.

The probability family states bag weights, all ball counts, uniform sampling
within each chosen bag, and the absence of other observations. Its target is the
exact yes/no distribution calculated using rational arithmetic. Labels are not
teacher-model confidence or invented certainty about missing information.

Every root contains an evidence-change pair that requires a different correct
deterministic answer, or a different most likely event for a probabilistic
forecast. Both sides also receive:

- Equivalent question wording.
- Opaque option identifiers, with a separate semantic mapping.
- Explicitly irrelevant display metadata.
- Reversed choice order for ordinary and binary choices.

Ordered levels retain their original order in every variant. Their identifiers
can change without changing their descriptions or positions. Evidence-change
pairs are not treated as invariances: a model should respond to those changes.

## Serialization contracts are separate from learned skills

The existing prompt serializer omits option identifiers and emits positional
labels plus descriptions. Renaming identifiers therefore produces byte-identical
model prompts. Passing this check demonstrates correct serialization and output
mapping, not learned semantic robustness. The audit verifies this for 96 paired
world states.

It also calls `general_lab.interface.requests` with a focal question alone and
beside an unrelated question. The resulting focal input and serialized prompt
must be identical. This checks 96 question-independence contracts without
constructing a model. It does not establish identical floating-point predictions
under different inference batching, or shared-state computation efficiency.

## Measurement and accounting

The runner accepts a callback that receives only a list of public
`state`, `question`, and `options` dictionaries. Specifications, exact targets,
root identifiers, and semantic mapping receipts remain outside this interface.
Returned distributions must use exactly the supplied option identifiers, contain
finite numeric probabilities in the unit interval, and sum to one within
0.000001. Missing choices, invented labels, booleans, and malformed batches fail
the audit. Public choice order breaks probability ties.

Deterministic questions report accuracy, logarithmic loss, and the full
multiclass Brier score. Probability questions separately report modal accuracy,
expected accuracy of the chosen event, exact expected logarithmic loss, exact
expected Brier score, and mean squared error to the known distribution. A perfect
probabilistic forecast generally has nonzero expected logarithmic and Brier loss
because the event itself is random. Its distribution error is zero.
Log scores use a recorded probability floor of 0.000000000001, so they remain
finite even if a model assigns zero to a possible event. An exact expected log
score here is the exact finite-distribution expectation of that **clipped**
score; it is not the unbounded log loss at zero probability.

Invariance comparisons report total variation distance, the largest probability
change, and answer flips after mapping outputs back to their semantic choices.
Evidence-change pairs report answer flips and whether both required modal
answers were correct. A model that always picks the first option can look stable
under paraphrases while failing both order invariance and evidence sensitivity;
the tests explicitly check this failure mode.

Results include every root and each family. Aggregate metrics average within
roots and then across roots, so the smaller number of ordinal variants does not
silently reweight families. Root count is 48; question count is 456. The roots
share authored templates, and paired worlds and transformations are correlated.
No independence-based confidence interval is claimed.

## Running the checks

Generate a new, non-overwritten fixture file:

```sh
python -m general_lab.robustness --output output/typed-question-robustness-v1.json
python -m unittest test_general_robustness -v
```

The file includes a deterministic content checksum. Before actual inference,
record that checksum, code versions, model/adapter identity, prompt serializer,
batching settings, and the comparison plan. Reserve it as a supplementary
evaluation rather than adding its targets to training.

Use the callback runner from Python:

```python
from general_lab.robustness import make_corpus, interface_contract, run

corpus = make_corpus()
contracts = interface_contract(corpus)  # No model inference.
# predict_many(public_inputs) must return one {option_id: probability} per input.
report = run(corpus, predict_many, batch_size=16)
```

The unit tests use exact and deliberately broken fixture callbacks. Their
perfect or failing scores validate audit mechanics; they are not results from
the trained First Instinct model.
