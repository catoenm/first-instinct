# Fixed-continuation forecast probe

Declare before any forecast predictions. This supplementary diagnostic follows
the already frozen reservation language baseline; it does not change that
baseline's cases, prompts, outcomes, or model. Both are development studies on
the same authored mechanism, not independent evidence of broad generalization.

Use the same resident supervised Qwen3.5-9B step-2742 checkpoint. Select all 36
initial-history action targets from the published reservation qualification,
plus all uncertain-success targets at noninitial histories. That adds three
targets, giving **39 distinct event questions**, of which eight have success
probabilities strictly between zero and one. These selection criteria depend
only on the existing qualified corpus, never on new model predictions.

Ask whether the task will succeed after a specified first action followed by
the exact named public continuation. State that continuation's ordered rules,
the transaction semantics, current public history, prior, and turn budget.
The two answers are success and no success. Query each question once with
each option order: **78 model attempts maximum**, serial requests, 60-second
HTTP timeout, no automatic retries, and a one-hour wall-time cap. Run after the
adaptive language baseline releases the resident server. No reward controller
is executed using these forecasts in this probe.

Prepare and tokenize every input locally, rejecting inputs over 1,536 tokens
without truncation. Freeze sources, checkpoint/tokenizer/runtime binding,
public inputs, encoded messages/token IDs, and separate exact targets before
inference. Gold probabilities, sampled private worlds, and verifier results
must not enter model requests. Record an attempt before dispatch; retain
failures and stop rather than replacing missing predictions.

Report each option order separately, with deterministic and uncertain targets
as separate slices. Primary probability error is mean squared deviation from
the exact conditional success probability. Also report expected binary log
loss and excess log loss above the exact probability oracle, plus option-order
sensitivity. Uniform probability 0.5 is a fixed comparison. This selection
overrepresents uncertain cases and is not a population calibration estimate.
The two presentations of a target are paired, not independent observations.

The exact labels come from actual executed branches under a fixed continuation,
not from another model's opinion. The probe performs no new SQLite attempts,
no training, temperature fitting, prompt tuning, or checkpoint selection.
Later training comparisons must use new split ownership and fresh evaluation
cases; these opened development targets cannot become a pristine test set.
