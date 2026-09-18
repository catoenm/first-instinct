# Post-hoc ToolSandbox answer-order audit

This supplementary diagnostic was specified **after** the supervised reference
completed: its forecast selector chose stop at all 48 initial contexts, while
stop was optimal at 17. It asks whether those predictions are sensitive to
answer order. It does not change the primary 720-question result, select a
checkpoint or prompt, or provide evidence of reinforcement-learning gains.

Use all 240 nontrivial initial-state questions in their existing corpus order:
144 outcome questions and 96 cost questions, five per root across all 48 roots.
Reverse each complete option list exactly once, preserving state, question,
semantic IDs and descriptions byte-for-byte. Query each reversed input once;
reuse its original prediction from the completed published reference. Do not
query original order again, sample questions, try other permutations, omit
failures, or choose a better ordering. The 48 initial stop-cost singletons
remain deterministic zero-cost bypasses. Phone-state questions are not queried.

Preparation checks the original read-only analyzer, exact completed raw-result
hashes from the published summary, original freeze/mapping, complete corpus,
supervised step-2742 adapter files and receipt, current package versions,
tokenizer/config files, and runtime/serialization sources. A different
self-consistent result stream cannot substitute for the published reference.
The cached tokenizer runs locally with offline flags and a Python socket guard;
no model weights or HTTP service are loaded/contacted. Preserve all reversed
public inputs, exact token streams and labels, and the 1,536-token/36-choice
limits. Reject overlength input; never truncate or filter. Pin the new helper,
tests and this protocol plus imported helper/analyzer sources. Prepare a new
folder, publish its freeze, then execute separately. Do not prepare a real
freeze or run inference merely by testing this implementation.

Execution uses the existing owned demo only, on loopback port 8766, and the
explicit PID frozen by `--expected-server-pid`. The intended current PID is
87342; no service startup, replacement, model load, training, rental or tool
execution belongs to this helper. Check the listener PID/start identity before
and after; check service checkpoint/device metadata on every response and
the session token again at completion. The token stays in memory. The unchanged
`LocalPredictor` callback preserves the original single-question typed-choice
serialization, disables proxies and rejects redirects, with a 60-second HTTP
timeout. Metadata and disk hashes do not attest to the actual resident tensors.

The cap is 240 forwards, no retry or automatic resume, with a 3,600-second wall
deadline covering preflight, calls, final verification and scoring. Check time
before and after every request; a process alarm also interrupts the workflow
where Python can receive signals. This is not an OS-enforced hard kill during
uninterruptible native work. Only the caller's service lifecycle may impose an
additional external termination bound. A budget failure preserves partial
artifacts and is not a completed experiment.

Before every request, flush and synchronize its attempt ledger. Preserve the
received response before validating its metadata/probability map, synchronize
the unchanged callback's probability journal, then validate finite normalized
probabilities against exactly the offered IDs and synchronize a validation
ledger. Record attempted, received and validated counts separately. Any error
stops subsequent questions; error receipts contain the exception class, not
token-bearing error text. Existing results directories cannot be resumed or
overwritten. After all responses, recheck source, reference, corpus, checkpoint,
runtime, tokenizer, token streams, freeze bytes and service identity before
writing a complete receipt. Read-only `analyze` requires all four journals,
reproduces the probability and timing checks, and independently regenerates
the comparison with the unchanged transfer scorer.

Public read-only analysis may remap the reference and corpus folders using
`analyze --reference-folder ... --corpus-folder ...`. Exact published bytes and
source hashes remain required. Model packages, checkpoint weights and the
tokenizer cache need not be installed to rescore retained predictions: report
absent artifacts as unavailable and distinguish historical inference versions
from current analysis versions. Available but changed adapter/cache files still
fail. This exception applies only to `analyze`; preparation and execution retain
all strict runtime, disk and tokenization requirements.

Report semantic-ID probability changes separately for outcome and cost:
mean total variation, summed squared drift, maximum absolute drift, modal-ID
flips and modal-set changes, with exact tie counts. Modal ties follow first
presented option; modal-set changes distinguish an arbitrary tie switch from a
different maximizing set. Retain every per-question probability pair. Report
probability scores without the unchanged scorer's canonical-order modal
accuracy, whose tie rule would not describe reversed presentation. Report
the original and reversed implied initial action, expected value and regret
with equal weight per root, retaining exact singleton stop costs and the same
declared continuation. Original phone predictions may be passed through the
unchanged 720-input scorer internally; no phone comparison is reported.

These are implied decisions under authored finite priors and existing tool
execution labels, not newly realized policy returns or general calibration.
The original and reversed calls occur at different times; with no new
original-order repeats, temporal or numerical variation is not separately
estimated. One reversal does not establish permutation invariance. Timing is
bookkeeping, not a controlled latency comparison. No ordering is promoted.

```sh
.venv/bin/python -m general_lab.toolsandbox_transfer_order prepare \
  --folder output/toolsandbox-transfer-order-v1 \
  --reference-folder results/toolsandbox-transfer-supervised-v1 \
  --corpus-folder results/toolsandbox-partial-v1 \
  --adapter-run output/general-supervised-complete-v1/runs/supervised-01 \
  --tokenizer-cache /absolute/path/to/the/original/cached/tokenizer \
  --expected-server-pid 87342

# Separate execution after publishing the immutable preparation artifacts.
.venv/bin/python -m general_lab.toolsandbox_transfer_order run \
  --folder output/toolsandbox-transfer-order-v1
.venv/bin/python -m general_lab.toolsandbox_transfer_order analyze \
  --folder output/toolsandbox-transfer-order-v1
```
