# Retail decisions and consequences now have paired training questions

The completed retail executions now supply **67 distinct training questions**
from 72 authored variants. They add no new independent tasks or world executions.
All belong to one connected training-only component; they are not a new test set.

| Question | Distinct inputs |
| :--- | ---: |
| Success after exactly one attempted command, then stop | 13 |
| Success after the entire displayed continuation | 18 |
| Best offered procedure under the displayed costs | 18 |
| Whether the offered inspection beats the best displayed blind-or-stop procedure | 18 |

The source remains three goal contracts over two mechanisms and ten distinct
physical initial states. Its 120 alternative branches and 120 independent
replays were already executed and verified. Address goals average eight equally
likely compatible worlds; the payment goal averages four. Five immediate-command
variants collapse because their visible input is identical. Every source witness
and variant remains in lineage.

**Twenty-five forecasts retain fractional probabilities.** For example, inspecting
an order's current address has a 50% immediate success rate under the declared
prior. Following the displayed inspect-and-conditionally-update procedure raises
that to 75%. The first forecast stops after the read; the second includes the
specific conditional write. Neither assumes an unspecified future policy.

Every procedure displays its actual commands, argument bindings and branches.
Replaying those displayed programs against all 240 saved traces reproduced every
attempted command and response. Only fixed authored program strings are accepted
by the replay checker; no source-provided or model-generated code is executed.

Decision labels average verified terminal reward minus actual command-attempt
fees across compatible worlds before choosing an action. Errors and redundant
reads cost money; sunk cached observations, background changes and private
verification do not. Six frozen read/write cost pairs generate different optimal
procedures. With a success value of 20, an order-address inspection costs 0.10 and
an attempted write costs 0.25: inspect-and-repair returns 14.8375, versus 14.75 for
the blind write. With read cost 10 and write cost 6, stopping returns 10 and is
best among the offered procedures. These are authored simulated utilities.

Forecast targets use categorical distributions and proper probability loss.
Decision ties, if present, use the protocol's uniform tied-action target. The
180 saved joins between decision/inspection-value rows and their consequence
forecasts keep the labels tied to the same execution evidence.

All inputs fit without truncation: the maximum is 1,910 tokens, with 55 questions
also fitting the current 1,536-token demo limit. The corpus totals 89,812 input
tokens. The independently checked labels, token sequences and training-only
usage are bound to hashes. Exact stored-state, full-trajectory, visible-state and
token-input comparisons found no overlap with the indexed prior sources;
cross-simulator semantic equivalence is outside that exact check. Corrected
telecom ownership remains evaluation-only.

Six fixture tests and eight saved corruption controls pass. The controls reject
hidden current truth, incorrect fractional targets, changed horizons, omitted
costs, missing witnesses, an incomplete compatible-world prior, a display/trace
mismatch and using this training component as transfer evaluation. Local tensor
checks pass for the probability loss and its gradients. No pretrained model,
optimizer or rented hardware was used.

The [aggregate report](../results/release-retail-questions-v1/summary.json),
[corruption controls](../results/release-retail-questions-v1/negative-controls.json)
and [prospective protocol](retail-questions-v1-protocol.md) preserve the evidence.
The [question builder](../release_lab/retail_questions.py),
[displayed programs](../release_lab/retail_programs.py) and
[independent label audit](../release_lab/retail_audit.py) are public; source
receipts and candidate rows remain local. Prepared questions are not consumed
training data: **new training presentations and model updates remain zero**.
