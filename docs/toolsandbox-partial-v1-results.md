# Partially observed ToolSandbox pilot v1: executed results

The frozen local collection completed on 2026-09-18: **48 public contexts, 192 concrete database worlds, 1,152 distinct world/program labels, and 432 exact conditional forecasts**. Every program was executed twice, producing 2,304 production branches with matching replay receipts. No model inference, training, paid compute, new dependencies, or external service was used.

The [prospective protocol](toolsandbox-partial-v1-protocol.md) defines the visible prior, public history, continuation, costs, outcome contract, grouping, and budget. The new implementation is [toolsandbox_partial.py](../general_lab/toolsandbox_partial.py); the original fully observed adapter and frozen GPU experiment were not changed. All downloaded ToolSandbox material remains ignored; the Apple license/dependency notices are described in [the original pilot notice](toolsandbox-pilot-v1.md).

## What was actually verified

The four hidden worlds are real ToolSandbox database states. Exact-phone lookup identifies the current contact name; the desired reminder content is constructed from that visible result. In one world, five off-window reminders rank above the desired reminder in the bounded fuzzy search. The content filter runs before the timestamp filter, so the combined query returns empty despite the desired row existing. A timestamp-only query finds it.

All 192 registered worlds passed actual query-count checks. The desired longer content scored **90**, while all five short-content distractors scored **100**. Target exclusion therefore does not depend on ties among the distractors. The recorded phone histories are identical within W1/W2 and within W3/W4, including IDs, field values, response order, costs, and clock/tick fields.

Posteriors use the likelihood of those actual full histories under the declared public prior. The collection then executes each offered action and the specified continuation in every compatible world, weighting verified outcomes with exact rational arithmetic. The corpus contains outcome, future-cost, and joint distributions. It does not substitute a sampled sensor label or a hand-written success probability for execution.

The read-only artifact audit independently reconstructed every terminal label from complete initial/final state, all 432 posterior/marginal/joint distributions, event-metered costs, visible provenance of mutation IDs, replay digests, and file hashes. A separate reviewer then independently recomputed all 432 forecasts and checked all 864 public question payloads; that review found no blocking issue and executed no tools or models. Mutation exposure was also checked from every actual program trace:

| Split | Public contexts | Executed write API |
| --- | ---: | --- |
| Train | 16 | `add_reminder` |
| Validation | 16 | `modify_reminder` |
| Test | 16 | `remove_reminder` |

The search APIs are shared. Private fixture setup uses add functions in every split; setup calls are never model inputs or policy trajectories. Thus the claim concerns exposed write APIs, not unseen internal database methods or unseen read tools.

## Query value changes with its actual output cost

For the default prior `(3,2,2,1)`, observing the B contact name leaves W3/W4 with posterior `(2/3,1/3)`. In the update task, actual executions yield:

| Offered action after the phone lookup | Outcomes | Expected utility, row price 1 | Expected utility, row price 12 |
| --- | --- | ---: | ---: |
| Stop | Missed change 2/3; justified decline 1/3 | −80/3 | −80/3 |
| Fast content+time query | Missed change 2/3; justified decline 1/3 | −88/3 | −110/3 |
| Complete time-only query | Completed change 2/3; justified decline 1/3 | **160/3** | −148/3 |

The complete query is best at the lower row price; stop is best at the higher price. Costs come from actual returned row counts: `2 + price × returned rows`, plus 3 for a write. These are declared research credits, not claimed service charges. Earlier prefix costs are excluded from these forecasts.

Creation also exposes the harmful counterpart: trusting a truncated empty response can create an actual duplicate. The 1,152 unique world/program labels contain 256 completed changes, 160 already-satisfied cases, 480 justified abstentions, 224 misses, and 32 duplicate creations. These are counts of correlated counterfactual programs, **not estimates of outcome prevalence in real usage**. Prior-weighted distributions are stored separately.

## Prompt readiness and current limits

There are 864 marginal questions: **720 questions requiring prediction and 144 singleton cost answers** that deterministically bypass the model. The cached Qwen3.5-9B tokenizer and current chat template measured every declared root/history/action question without inference. All 720 model prompts fit: **1,078–1,458 tokens**, against 1,536; at most **22 choices**, against 36. There was no truncation or label-dependent filtering.

`public-questions.jsonl` contains the exact public-only question payloads whose hashes match that tokenizer audit. Its `deterministic_bypass` flag marks singleton costs, which must not be passed through the encoder's two-choice minimum. Full verifier receipts and exact targets remain outside each question's `input`.

This establishes prompt compatibility, not training readiness of an approved experiment. No empirical-label sampling/training protocol, optimizer comparison, learned calibration result, policy evaluation, or retained-language evaluation has been run on this corpus. Feeding the raw forecast/receipt envelope directly to the model would be incorrect; use the exported question payloads or the specified `question(public_input(...), action, kind)` rendering.

## Review, freeze, and accounting

Independent protocol/source review approved collection after two corrections: the freeze includes the imported base adapter, and frame checks protect existing or ambiguous targets while allowing only the documented timestamp changes on a unique update target. Eight isolated-runtime tests passed, including real replay, negative mutations, impossible-history rejection, rational forecasts, and a cost-order reversal. In the normal project environment, six core tests pass and the two optional integrations skip when their dependencies are absent.

The prospective freeze hashes the protocol, new adapter/test, imported base adapter, pristine upstream source map, dependency pins/full runtime versions, 192-world guard receipt, and independent approval. Freeze SHA-256: `8dd4cad766f6434cc86034561d245e1f051170cdc425d42aadf1eccb97e1b258`.

The shared append/fsync ledger charged every new real-tool branch before execution, including tests, repeated guards, and deliberate impossible-history failures. It records **2,730 new attempts**: 2,304 production, 392 guard, 24 other integration-test, and 10 negative-gate branches. The conservative historical debit is 1,322; retained previous files alone establish a lower bound of 960, while prior task history establishes the larger conservative charge. The combined debit is **4,052 / 4,096**, leaving 44. It is not represented as an exact count of all historical executions.

Production additionally reports 30,528 setup calls, 1,152 prefix calls, 1,536 offered-action calls, and 1,344 continuation calls, including repeat executions. These call counts are distinct from branch counts and unique labels.

The Python socket/subprocess guard blocked its four active probes plus one socket-creation attempt during imports. No additional attempt occurred during actual tool replay. This is Python-level interception, not an OS sandbox guarantee.

## Published artifacts

The [published receipts and data](../results/toolsandbox-partial-v1/) are byte-for-byte copies of the collected artifacts under `output/toolsandbox-partial-v1/`. Absolute local paths inside the original freeze are retained as provenance; the corresponding files are included beside it in the public directory.

- `review.json`, `guards-final.json`, `freeze.json`: independent pre-collection approval and prospective evidence.
- `post-collection-review.json`: independent completed-artifact audit, SHA-256 `8437a008d03957c1c1ad644bbbcaa3edd8544f43cf1c059c1e460ba50e37b296`.
- `collected/manifest.json`, `collected/executions.jsonl`, `collected/forecasts.jsonl`: actual counts, full state/trace receipts, and exact rational distributions.
- `prompt-audit-final.json`, `public-questions.jsonl`: tokenizer measurements and matched question payloads.
- `artifact-audit.json`: read-only reconstruction of every label and distribution.

The [published budget ledger](../results/toolsandbox-partial-v1/budget-ledger.jsonl) preserves the branch accounting; test/guard/collection logs remain in the corresponding `.local/toolsandbox-partial-*.log` files. Do not repeat real integration tests or collect again casually: they debit the same remaining budget. Read-only artifact/tokenizer audits do not execute tools or consume branch capacity.

This is a meaningful new local diagnostic for query completeness, cross-tool value construction, and state preservation. Retry/workflow v2 already cover evidence acquisition and explicit continuations; those concepts alone are not new here. The useful difference is that uncertainty and failure depend on third-party executable query semantics and concrete rows. Forty-eight contexts still represent one authored four-template mechanism with three write operations. They provide no evidence of broad Jev-like capability, general arbitrary-question calibration, or real-world prior accuracy.
