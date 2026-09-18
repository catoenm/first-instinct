# Proposed partially observed ToolSandbox pilot v1

Status: prospective protocol, 2026-09-18. No collection or learning result is claimed here. This is a new adapter and new artifacts; the completed fully observed pilot remains unchanged. Full collection requires independent protocol/leakage review and passing bounded guard tests.

## Question and genuine mechanism

Can a typed predictor distinguish the risk of acting on a cheap, incomplete query from the value and cost of a complete query, when the labels come from executing real contact/reminder tools?

Use pristine ToolSandbox commit `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1` and the existing isolated Python 3.11 runtime. Its [fuzzy filtering helper](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/common/utils.py) calls `process.extract` without overriding the five-result limit; later filters operate on that already limited result. [Reminder search](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/tools/reminder.py) applies content filtering before timestamp bounds. [RapidFuzz documentation](https://rapidfuzz.github.io/RapidFuzz/Usage/process.html#rapidfuzz.process.extract) and installed 3.9.3 Python source confirm the limit and index-based tie ordering. Actual pinned-runtime gates must confirm the proposed fixture behavior before collection.

The desired reminder has content `Call <current contact name> about invoice`, at timestamp T. A fast query uses the shorter content `Call <current contact name>` plus exact timestamp bounds. Five off-window reminders whose content is exactly that short query can rank strictly above the desired longer-content reminder. Filtering their top-five results by time then returns empty. A time-only query returns the desired row. This is an upstream query/completeness mechanism; the adapter must not replace its response with a sampled empty/nonempty signal.

The target must rank **strictly below all five distractors**. Ties among those five do not affect exclusion. If the actual runtime does not satisfy this gate, fail and revise the prospective protocol rather than collecting with an assumed rank. Do not depend on the desired row losing an undocumented equal-score tie.

## Public request and four-world prior

The request identifies a contact by exact phone number Q. The authoritative contact's current name is either the public alias A or public alias B; it is initially unobserved. The reminder content is constructed from the name returned by `search_contacts(phone_number=Q)`, not from a hidden target label. The action is one of:

- **Create:** if no exact matching reminder exists at T, create exactly one; if exactly one already exists, preserve it; if multiple exist, leave state unchanged and decline.
- **Update:** if exactly one exists, reschedule it to T+3600; otherwise preserve state and decline. The documented `modify_reminder` side effect resetting `creation_timestamp` to the public execution clock is explicitly permitted in the public goal; preserve every other unrequested field and unrelated row/namespace.
- **Delete:** if exactly one exists, remove it; otherwise preserve state and decline.

“Exact matching” means exact content equality to the complete requested string and timestamp equality to T. Fuzzy similarity is only a search behavior, not the verifier's target definition.

Every root has these four fully specified possible initial database templates:

| World | Name at Q | Exact targets at T | Other relevant reminders | Default weight |
| --- | --- | ---: | --- | ---: |
| W1 | A | 1 | None matching the short query | 3 |
| W2 | A | 0 | None matching the short query | 2 |
| W3 | B | 1 | Five short-query exact matches outside T, each strictly outranking the target | 2 |
| W4 | B | 2 | No truncating off-window matches | 1 |

All worlds also contain eight unrelated reminder rows at T, one protected unrelated reminder outside T, and a protected unrelated contact. The eight in-window rows make the complete query's returned payload larger. Their content must rank below the target, and the runtime gate must show the fast query returns exactly 1/0/0/2 rows in W1/W2/W3/W4 after all filters. The complete time query must return exactly 9/8/9/10 rows.

The prior is authored, not fitted to customer traffic. Four predeclared positive weight profiles are `(3,2,2,1)`, `(1,3,3,1)`, `(3,1,1,3)`, and `(1,1,1,1)`. Normalize each with exact rational arithmetic. There are no independent success/sensor coin flips. All outcomes are deterministic given a complete database template and action program; uncertainty is over the initial database.

**The complete prior specification, normalized weights, aliases, row-count/shape assumptions, query truncation semantics, costs, goal contract, and continuation must appear in every public forecast input.** Only the realized world, concrete unobserved rows, and verifier state remain private. A model must not need access to an unpublished prior to interpret an “exact” probability.

Generate opaque entity IDs from root identity plus stable entity-slot names, excluding world identity. Keep common rows, clocks, phone values, and contact IDs identical in worlds grouped under one observation. Never seed UUIDs from W1/W2/W3/W4 or shift shared IDs merely because another hidden row is absent. Contact lookup must produce byte-equivalent public histories for W1/W2 and separately W3/W4, while the two groups differ in the returned name. All collection roots retain the same four-world prior even when later histories eliminate worlds.

## Public programs, continuation, and costs

At the initial public state offer three actions: stop; look up Q; query reminders using timestamp bounds T..T without a content filter. At each of the two possible phone-lookup prefixes offer three actions: stop; fast content-plus-time query; complete time-only query. All callable arguments come from the request or actual visible returns. Mutations may use only IDs in visible query results. No hidden-state masking of options is permitted.

Every non-stop offered action is followed by the same declared `fast_then_act` continuation:

1. If Q has not been looked up, perform its exact phone lookup. If it does not return exactly one contact, decline without writing.
2. If no reminder result is available, perform the fast query using the returned contact name and T. If a complete time query is already available, reuse it.
3. Filter the available returned rows locally by exact complete content and T. For create, add on zero matches, preserve one match, decline on more than one. For update/delete, apply the requested write only on exactly one match, otherwise decline. Then stop. Any exception stops the program immediately.

This continuation deliberately acts on what the query returned and may mistake a truncated empty result for absence. It is a transparent fallible program, not an oracle or optimal planner. Exact forecasts describe this program, including its mistakes. A later replanning controller would require separate executed evaluation; these forecasts would not automatically calibrate that different policy.

All calls, including validation errors, are charged. A read costs **2 + p × number of returned rows** credits, where `p` is publicly one of `1,4,12,20`. A write costs 3 credits. Stop costs zero. These are explicit research utility units, not a claim about ToolSandbox billing or wall-clock time. Costs depend on actual tool outputs. Failed reads cost the base 2. Each rollout has at most two reads and one write; prefix replay costs are recorded but excluded from the forecast's future-cost target.

Public cost menus are conservative supersets constructed from program structure: a fast query can return 0..5 rows, a complete query 0..10, and the guaranteed unique contact lookup returns one. Include both possible write/no-write branches. Thus each question has at most 22 cost classes; never prune options using the posterior's private support. Outcome probabilities and cost probabilities normalize separately, not across offered actions. Preserve the joint distribution in verifier receipts so dependence can be inspected.

Stop's future cost is the singleton `[0]`. It is a deterministic bypass, not a valid input to the existing encoder's minimum-two-option model path. Count these separately from questions requiring prediction; do not add a fabricated second cost to satisfy the encoder.

## Exact conditioning and outcome verification

For prior weights w and observed public action/response history h, determine compatibility by replaying **the actual observed actions in every candidate world** and comparing the full observable responses, returned list order, errors, costs, and clock/tick history. Do not condition using a manually assigned “sensor outcome.” An impossible public history is an error, not permission to fall back to the prior.

For compatible worlds, posterior weight is `w(world) / sum_compatible w`. Execute the offered action and declared continuation in every compatible world. The exact probability of outcome y is the sum of compatible posterior weights whose verified execution produces y. Compute cost and joint distributions the same way with `Fraction`. Root forecast traces can supply cached actual phone-prefix observations; do not execute extra uncounted probes to construct conditioning groups.

The verifier checks all five database namespaces, all untouched rows and fields, reminder creation timestamps, and ID uniqueness. It independently derives the initial exact target set from Q's authoritative contact and exact content/time predicate. Terminal classes are: completed change; already satisfied; justified abstention; missed change; duplicate created; wrong target or collateral change; invalid call; other failure. A new duplicate gets its own class even when the upstream add call returns normally. Initial duplicate rows are not charged as newly created duplicates. An error after a partial write cannot pass success.

Predeclared utilities are respectively `100, 100, 0, -40, -150, -200, -80, -80` before future costs. Justified abstention applies only when the request explicitly requires declining on absent/ambiguous targets; it is not a free success label for declining a feasible requested change. Create with one existing target is already satisfied; update/delete with one untouched target is missed.

Prospective sanity check, not yet an execution result: with default weights, a phone lookup returning B gives posterior `(W3,W4)=(2/3,1/3)`. For update at p=1, the fast query should miss with probability 2/3 and decline with 1/3; costs should be 2 and 4. Complete query should complete with probability 2/3 and decline with 1/3; costs should be 14 and 12. Expected utilities are then `-88/3` for fast, `160/3` for complete, and `-80/3` for stop. At p=12, complete-query utility should fall to `-148/3`, below stop. These numbers must emerge from actual executions and the public meter; do not hardcode them as labels.

## Splits and leakage boundaries

Use 48 public contexts: 16 create/train, 16 update/validation, and 16 delete/test. Each operation cycles through all 16 combinations of prior profile and row price. Names, Q, IDs, and all cost/weight variants retain their operation-group ownership.

Shared read primitives appear in every split. Training writes use only `add_reminder`, validation writes only `modify_reminder`, and test writes only `remove_reminder`. Training negative cases must use add-only errors/duplicates; do not reintroduce held-out remove/modify primitives as collateral programs. Fixture setup uses add functions in every split, but setup calls are private receipts, never policy/training trajectories. Audit both requested-goal groups and actual proposed/executed write primitive exposure. This corrects the completed pilot's narrower holdout: its deletion goals were held out, but deletion primitives occurred in training collateral candidates.

Every world, prefix, action, replay, prior/cost variant, and derived row of a public root remains in one split. Bootstrap units, if later used, must be whole public roots, not worlds, prefixes, or duplicated labels. Even 48 roots still represent one authored four-template query mechanism with three requested write operations.

## Bounded prospective collection and gates

Use a persistent atomic budget ledger, shared across tests and CLI processes. Debit before execution, including failed attempts. Conservatively charge the earlier work **1,322 branches**: 768 completed collection branches; two successful integration suites of 192 branches each; two branches in an early failed suite; and at most 168 branches in a suite that failed during contact-removal replay comparisons. The retained files alone establish a lower bound of 960 (collection plus final successful suite); this task's tool history establishes the additional runs and the upper bound. This is a conservative historical charge, not a claim that the historical count is exactly 1,322. The independent reviewer executed zero additional upstream branches. The new allocation is:

| Work | New branch executions |
| --- | ---: |
| Root forecasts: 48 contexts × 4 worlds × 3 actions × 2 repeats | 1,152 |
| Phone-conditioned forecasts: 48 × 2 histories × 2 worlds × 3 actions × 2 repeats | 1,152 |
| Guard, posterior, and negative-program integration checks (including all 192 registered worlds) | At most 256 |
| Reserved failed attempts or additional corrective checks | At most 214 |
| New maximum; combined with conservative historical charge | 2,774; **4,096 total** |

There are 192 concrete initial database worlds behind the 48 public contexts. Together with the previous 64 concrete fixtures, this is 256 initial roots, below 512. Guard tests reuse selected registered roots. Reset/prefix replay remains part of its charged branch, not a new independent observation. Count and report all setup, prefix, offered, and continuation calls separately as well.

Before full collection, freeze the protocol hash, new adapter/test hashes, pristine upstream hashes, dependency versions, root/split registry, exact prior profiles, costs/utilities, program grammar, continuation version, bounds, and budget ledger state. Obtain independent review of this protocol and implementation. No source changes after this freeze without a new version and review.

Required gates: strict fuzzy-rank separation and actual 1/0/0/2 versus 9/8/9/10 query counts; identical histories within each expected group; posterior equals conditioning on actual observations; byte-identical public input/menu for hidden-equivalent states; rejection of impossible histories; visible-ID provenance; reset/replay equality; conservation of unrelated namespaces/rows/fields; rejection of duplicates, wrong-target writes, successful writes plus additional mutation, and errors after mutation; empty-query absence is not assumed; exact probability/joint sums and cost expectations; action-cost ordering reversal in the stated fixture; actual primitive split audit; source pin and environment guard checks. Only absent optional dependencies may skip integration tests. Negative tests and retries use the same ledger.

Reuse the clean environment, pinned source checks, deterministic time/UUID controls, and Python socket/subprocess guard from the previous work, with the guard's same explicit limitation: it is not an OS sandbox. No models, services, network during replay, new dependencies, GPU, or training are needed. Store new artifacts separately; publish no commit before root review.

## Expected contribution and limits

This adds real implementation-dependent query truncation, cross-tool value construction, variable result sets, negative-evidence ambiguity, row identity, and full-state preservation. Retry v2 already tests stale evidence and duplicate harms, and workflow v2 already tests information value and explicit continuations. Therefore uncertainty or duplicate penalties alone are not new contributions. The useful advance is making those decisions depend on third-party executable query semantics and actual database rows, with exact history-conditioned probabilities.

It remains a tiny, authored finite prior. Exposing that prior makes calibration well-defined but also permits a small exact planner to solve the problem. A good model score would not establish arbitrary-question calibration, real-user priors, independence of mechanisms, or Jev-like broad capability. The appropriate deliverable is verified outcome-distribution infrastructure and a harder local diagnostic, not a capability claim.
