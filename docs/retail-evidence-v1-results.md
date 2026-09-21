# Stale evidence changes what a useful action means

The new retail collection qualifies three goals: change an order's shipping
address, change a customer's default address, and move an existing payment to a
different card. It uses the real pinned retail tools on synthetic accounts, with
independent checks of the requested outcome, money and unrelated state.

The actor starts with actual cached responses. Declared background changes can
make those responses stale before the decision begins: an address may already be
correct, fulfillment may prevent an order edit, funds may have been spent, or the
payment may already have moved. These are authored finite priors, not measured
customer frequencies. No background changes occur during the decision itself.

The same visible history therefore supports different outcomes. Across 36
immediate and continued forecast targets, **30 are fractional**. Seven of the 18
procedures have different probabilities of success immediately after their first
command versus after the declared continuation. An inspection is useful only
when its information changes what is worth doing next.

For example, the payment goal starts equally often in four compatible states.
Blindly attempting the change and inspecting first both finish successfully in
half of them. Inspection avoids unnecessary or refused writes, but costs reads.
With a 20-unit goal reward, the best offered procedure changes with the fees:

| Read fee | Attempted write fee | Best payment procedure | Expected return |
| ---: | ---: | --- | ---: |
| 0.10 | 0.25 | Attempt migration | 9.7500 |
| 0.10 | 6 | Inspect, then migrate when useful | 8.3000 |
| 10 | 6 | Stop | 5.0000 |

All three goals have cost-dependent best procedures. These are optima over the
six offered procedures and declared priors, not unrestricted optimal policies.
Fees include failed attempts; cache collection and private verification are sunk
work. Extra refunds never increase reward.

Scope also matters. A successful shipping-address edit can violate a request to
change only the customer profile. Conversely, a refused shipping edit can leave
the world intact so a subsequent profile edit succeeds. We check the final state
against the goal's preservation contract, rather than treating a normal return as
success or every exception as permanent failure. A different, verified restoration
plan could repair some scope mistakes; this study does not label all of them
irreversible.

## What was executed

| Quantity | Count |
| --- | ---: |
| Authored goal contracts / underlying mechanisms | 3 / 2 |
| Goal/configuration slots / distinct complete initial states | 20 / 10 |
| Distinct alternative branches / independent exact replays | 120 / 120 |
| Actual tool calls | 1,744 |
| Actor / cache / background / private verifier calls | 340 / 720 / 204 / 480 |
| Expected, state-preserving actor errors, including replays | 46 |
| Distinct contradictory fresh-read groups | 14 |
| Goal/world pairs where stopping already succeeds | 9 |
| Admitted training questions / model calls / optimizer updates | 0 / 0 / 0 |

The 120 explicitly authored background transitions are counted separately from
tool calls. Every branch has an exact independent replay and public-response-only
procedure replay. Seven focused tests passed; five saved-receipt corruption checks
reject hidden input, wrong forecast timing, incorrect fee types, infrastructure
errors disguised as outcomes and unrelated mutations. No new worlds were run for
those corruption checks.

Eight physical states and 24 complete trajectory groups are shared across goal
contracts. The entire retail cohort, including the earlier ledger fixture, stays
in **one training-candidate group**. It must not be split into related training and
test tasks. There is no exact complete-state or trajectory overlap with the
reserved telecom evidence. Broader source admission remains a later gate.

This collection improves the execution evidence; it does not show a better model
yet. Next is a bounded live interface that lets a policy choose individual tools,
charges each attempt and pays verified terminal reward once. That interface must
pass local replay and reward checks before another learning comparison.

[Frozen protocol](retail-evidence-v1-protocol.md) ·
[Aggregate audit](../results/retail-evidence-v1/summary.json) ·
[Live-step protocol](retail-live-v1-protocol.md) ·
[Pinned upstream tools](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/domains/retail/tools.py)
