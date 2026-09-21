# A normal tool return can still violate the financial goal

A local check reproduced the suspected refund problem in the pinned retail
simulator. Moving an order's payment to a different gift card and then cancelling
the order returns normally, and the order becomes cancelled—but the original
card receives an excessive refund. Our independent accounting check rejects it.
This is a synthetic account in upstream simulator code, not a claim about a real
store, production payment system or the whole benchmark.

The fixture starts with a 20-unit charge on the old gift card, leaving its balance
at 80. The new card has 50. The target cancellation should restore balances to
100 and 50, with zero net charge on both cards.

| Executed procedure | Old card | New card | Target net charge | Verified goal |
| --- | ---: | ---: | --- | --- |
| Cancel the target directly | 100 | 50 | 0 | Pass |
| Move payment to the new card | 100 | 30 | 20 on new card | Pass for migration |
| Move payment, then cancel | **140** | 50 | **−40 on old card** | Fail |

These are simulated units. The final sequence creates a 40-unit excess refund
relative to the declared cancellation goal. Its returned status alone would have
been a misleading success label. Checking that balances agree with the recorded
ledger is also insufficient: the over-refund is internally consistent with that
ledger. The verifier must additionally require the user's intended net charge.

The pinned implementation's cancellation loop constructs refunds from every
payment-history entry without distinguishing payments from prior refunds.
Payment migration adds both a new payment and a refund to that history. The
executed composition exposes this interaction. Upstream code was left unchanged;
the separate multi-item price/option concern remains an unexecuted source finding.

## Qualification scope

Two goal contracts—correct cancellation and correct payment migration—passed.
Stopping, cancelling the wrong order and changing an unrelated card after a
correct migration all failed the appropriate goal or frame check. Six branches
were each independently replayed: **12 world executions and 48 public tool calls**.
Complete states, responses and state-change chains matched exactly; read probes
preserved state. Four focused verifier tests passed.

This is **one financial mechanism and one physical initial state**, not twelve
independent tasks. The database is entirely authored and synthetic. The worker
read the public policy and pinned implementation, blocked official task/database
reads and outbound connections, and made no model calls. No data from the
official retail benchmark was executed.

The useful next data stage is varied, independently verified order and ledger
workflows with missing or stale observations, failed prerequisites and costs. Any
recorded bad composition must receive its actual failed-goal label; do not teach a
model to exploit a simulator refund bug for reward. This qualification supplies
receipts only: **zero admitted retail questions, training presentations or optimizer
updates**. It does not establish better model performance.

[Prospective protocol](retail-ledger-v1-protocol.md) ·
[Aggregate checks](../results/retail-ledger-v1/summary.json) ·
[Pinned upstream tools](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/domains/retail/tools.py)
