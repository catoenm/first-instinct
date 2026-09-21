# Qualify signed-ledger outcomes in real retail tools

This local substrate check follows the static retail source review. Use the same
pinned tau2 source and isolated Python runtime as telecom. Do not modify upstream,
load its retail database, official tasks or solutions, query models, rent hardware,
or change any earlier frozen studies. This is an authored synthetic account in the
real retail tool implementation, not an official benchmark evaluation.

Create one complete synthetic database with one customer, two pending orders,
one product and three gift cards. The target order has one 20-unit payment on the
old card, whose remaining balance is 80. The new card has 50. An unrelated order
has one 30-unit payment on a third card with balance 70. These are simulator units,
not real accounts or money. User identity and card/order identifiers are obtained
from actual public email lookup and user/order-details responses. The simulated
user explicitly permits the tested cancellation or payment migration. Wrong-target
and collateral interventions are negative controls inside the synthetic world.

There are two goal contracts over one financial mechanism and one physical initial
state: cancel the target and refund only its net outstanding charge; or move its
20-unit charge to the new card without changing its items, address or order status.
Use six declared branches, each independently replayed once, for **12 worlds**:

1. Correct target cancellation.
2. Stop without cancelling.
3. Cancel the unrelated order instead.
4. Migrate the payment, then cancel the target.
5. Correct target payment migration.
6. Correct migration followed by an injected change to the unrelated card balance.

Execute public tools through the actual environment dispatch. Record every return,
complete initial/final state, read-only probe and call count. Reject unexpected
exceptions; never convert infrastructure failures into negative training labels.
Block outgoing sockets and attempts to load upstream retail task/database files.
Freeze source and synthetic fixture before execution. Cap each world at 10 calls
and 30 seconds, the whole collection at 120 calls. Stop on infrastructure failure
and preserve partial call journals; no automatic rerun.

Verify signed transaction sums independently: payments add to net charge, refunds
subtract. After cancellation the target's net charge must be zero on every card,
the old card must be 100 and the new card 50. After migration the old card's net
charge must be zero and the new card's 20, with balances 100 and 30 respectively.
Balance changes must also agree with the change in signed ledger entries. Require
the correct status/reason, preserve all unrelated orders, cards, products, customer
fields and target item/address fields, and compare complete states and responses
exactly across replays. A tool returning an order normally is not success.

The composed migration-then-cancellation path is a diagnostic of the suspected
refund issue, not a presumed positive or negative label. Determine its label from
the independent contract, retain the result, and qualify it only as a recorded
counterexample if the verifier rejects a normal-returning path. Do not claim that
the entire external benchmark is broken. The separate multi-item variant concern
remains source-only and is not exercised here.

The correct cancellation and migration must pass; stop, wrong-target and collateral
controls must fail; exact replay and frame/ledger audits must pass. If a purported
positive fails, quarantine the affected task structure rather than weakening its
goal. This stage produces execution receipts and substrate qualification only,
not admitted learning questions, a trained model or a new budget.
