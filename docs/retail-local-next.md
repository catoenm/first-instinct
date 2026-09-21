# A different executable mechanism after telecom

The pinned telecom runtime also contains the upstream retail implementation.
Reviewing its source, without loading official tasks, solutions or evaluation
results, suggests a useful next local qualification: money conservation and
order-state prerequisites. Reuse the pinned dependency and network-blocking
arrangement; do not build another imitation service or rent hardware first.

Two potentially useful task structures are changing an order's payment method
while preserving the exact net charge, and cancelling a pending order while
refunding only the outstanding payment. Gift-card balances and payment/refund
history permit independent checks against the complete database. Customer-profile
address changes versus order-address changes offer a further scope distinction,
but should not all be declared independent mechanisms just because tool names
differ. Whole underlying workflows must own every variant and split.

Source inspection also identifies reasons to verify actual state instead of
trusting a returned order object:

- `cancel_pending_order` loops over every existing payment-history entry when
  constructing refunds; it does not branch on each entry's transaction type.
  `modify_pending_order_payment` appends a new payment and a refund to that same
  history. Their composition may over-refund an account.
- `modify_pending_order_items` validates variants in one loop, then its later
  mutation loop reads the earlier loop's final `variant` when assigning every
  replacement's price and options. A multi-item request may return successfully
  with incorrect item fields.
- An exchange request is a stored request, not a completed physical exchange.
  Forecasts must name the actual immediate event that execution verifies.

These are **static-source findings, not reproduced failures or dataset labels**.
Do not train on them, claim a benchmark vulnerability result, patch the pinned
upstream implementation, or quietly simplify a fixture until its verifier passes.
Before any retail execution, freeze a small protocol with authored synthetic
fixtures, explicit user authorization inside the simulated task, independent
signed-ledger arithmetic, full frame checks, correct/no-op/wrong-target controls,
expected-error handling, complete-state replay and a strict world/call ceiling.
Execute suspicious compositions as negative controls. Quarantine task structures
whose success semantics the existing substrate cannot faithfully support.

This review reads only
[the pinned retail tools](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/domains/retail/tools.py)
and data-model source. No retail database, official benchmark task or model was
executed or evaluated. The qualified telecom question-admission work can continue
independently; three telecom contracts alone are not grounds for another rental.
