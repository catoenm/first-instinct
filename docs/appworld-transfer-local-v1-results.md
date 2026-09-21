# Four reserved application programs pass the first execution check

The seven payment-related application programs were reserved before the original
application-data collection. We selected one representative per program using
only metadata, then executed a no-op, reference solution and independent reference
replay under [the frozen local protocol](appworld-transfer-local-v1-protocol.md).
No model saw these tasks and none entered training or checkpoint selection.

Four programs pass the complete substrate contract. Three remain quarantined:
one produces different saved database hashes on reference replay; two raise
non-assertion exceptions when evaluating a no-op (`IndexError` and `ValueError`).
All seven reference solutions and all seven reference replays pass the task
assertions. That alone is insufficient: a usable verifier must also handle failed
behavior cleanly, and replay must meet the declared state contract. We did not
replace the failed representatives with easier instances or relax the gate.

| Quantity | Count |
| --- | ---: |
| Reserved task programs | 7 |
| Reserved underlying task instances | 21 |
| Representative task instances executed | 7 |
| Isolated world attempts | 21 |
| Clean execution receipts | 19 |
| Application calls in clean receipts | 480 |
| Passing programs | 4 |
| Quarantined programs | 3 |
| Model calls, training questions and optimizer updates | 0 |

The two failed no-op executions have no completed receipt; their call counts are
not included in the 480. Each process had a 96-call bound. No outgoing service
calls were allowed. The upstream development and test sets remain untouched.
Raw tasks, solutions, logs and database states remain private. The
[public receipt](../results/appworld-transfer-local-v1/summary.json) contains
aggregates and hashes only.

The next local stage executes alternative actions on the four passing worlds,
including failed authentication followed by recovery and redundant reads, with
independent alternative replays and collateral-change controls. Its
[branch protocol](appworld-transfer-branches-v1-protocol.md) is separately frozen.
Public-information argument closure and context admission still follow; passing
this first screen is not a qualified transfer dataset or a model transfer result.
All four programs must survive those later checks before a transfer inference
rental is justified.
