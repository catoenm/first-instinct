# Verified consequences after writes and failed actions

The counterfactual interface now passes local qualification after successful
writes, a refused write, a wrong-scope write, cancellation, and five-action read
histories. It replays each prefix in a fresh copy of every original prior world,
keeps only worlds matching the complete public observation, and executes each
alternative in a separate process. The declared continuation is exactly one
command followed by stopping, including automatic termination at the sixth turn.

This extends the earlier read-prefix checks to histories that actually changed
the world. It uses the same three retail goals and existing synthetic states;
it does not add an independent task family or show model improvement.

| Quantity | Result |
| --- | ---: |
| Public histories / commands per history | 8 / 5 |
| Primary world/history/alternative slots | 260 |
| Compatible primary branches supplying outcomes | 100 |
| Incompatible primary resets screened out | 160 |
| Independent replay resets, including screening | 260 |
| Total new reset executions | 520 |
| Actual tool calls / actor turns including stop | 1,400 / 1,896 |
| Documented command refusals, including replays | 184 |
| Existing original goal/world pairs / physical states | 20 / 10 |
| Unique candidate forecasts / fractional targets | 40 / 13 |
| Admitted training questions / model-training presentations | 0 / 0 |

All replay states, responses, labels and rewards matched exactly. Within each
original world, the state after replaying a prefix also matched across every
alternative. Screening depends only on the observations before the proposed
command, never on its future outcome. Each compatible original world receives
its conditional prior weight once; replays add no probability mass.

## Why original state still matters

One history made distinct original worlds converge to the same current database.
The goal was to change a customer's profile address while preserving the order's
shipping address, but an earlier command had changed the order address instead.
The returned order can look identical whether that write changed the original
shipping address or merely rewrote its existing value. Those worlds have different
preservation obligations even though their current database and visible response
match.

Under the declared conditional four-world prior, the independent branches give
**25% verified goal success for stopping** and **50% after changing the profile
address and stopping**. A successful profile update cannot undo an already broken
order-preservation requirement. Resetting the verifier's baseline to the state
after the wrong write would incorrectly reward that history; retaining the original
state avoids the error. These probabilities concern authored finite priors, not
real customer frequencies.

The same rule applies to every branch: verify the user's goal, unrelated-field
preservation and monetary requirements against the original world. A normal
command return is not the outcome label. Candidate targets describe goal success;
they do not label the best action under an unrestricted repair policy. The full
receipt ledger includes replayed prefix fees for accounting; those costs are
already sunk at the forecast question and must not be charged again in a future
continuation-value target.

## Qualification and limits

The independent audit rebuilt conditional memberships and all 40 target
distributions from complete states, actual command journals and the original
goal verifier. Seven CPU tests cover prior membership, replay weighting, missing
worlds, action-dependent selection bias, converged states, original preservation
requirements and the one-turn remaining horizon. Four additional saved-receipt
corruptions—label, compatibility flag, original state and source identity—were
rejected without executing more worlds. A private negative-test launcher initially
failed to locate the repository import before running any check; the corrected
launcher and failed attempt remain saved separately.

All forecast inputs fit without cropping: maximum **2,681 tokens**, below 4,096.
The largest local guarded job stayed below 0.59 GiB resident memory with no added
system swap. No foundation model, optimizer or GPU rental was used. All rows keep
the existing `retail_workflows` ownership group; the new history-source tag does
not make them an independent mechanism.

These 40 questions are execution-qualified candidates. Training admission still
needs token-conflict/deduplication checks against the existing mixture, exact
source-role binding and a new frozen learning recipe. Fresh learned-policy
collection and broader mechanism transfer remain separate gates. The selected
model and unopened release evaluation are unchanged.

Execution freeze:
`e2bf5ffda4ac7b093e89cb11c7045728c2b1325726797f395741658269bf13f4`.
Candidate question hash:
`834bc4136276f43603aa03f8a7e5941593d23f02be2ea2ece0188e04278aeeeb`.

[Prospective protocol](retail-branches-v1-protocol.md) ·
[Independent aggregate audit](../results/retail-branches-v1/summary.json) ·
[Saved-receipt negative controls](../results/retail-branches-v1/negative-controls.json)
