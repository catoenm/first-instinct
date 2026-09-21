# Hidden causes change what a good forecast means

The next local data qualification passed. Real telecom tools now produce different
outcomes from **identical public observations**, under a declared finite prior.
This provides execution-derived probability targets and cost-sensitive decisions.
No model was trained or evaluated in this stage.

Each of three task contracts has four equally likely compatible worlds. The actor
sees the same customer lookup, a historical complaint, the potentially faulty
prerequisites and their independent 50/50 prior. The private initializer and
verifier are never actor observations. The four worlds include a case where
service already works. This is a deliberately defined probability exercise, not
an estimate of real customers' failure rates.

For example, the device task can have airplane mode on or off and mobile data on
or off. With only the initial public history, service works in one of four worlds.
Executing a speed probe therefore leaves service working with **25% probability**.
Executing that same probe **and then following the declared repair procedure**
restores service in all four worlds. Its continued success probability is **100%**.
The difference comes from executed later commands, not confidence invented by a
model. A failed zero-gigabyte purchase similarly leaves the world unchanged;
the declared recovery procedure can still complete successfully afterward.

We executed six alternatives separately in every world: stop, toggle mobile data,
take a mechanism-specific blind action, inspect and repair, repeat an inspection
before repairing, and recover from an invalid purchase before repairing. The
continuation branches only on actual public speed, network-status and usage
responses. Customer and line identifiers come from the public lookup. A second,
independent execution exactly reproduced every branch's complete states,
responses, errors, charges and outcome labels.

## Cost can reverse the correct decision

The utility values below are **simulated dollars**, with verified service worth
20, each attempted non-read command costing 0.25, and actual additional billing
deducted separately. Inspection fees apply to every actor read, including redundant
reads; the already-paid customer lookup and private verifier calls are excluded.
We average outcomes across compatible worlds before choosing an action.

| Mechanism | Best choice at 0.10 per read | Expected return | Best choice at 10 per read | Expected return |
| --- | --- | ---: | --- | ---: |
| Device switches | Inspect and repair | 19.575 | Stop | 5.000 |
| Carrier and device roaming | Inspect and repair | 19.5125 | Enable carrier roaming, then stop | 9.750 |
| Billed allowance and data switch | Inspect and repair | 17.000 | Stop | 5.000 |

The high-cost choice is an expectation under missing information; it is not an
oracle that sees the hidden fault. At the intermediate inspection fee of 2,
inspection and repair remains best in all three mechanisms. Redundant inspection
reaches the same terminal state but loses exactly one inspection fee. The
recoverable invalid command loses exactly one non-read fee without an extra bill.
Thus a longer command sequence is not automatically preferable.

## Execution and accounting

| Quantity | Count |
| --- | ---: |
| Authored task contracts / mechanisms | 3 / 3 |
| Compatible configurations across contracts | 12 |
| Distinct complete physical initial states | 10 |
| Physical states reused from earlier qualification / new states | 3 / 7 |
| Initial visible information sets | 3 |
| Alternative branches / independent replays | 72 / 72 |
| Total new world executions | 144 |
| Total calls | 750 |
| Actor calls / sunk lookup calls / private verifier calls | 318 / 144 / 288 |
| Expected failed commands, including replays | 24 |
| Fractional immediate or continued probability targets | 25 of 36 |
| Alternatives whose immediate and continued probabilities differ | 9 of 18 |
| Prepared training questions / consumed presentations / optimizer steps | 0 / 0 / 0 |

Some distinct contracts share the exact same physical state, which is why twelve
configurations are not twelve unique physical states. Replays do not increase the
prior weight of a world. Cost variants do not create new mechanisms. The earlier
qualification, including its failed adapter attempt, remains separately preserved.

An independent stored-state predicate agrees with real cellular speed probes.
Frame checks preserve unrelated state, and billing checks require exactly the
permitted allowance change and corresponding charge. Read probes and the expected
error preserve complete state. All actor calls are public decorated tools; seven
local tests cover information-set averaging, cost accounting inputs, public-only
continuations, error propagation, unnecessary purchases and collateral damage.
No outbound connection, model call, rented machine or optimizer update occurred.
The pinned simulator and restricted import arrangement are the same as the
[qualified substrate](telecom-local-v1-results.md); this is not an official
conversational benchmark run.

## What this enables, and what is still missing

Device switches are training candidates, roaming is development-only, and the
entire allowance mechanism is reserved for transfer. All related configurations,
histories, wording and cost variants stay with their mechanism. No model has seen
this new transfer evaluation. Simulator qualification may inspect labels; model
selection may not inspect its future scores.

The next stage is question admission: pair decisions, immediate/continued forecasts
and value-of-observation questions with these exact receipts, preserve fractional
targets, prove that the displayed continuation matches execution, and check token
limits without removing evidence. A categorical distribution loss or balanced
executed outcomes can teach uncertainty; marking both yes and no as acceptable
answers cannot. Prepared questions must remain separate from actual training use.

This is useful data machinery, not evidence of a better mini-Jev. Three authored
mechanisms are too few to justify another learning comparison by themselves.
Realistic time-varying or contradictory evidence and additional task structures
remain open. The completed AppWorld transfer failure remains visible; neither this
collection nor a format change replaces it. Development-only format diagnostics
and broader qualified data should precede more paid training.

[Prospective protocol](telecom-hidden-causes-v1-protocol.md) ·
[Aggregate metrics](../results/telecom-hidden-causes-v1/summary.json) ·
[Collection hashes](../results/telecom-hidden-causes-v1/collection.json)
