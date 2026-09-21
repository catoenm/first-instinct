# Three telecom mechanisms pass local execution controls

The pinned Sierra telecom tools now provide a qualified local substrate for
device switches, coupled carrier/device roaming permissions, and a billed data
allowance. This is three authored fixtures, not a benchmark score or a new trained
model. The checks use the real tool and account/device synchronization code from
[the pinned environment](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/domains/telecom/environment.py).

All three correct repairs and their independent replays succeeded. All three
no-ops and three incomplete/wrong repairs failed. Three collateral controls
restored service but changed an unrelated battery field; the separate preservation
check rejected them. Replay compared complete states and public responses exactly.
Read probes preserved state and no outgoing network connection was attempted.

The billing case verifies the exact allowance increase, tariff-based charge,
associated bill update and preservation of other accounts and fields. A public
cellular speed observation independently agrees with the stored-state goal
predicate. Wi-Fi is not substituted for cellular connectivity. These contracts
matter because a successful tool call is not evidence that the user's goal was met.

| Accounting | Count |
| --- | ---: |
| Underlying authored tasks / mechanisms | 3 / 3 |
| Completed control-world executions | 15 |
| Earlier failed adapter attempt | 1 |
| Completed / earlier failed-attempt tool calls | 78 / 3 |
| Exact independent replays | 3 |
| No-op / incomplete / collateral controls | 3 / 3 / 3 |
| New model calls / training questions / optimizer steps | 0 / 0 / 0 |

The first attempt failed while our adapter decoded a plain-text response as JSON,
before it executed a repair. Its source and log are preserved. A separately frozen
correction records public response text exactly and parses only the structured
customer lookup needed for identifier binding. The correction did not change
fixtures or success criteria. See the [correction protocol](telecom-local-v1-correction.md).

The base package initializer also tried to import an optional voice dependency.
This adapter uses unchanged telecom/environment modules under their normal package
namespace while omitting the unrelated top-level runner/voice exports. It is a
restricted substrate integration, not the standard conversational benchmark runner.
Every imported simulator source was frozen; package versions are recorded privately.

These fixtures establish execution and verification, not dataset diversity by
themselves. The next useful collection is compatible hidden causes under identical
visible histories, paired with tool inspections, their costs and public-information
continuations. That can supply genuinely uncertain forecasts and measured value
of information. It needs a separate prospective protocol and mechanism-level
splits before producing training questions. Do not expand rows by merely repeating
these three repair demonstrations.

[Qualification protocol](telecom-local-v1-protocol.md) ·
[Public aggregate and receipt hashes](../results/telecom-local-v1/summary.json)
