# Saved-forecast selection diagnostic

See [results](../../docs/forecast-selector-v1-results.md) and the [prospective local protocol](../../docs/forecast-selector-v1-protocol.md).

This reuses already audited report-development forecasts and execution branches.
There are no new model calls, task executions, optimizer steps, or training rows.
The `private` context file name marks verifier-only fields: these must never enter
the selector input. The selection function accepts only public state/menu plus
predicted probabilities. Oracle references deliberately use verified distributions
and are labelled separately.

The freeze identifies every source, predictor and already-selected checkpoint.
Source branch hashes refer back to the original shell-curriculum receipts;
forecast predictions are preserved in expanded-decisions-v1's public archive.
Full decisions and per-regime measurements are included, without weights.
