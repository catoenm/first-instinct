# Reservation language decisions, forecasts, and development data

Read the [results](../../docs/reservation-language-results.md),
[decision protocol](../../docs/reservation-language-v1-protocol.md),
[forecast protocol](../../docs/reservation-forecast-v1-protocol.md), and
[development data contract](../../docs/reservation-dynamics-data-v1.md).

- `decision/`: frozen cases/model/source bindings, 254 exact requests and
  responses, executable transitions, 72 completed episodes and controls.
- `forecast/`: frozen public inputs/token IDs, separate gold event probabilities,
  78 attempts/responses covering 39 events, and the original run receipt.
- `dynamics/`: 3,456 development training rows derived from actual qualified
  one-command outcomes. Their 1,152 canonical event questions include 194
  uncertain targets. No heldout test partition or new independent mechanism
  is implied by the row count.
- `analysis.json`: deterministic trajectory/metric reconstruction, exact-token
  checks, matched previous numerical controls, and explicitly marked post-hoc
  diagnostics of public-state mistakes and duplicate presentations.
- `manifest.json`: complete SHA-256 inventory of the other files here.

These artifacts contain no model weights, authentication tokens, host binaries,
or customer data. Private simulator state appears only in audit/gold receipts;
it is excluded from model inputs. The source qualification remains in
`../puffer-reservation-v1/qualification` and is not overwritten.

Both model studies used the unchanged supervised Qwen3.5-9B step-2742 checkpoint
on the existing local demo. No training or paid hardware was launched. The
consequence rows are development supervision, not evidence of model improvement.
