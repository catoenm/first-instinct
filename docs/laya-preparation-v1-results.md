# External baseline preparation: input limits and probability extraction

The shared action format improves full-input coverage for the primary Laya typed
checkpoint from 9 to 23 of 41 existing action presentations. It does not make all
of our tool histories fit. No task-quality scores or new training results were
produced by this work.

## Shared action framing

The original state, decision rules and complete commands move into a JSON state
with short option references. All 41 inputs round-trip exactly. Both compared
models receive the same logical form, and original tool identifiers stay outside
their prompts. This is a framing change, not a semantic summary.

| Checkpoint | Original complete inputs | Shared-frame complete inputs |
| --- | ---: | ---: |
| Laya English, 512 tokens | 0 / 41 | 0 / 41 |
| Laya multilingual, 1,024 tokens | 9 / 41 | 25 / 41 |
| Laya typed decisions, 1,024 tokens | 9 / 41 | 23 / 41 |
| Qwen3.5-9B, 4,096 tokens | 41 / 41 | 41 / 41 |

The shared frame removes instruction and option-description loss in this sample.
Remaining Laya losses occur in the state. Qwen inputs grow from 624–2,277 tokens
to 817–3,210 tokens; moving text into JSON can add escaping and framing overhead.
All 246 Laya formatted presentations (three checkpoints, two forms, 41 inputs)
match the pinned upstream formatter exactly. Four focused tests cover exact text
preservation, dispatch identity, metadata exclusion and malformed references.

The sample contains 40 distinct logical inputs from existing training-owned
preflight trajectories. It adds zero tasks, branches, labels, training consumption
or model predictions. Coverage must be checked again on every history in a later
interactive evaluation. Neither this subset nor its format proves unfamiliar-task
transfer, and we cannot claim general superiority from context coverage alone.

## Unrounded probability extraction

The new adapter captures logits from one native Laya forward and reproduces its
public probabilities exactly after native calibration and rounding. It retains
the unrounded distribution separately, including positive probabilities that the
public interface rounds to zero. It verifies complete input, native token IDs and
markers, evaluation mode, device, and applied temperature, and rejects deviations.

Seven tests passed, including the actual pinned Laya decision head with a tiny
random encoder. A separate qualification passed 15 synthetic calls across three
shipped configurations and five menu sizes. No pretrained weights were loaded.
These software checks do not measure model quality or establish GPU numerical
parity. Instrumentation adds overhead, so its timing is not a native latency claim.

The upstream runtime clamps the English and typed checkpoints' shipped temperature
near 0.1006 for menus of eleven or more options to 0.5. Future metrics must identify
both checkpoint and runtime revisions and the applied temperature. Choice
probabilities, entropy confidence and the separate act-head output stay distinct.

## Remaining work before scoring

The first native typed-decisions forecast comparison can use the 308 already
exposed development forecasts that fit without shortening. It remains a development
baseline. The reserved tool/telecom evaluation is unopened and remains governed by
its existing protocol. An interactive comparison needs a separately fixed cohort
and explicit treatment of histories that exceed native limits.

Next qualify revision-pinned checkpoint loading, actual inference on the target
device, scoring, and a separate latency procedure. Only after the running learning
experiment is recovered and its costs reconciled can a bounded scoring rental
begin within the original budget. The completed tokenizer and tiny-model checks
authorize no additional rental themselves.

Protocols: [shared action frame](shared-action-frame-v1.md),
[native probability runtime](laya-runtime-v1-protocol.md).
Aggregates: [input coverage](../results/shared-action-frame-v1/summary.json),
[synthetic runtime qualification](../results/laya-runtime-v1/summary.json).
Private inputs and per-presentation receipts remain outside the public repository.
