# Matched retail execution qualified, with no model training

The local runtime now qualifies every combination of the existing retail hidden
conditions and six cost pairs. Both immediate stopping and inspecting before an
appropriate write ran in the actual pinned retail implementation. The independent
audit passed: commands, responses, full terminal states and verifier outcomes
matched the previously executed procedures. Fees did not change these deliberately
cost-blind controllers' paths, and every charge matched an actual attempt.

| Quantity | Count |
| --- | ---: |
| Existing goal contracts | 3 |
| Existing physical initial states | 10 |
| Distinct goal/condition/cost cells | 120 |
| Primary reset executions, two controllers per cell | 240 |
| Additional independent replays | 12 |
| Completed reset executions | 252 |
| Actual tool calls | 200 |
| Actor turns, including stop | 452 |
| Distinct public actor inputs | 114 |
| New independent tasks / physical worlds | 0 / 0 |
| Foundation-model calls / optimizer steps | 0 / 0 |

The three goals cover two existing mechanisms in one connected training ownership
group. Repeated resets and alternative costs do not create independent tasks.
All twelve new replay paths matched exactly; other cells received one new primary
execution per controller in this stage. No scripted action became an optimal-action
training label. No new supervised questions were admitted or consumed.

## Costs change which diagnostic controller is better

Expected returns use the declared uniform hidden-world prior for each goal. They
include the 20-unit independently verified terminal reward and every tool fee.
Replays contribute no extra probability mass.

| Goal | Stop | Inspect: read 0.10, write 0.25 | Inspect: read 10, write 6 |
| --- | ---: | ---: | ---: |
| Order shipping address | 10.0000 | 14.8375 | 3.5000 |
| Customer profile address | 10.0000 | 19.7750 | 7.0000 |
| Payment migration | 5.0000 | 9.7375 | −11.5000 |

Inspecting beats stopping at the low costs; stopping beats inspecting at the high
costs. This compares two diagnostic controllers, not all possible policies. The
actor is never given the private condition identifier, verifier state or these
expected returns. Within every goal/cost/controller cell, its initial public
input is identical across compatible worlds. Thirty such cells have different
verified outcomes despite identical initial observations, preserving legitimate
uncertainty rather than discarding it.

## What is prepared for a future comparison

Each of two seeds has a deterministic shuffled block of 144 reset presentations:
48 per goal and eight per goal/cost cell. The address goals have eight conditions;
payment has four, so each payment condition appears twice. This maintains equal
goal weight and each goal's uniform conditional world prior. There are 120 unique
reset cells per block; repeated payment resets are recorded as repeats. All future
arms must share the same ordered block within a seed, then collect their own
fresh on-policy trajectories. No model has consumed these blocks.

Every recorded actor input was re-encoded without cropping. The maximum was 2,262
tokens under the 4,096-token limit. This proves these paths fit; it does not cover
all future learned histories. The execution used at most about 0.66 GiB resident
memory under the local guard, with no additional system swap. The Mac never loaded
the foundation model. Nineteen relevant CPU tests passed across the schedule,
public actor boundary, terminal reward checks and environment-path correction.

The first worker startup failed before any tool call because its executable path
selected base Python, which lacked `toml`. That attempt and freeze remain intact.
A separate manifest restored the existing virtual-environment entry-point path,
bound its runtime identity and package versions, and changed no worlds, fees,
labels, schedules or ceilings. No dependency installation was needed. Count that
one failed startup separately from the 252 completed executions.

Corrected execution freeze:
`082f107e87791faf9dfe47b87ed9164ee0129cd3daa36bc188b1e8bff4360b3c`.

The next gate is [integrating the learning contracts](live-learning-next.md):
explicit action exploration, correctly scaled reward/value units, detached critic
gradients, exact forecast horizons, general replay and independent mechanism
evaluation. This qualification launches no GPU run and supplies no evidence of
learned performance or transfer. The independent release scores remain unopened.

[Protocol](retail-matched-v1-protocol.md) ·
[Independent aggregate audit](../results/retail-matched-v1/summary.json) ·
[Execution counts](../results/retail-matched-v1/collection.json) ·
[Runtime qualification](../results/retail-matched-v1/qualification.json) ·
[Preserved import failure and correction](retail-matched-v1-runtime-correction.md)
