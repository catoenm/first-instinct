# Verified examples of inspecting, acting, and stopping

The local identifier-evidence probe passed its predefined checks. The same
offered commands require different decisions depending on current evidence and
inspection cost. This is execution-verified training-data preparation, **not a
model improvement**. No training or paid compute occurred.

The task is to send exactly one simulated message to a named contact. The
contact's phone number is either unknown, present in a current directory query,
or present only in an explicitly historical query. Two equally likely current
contact-to-number assignments make an uninformed send genuinely uncertain.
Every menu offers the same two fully specified send commands, directory lookup,
and stopping. All effects occur in pinned ToolSandbox's local database; no real
message is sent.

| Visible evidence | Lookup cost | Verified best next action | Lookup value over the best plan without new observations |
| --- | ---: | --- | ---: |
| Missing or historical | 0.02 | Look up the current directory | +0.96 |
| Missing or historical | 1.20 | Stop | −0.22 |
| Current | 0.02 | Send to the observed number | −0.02 |
| Current | 1.20 | Send to the observed number | −1.20 |

These values include future costs under the two-decision continuation specified
in the [frozen protocol](identity-evidence-v1-protocol.md). They are not unrestricted
optimal-policy values. Sending costs 0.02; correct delivery earns 1, an incorrect
change earns −1, and stopping unfinished earns 0. Costs already incurred in the
visible prefix are sunk.

Forecasts name their horizon. A directory lookup followed by immediate stopping
leaves the delivery unfinished. The same lookup followed by the specified
continuation completes delivery. An uninformed send succeeds in one hidden world
and sends to the wrong person in the other: its target is a 50/50 outcome
distribution, even though each actual execution has a definite result.

## What was executed and prepared

| Unit | Count |
| --- | ---: |
| Underlying database worlds | 2 |
| World/goal tasks | 4 |
| World/goal/evidence/cost cases | 24 |
| Primary counterfactual branches | 288 |
| Exact repeat executions | 288 |
| Total branch executions | 576 |
| Tool calls, including setup and prefixes | 3,216 |
| Distinct visible contexts | 16 |
| Forecast questions | 128 |
| Forecasts with fractional outcome targets | 32 |
| Next-action questions | 16 |
| Observation-value questions | 16 |
| Questions consumed in training | **0** |

The 288 primary branches cross each case with four offered actions and three
continuation contracts; some contracts execute the same action sequence. They
are not 288 independent tasks. All 160 prepared questions belong to one
training-owned application-identity group. This adds identifier-resolution
coverage within the existing application family, not a new transfer mechanism.

Every primary branch replayed exactly. A separate offline audit checked all
288 primary receipts and 160 question targets. Its independent verifier requires
every non-message namespace—including all device settings—to remain unchanged.
It rejected wrong recipients, duplicates, modified contacts, modified settings,
and completion claimed without a message effect. This strengthens the reused
delivery verifier's frame checks; none of the actual labels changed. No model
prediction supplied ground truth.

Collection took 28.4 seconds with about 88 MiB peak sampled process memory; the
independent audit took 1.1 seconds. Neither increased swap use. The existing
Python network/subprocess interception recorded no execution-phase attempts;
it is not an operating-system sandbox. Frozen source copies, raw worlds,
executions and questions remain local. [Aggregate evidence and hashes](../results/identity-evidence-v1/summary.json)
bind the protocol, execution receipt, and audit.

## What this supports next

The [full-panel tool diagnostic](generalist-training-v1-results.md#tool-choice-diagnostic-after-training)
found a shift toward tools with no required arguments. This probe supplies a
counterbalance: inspection is useful in some visible states, wasteful in others,
and too expensive in others. It does not establish why the trained model shifted.

These targets remain excluded from the [decision-focused comparison](decision-supervision-v1-protocol.md),
which reuses the previously admitted curriculum. Any later admission must keep
related worlds and wording variants together and exclude development examples.
This small cohort alone does not justify another training rental or a claim of
broader capability.
