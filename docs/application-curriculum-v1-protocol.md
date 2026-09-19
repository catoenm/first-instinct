# Bounded application workflow qualification

This is a new, local training-family candidate for decision curriculum v3. It
does not reopen the previously inspected ToolSandbox reminder cohort or change
the completed evidence-decisions-v2 experiment. No model calls, paid machine,
training, real messages, or external application services are involved.

Use the pinned installed ToolSandbox commit
`c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`, its verified source hashes and
dependency pins, and the existing Python socket/subprocess interception guard.
That guard is not an operating-system sandbox. Only a fixed allowlist of
upstream local messaging, contact and device-setting functions can execute.
The upstream messaging function inserts a simulated database row; it does not
send an actual text message. Deterministically patch only UUID generation and
the clock. Record those patches as adapter behavior.

## Mechanism and visible state

One root fixture contains a self contact and two recipients. Cross three initial
message ledgers (neither recipient, first recipient, second recipient) with three
connectivity states (ready, cellular disabled, low-battery mode). Each concrete
world is paired with both recipient goals. A goal requires exactly one matching
message to its recipient, preserving all old messages, other contacts, reminders,
and unrelated settings. Low-battery and cellular settings may change; Wi-Fi,
location and device identity must be preserved throughout execution.

The same menu offers message lookup, two setting queries, disabling low-battery
mode, enabling cellular service, enabling low-battery mode, either message send,
and stopping. Command spelling and option IDs do not change with hidden truth.
ToolSandbox provides the real dependency: low-battery mode blocks enabling
cellular service, and disabling low-battery mode does not re-enable it. Repeating
a setting change can fail. Enabling low-battery mode can damage protected settings.
Duplicate or wrong-recipient sends are irreversible mistakes within an episode.

Initial contexts are missing evidence, current observations, expensive queries,
an actually failed send, and an explicitly historical message lookup made in a
separate old world. The historical observation is independent of current truth.
Failed-send contexts apply only to disconnected worlds. A public equal prior
covers the three initial ledgers and three connectivity states; observations
condition this finite cohort. It is an authored prior, not a deployment estimate.

Reads cost 0.02, or 1.2 in the expensive regime; mutations cost 0.02; stopping
is free. Maximum seven future decisions. Verified completion earns 1, an
irreversible mistake earns -1, and unfinished earns zero, less future costs.
Already-visible prefix costs are sunk and counted as execution work separately.
Return codes are observations, never the sole reward criterion.

## Counterfactuals and questions

Reinitialize each alternative from the same concrete world and replay its visible
prefix. Execute each offered action under three contracts: immediate termination,
a fixed public evidence-and-completion continuation, and that continuation with
new observations forbidden. The latter two stop at the same seven-decision limit.

The evidence continuation first stops if current evidence proves completion.
Otherwise it obtains the message ledger unless unaffordable, checks service and
low-battery status as needed, disables low-battery mode before enabling service,
and sends the exact requested message. It uses only the visible request/history.
Without new observations it stops when the ledger is unknown; with known missing
delivery it attempts the remaining work using available observations and errors.

Produce per-world categorical forecasts (completed, incorrect, unfinished) for
immediate and evidence-continuation contracts. From each identical visible context,
derive a best-next-action set from mean executed return under that continuation.
For observation value, compare message lookup plus the evidence continuation
against the best non-observation action plus the no-new-observation continuation.
All future costs count. These values are relative to specified continuations,
not unrestricted optimal policies. Preserve contradictory labels when hidden
worlds share identical public inputs. No model supplies any target.

## Bounds and gates

One new root, nine base database worlds, two goals, five context regimes yield
84 concrete world/goal/context cases. Execute nine actions under three contracts,
with a second exact replay for each: at most 4,536 production/replay branches.
Cap all setup, prefix and branch tool calls at 80,000. Prospective registry and
source hashes precede execution. No automatic expansion if a gate fails.

Require deterministic replay, independent database/frame verification, saved-row
read-back, exact reward/cost arithmetic, complete counterfactual menus, legitimate
hidden uncertainty, costly-query reversals, stopping on already-complete goals,
recovery after failed sends, and prerequisite-chain coverage. Negative controls
must reject duplicated messages, wrong recipients and protected-setting changes.
Report menu/root/world/context/branch/call/question counts separately. Entire
new application mechanism is a training candidate; CSV-report validation and
atomic-publication transfer families stay reserved in the shell curriculum.

This stage qualifies one additional mechanism, not thousands of independent
tasks. Broader data preparation, token checks, live-policy collection and a frozen
9B comparison remain separate gates before any paid training.

## Local qualification revision before training

The first collection executed all 4,536 branches but failed its final isolation
accounting gate. A traced import probe identified urllib3's IPv6 capability check:
it attempts to construct a socket during dependency import, which the guard
blocks. The collector had compared final counts against its pre-import probes.
The original failed collection and exact source are preserved, with 34,264
top-level calls and zero training consumption. It is excluded from training.

One bounded corrected replay is authorized by this revision, in a separate
output directory: at most another 4,536 executions and 80,000 calls. Record
blocked dependency-import probes separately; the socket stays blocked. Require
zero additional socket/subprocess attempts during task execution and persist
the guard and counters throughout collection. Use a separate deterministic UUID
scope for the historical world so it cannot change current fixture identities.
Both collections' work remains separately accounted; they do not count as
additional distinct training tasks.
