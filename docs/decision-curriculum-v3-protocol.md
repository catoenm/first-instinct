# Executed decision curriculum, next stage

This is a local curriculum qualification following `evidence-decisions-v2`,
not a replacement run. The active v2 dataset, source, six learning arms, stop
guards and recovery process remain frozen. No v3 training or additional rental
is launched by this protocol. The remaining original $500 authorization is
the only compute budget; storage and the current rental must be reconciled
before another allocation.

## Local execution and questions

Reuse the v2 isolated Docker executor and the independently checked configuration,
invoice and report fixtures. Add redundant current observations, mutually
contradictory historical reports independent of current truth, a disconnected
inspection service, an actual failed command, a required preparation step,
different information prices, and stopping. Irreversible completed mutations
retain the wrong-action penalty. The same commands appear across opposite goals
and hidden states. Infrastructure failures stop collection, not become labels.

Add one different mechanism: artifact publication. Actual files are staged,
their content hash is validated, and an atomic replacement publishes them.
Changing the staged artifact invalidates its previous validation; attempting
publication without a matching validation fails and permits recovery. A wrong
completed publication is irreversible within the episode. The verifier compares
actual output bytes and every protected input outside the worker.

Use a public equal prior over two concrete initial worlds. For each candidate
action, reset a separate disposable directory to exactly the same initial
files and replay the visible prefix. Execute that action under two contracts:

* `stop_now`: execute only this action, then inspect the final files.
* `evidence_then_commit`: use a published deterministic policy that reads only
  visible history, reconnects a failed inspection service, obtains current
  evidence, satisfies prerequisites and completes the requested change. It
  never exceeds six remaining decisions. It abandons an unobserved task when
  the inspection price alone exceeds the maximum success reward.

Consequences have three mutually exclusive categories: verified completion,
an incorrect irreversible mutation, and unfinished. Costs include the offered
action and subsequent actions, exclude the already-visible prefix, and are
metered from actual command events. A command exiting zero is not verification.

Group compatible worlds by the complete public input within one root and
regime. A next-action label maximizes mean executed return under the specified
continuation, **not** a hidden-world optimum or an unrestricted optimal policy.
Retain all tied best actions. For an observation-value question, compare its
executed continuation value with the best candidate followed by a continuation
forbidden to take further observations. Costs are included on both sides.
The label means worth taking under these explicit plans, not universal value
of information. Preserve per-world categorical labels even when identical
forecast inputs have different outcomes; exact aggregate distributions stay
in verifier records for probability evaluation.

The model never labels itself. Every answer points to executed receipts.
State, question and option descriptions are the only model input fields.
Goal/world variants, action alternatives and wording variants share root IDs.
Do not shuffle rows into train/test independently.

## Ownership and local gates

Configuration and SQLite repair are training families. CSV reporting is a whole
validation family. Atomic artifact publication is a whole transfer family.
These are unseen in this proposed post-training mixture, not claimed unseen in
the foundation's pretraining or all earlier project experiments. All generated
variants within a root stay together. Test forecasts must not tune training.

Before training: independently reconstruct terminal checks, rewards and public
histories; require opposing outcomes at identical hidden-state inputs; current
evidence must resolve ambiguity; duplicates must not justify another expensive
query; failed commands must leave recoverable states; mutation/protected-file
negative controls must fail; cost changes must reverse observation choices;
all declared alternative menus must contain a useful action, including stop.
The implementation must retain action-order changes without changing semantic
labels. Publish qualified counts and any failed gates. No row multiplication
counts as new mechanisms.

## Reuse of external environments

The existing ToolSandbox partial cohort already has actual database execution,
cross-tool lookup, incomplete queries, cost-sensitive stopping and independent
final-state audits. Preserve its published 48 roots and 720 model questions as
a diagnostic; they have already been examined and are not a pristine new test.
Do not relabel their `train` collection split as permission to train on them.
Do not casually repeat its finite-budget integrations. A new application
training family must use new semantics and an explicit separate collection
protocol. [ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox)
provides stateful APIs and dependency-sensitive evaluation; prefer extending
our installed adapter over writing another mock application engine.

[Harbor](https://www.harborframework.com/docs/tasks) supplies isolated task
execution and verifier hooks. The fixed-catalog local qualification uses our
existing Docker backend, not Harbor. Move arbitrary proposed shell commands
through the pinned Harbor integration only after separately qualifying it.
[Tau2-bench](https://github.com/sierra-research/tau2-bench) is another candidate
for application workflows, but its user simulator and policy evaluation add
moving parts; it is not installed or represented as executed by this stage.

For a dynamic proposer retain exact public history, full menu and commands,
proposer identity, and independently executed menu coverage. Report absence
of a useful proposal separately from failure to select one. A fixed authored
catalog is not evidence about proposer quality.

## Controlled learning gate, after v2 completes

Use the identical original Qwen3.5-9B step-2742 language adapter for forecast-only,
reward-only Proximal Policy Optimization and combined arms, paired seeds, and
the same replay schedule. Do not initialize one arm from a v2 winner. Decision
and observation-value supervised rows are initially **diagnostic only** so the
forecast-only intervention stays interpretable. General replay remains in all
arms. Critic gradients remain detached; prove nonzero pure policy gradient and
old-policy probability consistency before the first update.

Before renting, freeze exact dataset/checkpoint hashes, token audit, rollout
and optimizer limits, allocation and recovery guards in a separate launch
manifest. Compare validation expected return, consequence Brier/log loss,
and general accuracy/log loss. Retain v2 safety thresholds (accuracy loss at
most .02; general log-loss rise at most .05; full policy divergence at most
.02) and stop after two failed validation improvements once the minimum pilot
length is reached. This document alone is not a runnable launch manifest.

Advance a candidate only if both seeds improve held-out-family utility by at
least .03 and lower consequence Brier by at least .02 against identical starts,
without violating general retention limits. Report all arms and paired-root
uncertainty, including failures. If only authored-family accuracy rises, revise
the curriculum; do not scale the run. Test is opened once after selection;
failure there requires a newly reserved transfer mechanism for a new study.

Count separately: mechanism families; root fixtures; concrete hidden worlds;
goal/cost/prefix variants; executed action-continuation branches; actual commands;
forecast/decision/value questions; optimizer presentations and unique consumed
IDs. Prepared rows are not trained rows. Keep checkpoint parent hashes,
selected update, optimizer steps and consumed-row ledgers explicit.
