# Broader executable decisions after the baseline

The recovered real-tool pilot has five training mechanisms plus retail workflows,
but its decision and forecast development both concentrate on one report workflow.
More rows from those templates cannot establish general decision capability.
The next data milestone should be a small, independently verified collection with
different reasons that apparently plausible actions succeed or fail.

This is a proposed curriculum, not generated data, a frozen training protocol, or
authorization for an additional rental. The active Laya/Qwen comparison remains
unchanged. Qualification can use bounded local CPU execution.

| Proposed family | Missing causal feature | Executable substrate | Independent goal check |
| --- | --- | --- | --- |
| Revisioned database edits | Another writer can make a previously correct observation stale; a conditional update can safely refuse instead of overwriting | SQLite, two real connections, deterministic interleaving of transactions | Compare the final database with the requested update and preservation requirements; inspect transaction results independently |
| Indirect filesystem targets | A path and the object it refers to can differ; a symlink or changed target can invalidate a seemingly safe operation | Real temporary directories, symlinks, file descriptors and atomic replacement | Check object identities, contents and protected files through an independent directory scan |
| Recoverable work queues | Delivery, durable effect and acknowledgement are separate events; retrying or acknowledging at the wrong point can lose or duplicate work | A small executable durable queue and worker, with a recorded deterministic event schedule | Count durable effects and acknowledgements independently; verify exactly the requested effects and no unintended ones |

Existing ToolSandbox supplies genuine application dependencies and remains in the
curriculum. Its current messaging slice does not by itself supply database
interleavings, filesystem aliases or durable queue acknowledgement semantics.
AppWorld's larger application traces remain useful for broader imitation, with
their existing licenses and ownership. A new runtime is justified only for the
missing mechanism; it should not recreate an application framework already present.

## Start with mechanism tests, then questions

For each proposed family, first write a reference state transition and a separate
final-state verifier. Construct a handful of minimal worlds demonstrating an
actual action reversal: the same command is useful under one observed goal/state
and harmful or wasteful under another. Use real subprocesses or separate database
connections where the mechanism depends on execution. Schedule competing writes
explicitly rather than relying on nondeterministic timing races.

Prove that incorrect operations can fail visibly, refuse safely, or succeed at
the command level while violating the user's goal. Include a recoverable failure,
a required prerequisite, redundant inspection, and a correct stop decision. Charge
attempts according to a declared cost model, including failures. A changed cost
should sometimes change the best action; otherwise the cost variation adds little.

Keep the option descriptions factual. Vary option order and identifiers, and test
opposing goals using identical command wording. The verifier must not read the
model's answer or an authored "correct option" annotation. A menu with no useful
action is a menu failure, not evidence that the selector failed to choose one.

## Pair decisions, forecasts and observation value

At a visible history, clone every compatible world with its declared prior weight.
Replay the same history and reject incompatible observations before considering
the candidate action. Execute each alternative in its own copy. Repeated executions
check reliability; they do not add probability mass or independent task count.

Create separate questions for the next action, the consequence of a specified
command followed by stopping, and the consequence under a displayed fixed
continuation. Do not mix these forecast horizons. Compute the value of another
observation using expected verified utility over its possible results, including
the subsequent fixed decision rule and inspection cost. Preserve uncertain labels
when identical observations are compatible with different terminal outcomes.

Do not count mechanically generated wording or cost variations as new mechanisms.
Record underlying goals, physical worlds, interleaving/event schedules, compatible
world cells, executed branches, distinct questions and later optimizer
presentations separately.

## Preserve a useful unfamiliar-mechanism test

Propose the database and filesystem families for training qualification, with the
queue family reserved for transfer qualification. Before collecting anything,
check whether this mechanism split survives comparison with all previously exposed
data. If queue examples merely repeat existing payment retry/idempotency dynamics,
they must be grouped with those mechanisms and cannot be called fresh transfer.
Similarly, new symlink examples must be distinguished from existing path-scope
variants by their actual aliasing behavior, not a new family name.

Keep related worlds, goals, trajectories and wording variants together. Freeze
mechanism ownership before model predictions; reserve more than one independent
mechanism before making broad transfer claims. Exposed report, calendar and payment
diagnostics stay exposed, and the existing locked tool/telecom pack stays locked.

Only after local execution, clone/replay, label and context audits pass should this
become a training proposal. Use the external baseline results to choose a practical
foundation, then prospectively define matched starting weights, exposure schedules,
general replay, decision/probability checks and bounded pilots. There is no reason
to rent four large GPUs merely because the previous run ended.
