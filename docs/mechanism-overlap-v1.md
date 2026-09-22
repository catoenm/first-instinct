# Which proposed environments add a new decision mechanism?

A source review of the existing executors narrows the
[mechanism expansion proposal](mechanism-expansion-v1.md). This review executes
no new worlds and produces no training data. It is not a comprehensive semantic
audit of every public imitation example.

| Proposal | Existing coverage | Decision for the next stage |
| --- | --- | --- |
| Filesystem aliases and replacement | `tool_lab/filesystem_decisions.py` already executes hard links, in-place writes and atomic path replacement. Its goals distinguish changing one path from changing a shared physical object. | Symlinks alone are an extension of exposed aliasing. Do not call a new backend or path type an unfamiliar mechanism. |
| Lost acknowledgement and retry | `general_lab/retry_environment.py` already separates delivery, committed effects and acknowledgements, with delayed receipts and unique idempotency keys. ToolSandbox delivery also verifies duplicate prevention. | A basic durable queue repeats much of this mechanism. Do not reserve it as clean transfer merely because its objects are named jobs. |
| Database transactions | `puffer_lab/sql_oracle.py` and `tool_lab/expanded_runtime.py` execute partial versus atomic reservation writes, rollback and recovery. Calendar actions also test transaction safety. | Another atomic-versus-sequential transaction is familiar. |
| Stale application observations | `tool_lab/retail_evidence.py` executes historical observations and intervening state changes; report/application families also contain historical evidence. | Staleness alone is familiar. It is insufficient to justify new ownership. |
| Conditional writes during competing updates | The reviewed curricula do not implement a second writer invalidating a version-checked update, followed by a conflict refusal and refresh/retry. | Qualify this narrower missing behavior with two SQLite connections and a deterministic writer schedule. Treat it initially as a training candidate, with no broad transfer claim. |

For the next local prototype, distinguish an increment relative to the latest
state from an update approved only for a particular revision. Keep the command
menu identical across these goals. Execute unconditional replacement, an atomic
field update and a version-checked update in separate database copies. A command
may succeed while overwriting a colleague's work; a conditional write may refuse
while preserving a valid state. Read final rows through a separate connection
and check protected fields independently of the command's return code.

The visible cached observation should be identical in worlds with and without
an intervening writer. Declare the conditional prior and preserve fractional
outcome labels. Forecasts must distinguish command-then-stop from an explicit
refresh-and-retry continuation. Vary observation and mutation costs so that
refreshing, proceeding and stopping each have cases where they are useful.
Failed attempts must still incur their published costs.

Count the initial database, competing-writer schedules, goal variants, branch
executions and questions separately. Do not turn one fixture into hundreds of
claimed independent tasks. Preserve existing mechanism ownership and the locked
release pack. A genuinely unfamiliar transfer mechanism still needs separate
qualification; this review does not provide one.
