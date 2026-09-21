# Later tool choices: bounded inventory before admission

Inspect only witnesses already admitted to the training partition by
release-tool-data-v1. Preserve its request, server and schema ownership. Skip
other witnesses before inspecting their histories; do not create new splits or
open reserved model scores. Use the three pinned cached shards offline.

Ask for the next offered tool after at least one completed tool call. Require
serial calls, exact response linking, declared names and the existing complete
supported-schema checks on every call. Reject parallel or overlapping calls,
ambiguous response identities, additional user turns and unsupported messages.
Keep original system messages as quoted source data, the initial user request,
and earlier tool names, arguments and responses. Omit all assistant prose,
including reasoning and final answers. Never expose the target call's arguments,
response or subsequent messages. Tool-menu order depends only on public input.

These labels imitate the recorded teacher. They do not certify optimality,
successful recovery, correct task completion or calibrated consequences. No
stop/clarify labels or success rewards are inferred from a final answer or judge.
Retain refusals and error responses as observations without treating their shape
as a task-outcome label. Nothing is executed against a real service.

Limit inputs to 4,096 complete tokens and reject public history above 256 KiB;
never crop evidence. Count source trajectories, normalized requests, source
servers, later question presentations, exact-token inputs and conflicting
targets separately. More prefixes are not new independent tasks or families.
Write raw candidates and witnesses only under ignored output/. Admission remains
false until independent reconstruction, collision/ownership checks and conflict
filtering have passed. This stage performs no model inference or training.

Run one CPU process under release_lab.local_guard: 4 GiB process-group RSS,
30-minute wall time, normal host memory pressure and at most 512 MiB additional
system swap. Environment variables disable model backends, tokenizer parallelism
and online downloads. Polling is a coarse CPU-job safeguard, not proof of safe
GPU allocation; all local foundation-model work remains paused.
