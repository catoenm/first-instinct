# Isolated retail actor process qualification

Run a bounded local check of the new actor collector against the pinned real
retail tool implementation. Preserve every earlier freeze. Use the existing
Python 3.12 retail runtime as a separate process; the policy-side process uses
the pinned 9B tokenizer and a tiny scripted CPU policy. No foundation-model
inference, optimizer updates, GPU rental, or supervised-data admission occurs.

There are exactly **18 episodes**: three goals, three controllers, two independent
process replicas. Each episode makes six tool attempts and terminates at the
existing six-turn limit. For each goal, repeat user reads and order reads from an
already-satisfied world; separately repeat a documented refused write from a
processed-order world. Use the same fees, read 2 and attempted write 6, throughout.
This is runtime coverage, not a balanced prior or a policy-performance comparison.

For order-address reads, use the existing `010` world; for profile-address reads,
`100`; for payment reads, `already_migrated`. Refusals use address world `001` and
payment `processed_sufficient`, attempting `write_order` or `migrate` respectively.
Reset each worker from the existing qualified stop witness. No official benchmark
tasks, customers, accounts or external services are involved.

Ceilings: 18 independent resets, 108 real tool calls, 108 scripted actor decisions,
six tool attempts per process, 30 seconds per worker response and 60 seconds per
episode excluding startup. Network calls and official task/database reads remain
denied. Only documented, state-preserving refusals become tool responses; any
other exception fails the stage and preserves partial journals. Do not crop an
actor input above 4,096 tokens; stop and record it as a qualification failure.

Check command/response/state-hash chains, the real six-turn terminal payout,
failed-attempt fees, the independent final verifier, public actor prompts, action
likelihoods, exact replay between replicas, and all imported upstream code hashes.
Report historical underlying worlds separately from fresh reset executions.
These 18 diagnostic paths do not establish every possible future trajectory or
forecast correctness under an arbitrary learned continuation. Before learning,
cross the same costs with all worlds in each training prior and qualify any
additional worker host, live-action patterns or forecast branch interface.
