# Qualify a coupled account/device substrate locally

Use the already researched Sierra telecom implementation pinned at
`b7ea9074c1cba482b30687fecdb5c8425fd6f619`, in a separate Python 3.12 environment.
Its public code and data carry the MIT license. Do not alter the training runtime,
download model weights, call a model service, or rent hardware for this phase.
The existing AppWorld comparison remains separate and unchanged.

This is an authored substrate qualification, not an official tau benchmark score.
Do not read benchmark task solutions, evaluation traces or published model outputs.
Use the default synthetic application databases and explicitly authored fixture
initializations. Keep all fixture, observation and state receipts private initially.
No fixture becomes a training question in this stage.

The full package initializer imports optional voice components even for a local
tool import. Preserve the failed base-install import smoke test. For this narrow
adapter, load the unchanged telecom/environment modules under their normal package
namespace without executing the unrelated top-level runner/voice exports. Record
this restriction explicitly; it is not the standard benchmark runner. Freeze
every loaded source file. Fix the fixture's UUID stream and random seed before
each fresh execution so newly created billing records can be compared exactly.

Qualify three different dependencies: local device switches, carrier-plus-device
roaming permissions, and a carrier data allowance. Each case starts from a fresh
database copy. Choose a fixture account deterministically from public database
structure before execution; expose its phone number in the authored task and
obtain identifiers through the public customer-lookup tool before using them.
Do not turn initialization-only break functions or hidden IDs into actor options.
The device case starts with airplane mode on and cellular data off. The roaming
case starts abroad with both carrier permission and device roaming off. The
allowance case starts exactly at its data limit; its goal permits exactly one
additional gigabyte at the fixture plan's published per-gigabyte charge. All three
goals require cellular service and preservation of unrelated records. The frame
verifier allows only the declared device/network fields or the targeted carrier
flag/allowance and the exact associated charge; it checks all other fields.

Each case gets a fixed repair sequence, a no-op, a plausible incomplete/wrong
repair, an independent exact replay of the repair, and a collateral-mutation
negative control. Ceiling: three cases, five worlds per case, 15 executions,
20 tool calls per world and 30 seconds per world. Record all attempted executions.
Failed qualification excludes that mechanism; do not silently replace it.

Call the real public tools in solo mode. Preserve the upstream synchronization
between account state and device surroundings after each call. API-level expected
errors are explicit observations; initialization, synchronization and unexpected
runtime exceptions are infrastructure failures and cannot become negative labels.
Call `make_tool_call` followed by `sync_tools` directly, rather than the outer
response wrapper that catches every exception; these qualification plans contain
no deliberately failing API calls. Preserve all exceptions as failed attempts.
Block outgoing sockets and disable dotenv loading before importing the package.
No customer messages, payment processors, live carrier services or model APIs run.

The independent verifier uses explicit stored-state predicates for each authored
goal, separately checks every unrelated database field, and cross-checks a public
cellular-service observation. A generic successful tool response is not a success
label. Wi-Fi must not stand in for cellular service. Read probes must preserve
state; reference and independent replay must match responses and complete states.
Do not equate an upstream aggregate assertion fraction with probability.

The purpose is to establish that these mechanisms are sound enough for a later
data stage. Only after qualification may a separate protocol introduce hidden
causes with identical observations, fixed observation costs, alternative actions
and information-respecting continuations. That stage must define distributions
over compatible worlds, reserve entire mechanisms before question collection,
and report immediate outcomes, continued success and observation value separately.
No prediction supplies ground truth, and no model selection occurs here.
