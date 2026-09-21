# History, replay and verified forecasts: bounded pilot

This pilot tests a revised supervised recipe on the original Qwen3.5-9B
step-2,742 adapter. The preceding release pilots improved forecast scores but
did not improve tool selection enough to qualify. The original model remains
the release checkpoint while this experiment runs.

The new recipe adds later-tool histories, lowers the learning rate, preserves
general-task practice and penalizes departure from the original model's
distributions on general **training** questions. A transaction guard rejects
large individual updates. This is a practical recipe revision, not a causal
ablation or new online reinforcement-learning result. Reference predictions are
an auxiliary preservation target; they never label tool outcomes.

The maximum frozen schedule contains 320 updates of 64 questions:

| Training pool | Unique questions | Maximum presentations |
| --- | ---: | ---: |
| General-task replay | 10,240 | 10,240 |
| First-tool choices | 5,120 | 5,120 |
| Later-tool choices | 2,560 | 2,560 |
| Verified decisions and forecasts | 2,003 | 2,560 |
| **Total** | **19,923** | **20,480** |

This totals 15,636,387 input-token presentations. The 2,560 selected later-tool
questions cover 1,892 normalized requests. The verified pool includes 190 retail
history forecasts already supported by independently checked command executions.
Those forecasts concern executing the specified command and immediately stopping.
Other forecast tasks retain their own explicit continuation contracts. Questions,
requests, physical worlds, executed branches and repeated presentations are
different units; the row count does not imply 19,923 independent environments.

Preparation and the independent source audit found no exact training/development
token overlap. All selected labels match their admitted sources. The exact input
archive passed 11 tests after extraction. No model was loaded locally. The
prepared corpus is larger than this bounded schedule; actual consumption must
come from the run ledger, not this table.

The evaluation cohort has 6,284 questions: the unchanged original 6,072 plus 212
later-tool development questions. Every 80 accepted updates, the pilot checks
tool improvement, general-task retention, consequence probability quality and
later-tool retention. Two checks without eligible improvement or a retention
failure stop training. Passing the pilot does not establish unfamiliar-mechanism
transfer; reserved transfer data remains unopened.

One H200 rental was started at $4.59/hour under a three-hour provider-side stop
plan. The stage reserves at most $25 from the same original $500 authorization.
GPU forward/backward checks and a separate-process restart must qualify before
sustained training. Artifacts must be recovered and hash-verified before the pod
is deleted. Training status and final results are reported separately from this
launch record.

Parent adapter SHA-256:
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.
Data freeze SHA-256:
`079ed7ad7f0c537e2c62bdf0e3ed4e41df60df7466e5193225cfa4b80f66787d`.
Input archive SHA-256:
`27870cc1f4d84735df351f8006cda295ba13d26a03aa7c6350547c1fc4906328`.

See the [frozen protocol](history-pilot-v1-protocol.md) for exact bounds and gates.
