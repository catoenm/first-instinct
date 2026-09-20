# Live execution adapters: local qualification before learning

Reuse the qualified filesystem/calendar executors and the existing reservation
`SqlEpisode` SQLite implementation. Do not modify frozen source collections or
rerun them to produce more labels. Add live adapters that accept one offered
action at a time, expose only the resulting public observation, and let the
actor choose again. Forecasts keep their named continuation; live actor prompts
promise no fixed continuation. Calendar remains reserved for evaluation and no
model scores are opened during this qualification.

Filesystem and calendar episodes retain their four-decision horizon and end
after a committed mutation, stopping, or horizon exhaustion. An unchanged failed
calendar transaction is recoverable. Reservation episodes follow their existing
three/six-turn profiles: partial writes can be repaired before stopping or
horizon exhaustion. Use actual SQLite for each reservation trajectory, not the C
mirror. Verify exact relational state, schema, integrity and unrelated rows from
a separate set of predicates. Preserve the original units: 400 quarter-credits
of terminal reward become +1; each command's quarter-credit fee is divided by
400. State the same normalized units in actor inputs. Initialization-prefix
costs are sunk and excluded from the new trajectory's future return. Terminal
utility is paid exactly once, including when a prefix already satisfies the goal.

Before execution, freeze a deterministic local plan of at most 400 distinct
trajectories. Use the existing 40 filesystem cases, 80 calendar cases, and the
existing reservation profiles/worlds under empty, inspected and failed-attempt
prefixes. For each, run a public-observation reference controller and a seeded
random in-catalog action schedule. The random controller cannot inspect the
private world or verifier. Replay every selected trajectory in a fresh world,
requiring identical public inputs, observations, actions, states, costs,
termination and return. At most 800 native trajectories and 96 selected Linux
trajectories are allowed. Cap each platform at 9,000 tool actions and 15 minutes;
record attempts before initialization and preserve incomplete/rejected attempts.
SQL query/statement counts are reported separately from top-level tool actions.

Use a pinned network-disabled Linux image for representative parity across all
three mechanisms, worlds, evidence regimes and both controller modes. The
portable contract is actual logical state, public observations and protected
bytes, not database page layout or raw inode numbers. Match every selected
receipt against native execution. Runtime adapters must support fresh policies'
arbitrary in-catalog histories, not just replay the named forecast continuation.
The integration controls have a separate limit of 64 native episodes and 512
offered actions; their source hashes are frozen before running the test suite.

Integration controls must cover wrong mutations, recoverable failures, redundant
queries, horizon exhaustion, stopping, already-completed state, partial-write
repair, sunk fees, normalized reward, refusal to act after termination, illegal
actions and tampered receipts. Test private-state exclusion and identical
public observations across compatible worlds. Tokenize complete live actor
histories with the pinned 9B tokenizer, without truncation. These checks evaluate
execution mechanics only; they train no language model and provide no evidence
of decision improvement.

Only after runtime qualification may the broader learning comparison be frozen.
Keep original supervised step2742 as the parent, preserve general replay, and
retain whole-mechanism ownership from the qualified source registry. Further
qualified training is authorized within the original remaining budget; this
protocol creates no rental or fresh spending allowance.
