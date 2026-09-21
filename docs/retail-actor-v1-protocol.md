# Retail actor interface qualification

This is a local integration stage while the frozen balanced supervised pilot
runs. It does not change that pilot, rent hardware, execute new upstream worlds,
or claim a trained retail policy. The original retail source and runtime freezes
remain unchanged.

Adapt the existing step environment to the language policy interface. The actor
receives only its public context, command history, complete offered commands,
costs and remaining horizon. It controls subsequent choices itself. Forecasts
under fixed continuations remain separate questions. The environment's independent
terminal verifier pays 20 for success, zero otherwise, once. Every read or write
attempt, including a refusal or a redundant attempt, incurs its fee. Stop is free.
The model supplies action probabilities and a training value estimate, never labels.

Use all 120 already qualified runtime receipts (60 primary, 60 replays) to check
the public actor format on every saved pre-action observation with the pinned
9B tokenizer and a 4096-token limit. Do not crop, filter, or re-execute these
receipts. Report all overlength observations; any such observation prevents a
claim that this interface fits the existing curriculum. Record source hashes,
unique public inputs, and repeated presentations separately. This checks the
recorded traces only, not every possible six-turn future history.

Use small CPU stub policies and the existing unit-test environment to check
terminal reward accounting, horizon termination, hidden-world equivalence before
observation, likelihood/action mapping, refused attempts, and failure handling.
These are software checks, not model performance or actual upstream tool execution.
No optimizer runs and no generated question is admitted to supervised training.

Before real policy training, qualify the actor against isolated real-tool workers,
including the six-turn cutoff and worst-case prompt lengths. Cross the **same** fee
schedule with every latent world in each declared prior; the old runtime-validation
fee cycle is not a matched learning distribution. Freeze training-only source
ownership, policy-controlled continuation semantics, general-task replay, identical
starting weights, and controlled forecast-only/reward-only/combined comparisons.
Keep telecom's corrected development and reserved transfer ownership unchanged.
Preserve the original cumulative compute limit and require provider deadlines and
artifact recovery before any paid stage. No production or Jev-parity claim follows
from this interface qualification.
