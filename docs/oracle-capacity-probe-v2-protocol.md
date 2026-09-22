# Correct the capacity preflight's actor probe

Preserve oracle-capacity-v1, its failed archive and its zero-update result. Its
longest collected action happened to have zero earned advantage, making the
pure actor loss identically zero. Requiring a nonzero gradient on that particular
loss was an invalid connectivity test.

The only proposed runtime change is a new preflight entry point,
`tool_lab.oracle_capacity_qualify_v2`, using `diagnostic_records` from
`tool_lab.oracle_capacity_probe`. Select the longest actually collected
current-policy record whose earned advantage is nonzero, with question ID as a
deterministic tie-break. Persist the rule, selected ID, advantage, token length,
collection size and number of zero-advantage records before backward evaluation.
Fail explicitly if there is no informative record. Do not manufacture a reward,
change a critic estimate, relax any gradient requirement, or remove zero-advantage
transitions from optimizer training.

The failed original entry point remains immutable. The same original 9B adapter,
data freeze, three-arm recipe, consequence losses, replay, reward contracts,
stopping rules, evaluation questions and release restrictions remain in force.
All earlier current-policy likelihood and divergence checks remain unchanged.
Actual-device actor, critic, new decision and continuation gradients must still
pass before optimization; local tests are not evidence of GPU qualification.

Local tests must reproduce the original failure with a real database episode
ending in a zero-return stop, show nonzero actor gradients on an informative
record from the same trajectory, show the critic remains detached, and establish
that the selector neither mutates nor filters the training collection. All-zero
or nonfinite inputs must fail explicitly. The completed control predictions and
failure receipts must be independently audited before staging a correction.

This protocol allocates no hardware or new funds. A continuation needs a
separately verified input bundle and an explicit bounded allocation from the
remaining original authorization. Do not automatically recreate the deleted pod
or extend the closed rental. Preserve and report correction qualification,
optimizer work and previously completed measurements separately.
