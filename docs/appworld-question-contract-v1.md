# Questions over executable application continuations

This prepares a **supervised teaching slice**, not a general tool proposer or a
live reinforcement-learning result. Its menus are reference-assisted. The task
program roles were fixed before demonstration capture in
[the intervention protocol](appworld-interventions-v1-protocol.md).

Every question supplies the task, visible history, and an explicit finite
continuation. Each command is either a literal whose arguments are supported by
already visible evidence, or uses a declared reference to an earlier response in
that continuation. The latter is rendered as `result_of_call` and a path of
dictionary keys/list indices. For each executed alternative, independently prove
that these references resolve to the arguments actually executed. A literal from
a later response cannot silently appear in the prompt. Reject a branch if its
response bindings no longer match. This is a fixed supplied program, not a
claim that the selector generated or discovered it.

Allowed public transformations are bounded page indices, public Boolean/status
choices, visible path basenames joined to visibly queried roots (including home
directory aliases), zip suffixes when requested, and clock offsets explicitly
given in minutes. Keep unsupported transformations excluded. Credentials become
opaque per-application session handles; an invalid session stays distinguishable.
Password-retrieval records are omitted. No verifier state, reference correctness
flag, branch name, source program identifier or target enters model inputs.

The paired questions ask:

1. Whether the exact supplied continuation meets all task assertions while
   preserving unrelated applications. This is **continued success**, not success
   of the first command alone.
2. Whether that continuation changes any stored record in a named application,
   measured directly against the separately executed stopping state at the same
   history. This is a state-change predicate, not a desirability judgment.
3. Which of up to four supplied plans maximizes a completion reward of one minus
   an explicitly supplied per-call cost. Three declared costs (0.005, 0.05, 0.2)
   test whether stopping becomes preferable. Calls continue after API error
   responses. Costs are known from the supplied script, not hidden realized costs.

A skipped first call may break a later binding; that script is rejected rather
than assigned the outcome of a different literal script. Check every prior
demonstration command against evidence available when it occurred as well: a
private argument cannot be laundered into public history merely by replaying it.
Same input and same
world must have consistent truth. Group indistinguishable inputs over distinct
worlds before calculating a distribution; duplicate presentations of one world
do not add probability mass. Retain disagreements as soft outcome targets.

This slice does not yet supply information-gathering value under adaptive
replanning, a dynamic proposer, or new stochastic worlds. Preserve the previous
verified uncertain-world corpus for any subsequent training mixture. Do not
inflate diversity by counting costs, question types or repeated epochs as new
worlds or task programs.

Admission requires a passing control audit, independent replay of every retained
alternative, causal argument closure, exact preservation of program splits, and
the pinned Qwen tokenizer with an 8,192-token limit (reject rather than truncate).
The initial 4,096-token sizing check excluded all histories from four training
programs. Before any model inference, the pilot context ceiling was therefore
raised to 8,192. Repeated records use a lossless column representation with an
independently checked decode round trip; every value and record remains present.
This changes context capacity, not the nine-billion-parameter foundation.
Report question and decision coverage separately by program after these filters.
At least eight training programs and both development programs must retain useful
questions. Also report how many retain genuinely different, nonempty execution
plans; terminal-only examples are not a substitute for richer decision coverage.

Freeze the final dataset, mixture, replay pools, model parent, improvement checks,
training implementation and spend limit before any pretrained-model evaluation
or paid rental. This document alone does not declare training ready.
