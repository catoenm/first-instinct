# When is identifier discovery worth doing?

This bounded local probe tests a prospective training-data correction after the
generalist tool-choice diagnostic. It does not change development labels or the
completed run's selection rules. Use authored fixtures in the already pinned
ToolSandbox application runtime, with no benchmark tasks, models, external
services, or real messages. The send function only inserts a simulated row.

The existing application curriculum gives the requested recipient's phone number
in the goal. This probe instead names a contact. Two possible contact-to-phone
assignments have equal prior probability. The model-visible menu contains the
same two fully specified send commands in both worlds, a contact lookup, and
stopping. The correct command must depend on observations, not menu construction
from the hidden recipient identity. Connectivity is ready and no message has
already been sent; those facts are public and held constant.

Cross two assignments, two named recipient goals, three evidence conditions
(missing, current, explicitly historical), and two lookup prices (0.02 and 1.20):
24 world/goal/context cases from two underlying database worlds and four
world/goal tasks. Current evidence is an actual query of the current world;
historical evidence is a query of a separately initialized old world independent
of the current assignment. Sending costs 0.02; stopping is free. All variants
belong to one training-owned application-identity group. This is added prerequisite
coverage within the application family, not an independent transfer mechanism.

Execute every menu action in a fresh world under three contracts:

1. Execute that action and stop immediately.
2. Execute it, then follow a visible-history policy for at most two total
   decisions. With current contact evidence, send to the named contact's recorded
   number. Without it, look up the contact only when lookup costs less than 1;
   otherwise stop. Any send or irreversible mistake ends the episode.
3. Follow the same continuation while forbidding additional observations.

Verify exactly one correctly addressed message, preservation of every existing
row and unrelated setting, and no duplicate delivery using complete before/after
databases. A returned success code is not the outcome verifier. Reward is 1 for
verified completion, -1 for an incorrect change, or 0 for unfinished work, minus
all future costs. Prefix costs are sunk and reported separately.

Group identical visible inputs before averaging outcomes across compatible
worlds. Produce immediate and specified-continuation categorical forecasts,
acceptable next-action sets under the stated continuation, and lookup-value
questions comparing lookup plus continuation with the best non-query action
without further observation. Count a tie as not worth inspecting. No prediction
provides a label. Dynamic proposal quality is outside this fixed-menu probe.

Freeze code, fixtures, this protocol, and pinned upstream hashes before execution.
Permit exactly 288 primary branches plus 288 exact replays, at most 6,000 tool
calls and five minutes under the existing local memory guard. Require complete
menus, deterministic replay, identical initial visible inputs across hidden
assignments when evidence is absent/historical, and zero execution-phase network
or subprocess attempts. Verifier controls must reject wrong recipients,
duplicates, changed protected contacts, and a false success return without a
database effect. Replay observations must match independently reconstructed query
results.

The acceptance checks are: cheap missing-evidence lookup beats guessing; expensive
lookup makes stopping best; current evidence makes direct action better than a
redundant lookup; historical evidence cannot resolve current uncertainty; and
direct-action forecasts remain fractional under indistinguishable worlds. A
failure quarantines the collection. Passing qualifies local execution semantics
only. Count generated questions separately from training consumption, which
remains zero. Broader mechanism coverage, admission into the training pipeline,
and a prospective model comparison are separate work; this probe alone does not
justify another GPU rental.
