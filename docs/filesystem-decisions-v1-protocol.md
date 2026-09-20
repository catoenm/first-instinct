# Filesystem mutation scope: bounded local qualification

This implements the filesystem direction proposed in outcome-v3-data-design;
reservation work is already separate and is not repeated. There is no model,
optimizer, paid API, rental, arbitrary command proposer or Harbor run here.
This candidate family is `train_candidate`, not consumed data or a new test.
Keep its entire mechanism out of any later test used to evaluate training on it.
A new whole transfer mechanism, such as the previously proposed calendar family,
requires its own implementation and qualification before a learning comparison.

Execute two task structures: change one named path while preserving other paths'
contents, or update the physical file including its existing aliases while
preserving the sharing relationships. Four equally likely concrete worlds have
identical contents but different hard-link partitions: no shared files, target
and second file shared, target and protected third file shared, or second and
third file shared. Contents alone do not disclose file identity.

The catalog contains actual Python filesystem operations: inspect link groups,
inspect contents, write target in place, atomically replace target, write two
paths in place, replace two paths, or stop. Opposite mutation-scope goals and
costs must reverse correct choices despite identical command wording. The parent
verifier checks exact contents, permissions, file set, required sharing and
unrelated preservation, independently of command exit status. No raw inode
number enters a prompt or portability comparison: relationships come from
actual device/inode equality and are encoded as sorted path groups.

Use two task roots, four underlying worlds, five cost/evidence regimes per task:
cheap in-place writes, cheap replacement, current evidence, expensive inspection,
and prohibitively costly writes. This gives eight world-and-goal tasks and 40
context cases, not 40 independent tasks. At most seven actions, three
continuations and 840 distinct alternatives, each separately replayed once:
1,680 physical branch executions and at most 12,000 tool commands. Initial
filesystem construction and prefix inspections are counted separately. Programs
are immutable strings with fixed relative paths, run in owned temporary worlds.
Unknown failures stop collection; they never become negative examples.

A forecast means the chosen action then immediate stopping, or the named
four-decision continuation. With current link evidence the continuation chooses
the cheapest correct single-target operation for the user goal. Without it,
the evidence continuation inspects unless the inspection alone costs at least
one reward unit; the no-observation continuation chooses the cheapest single-
target mutation. Both stop when every mutation costs at least one. These are
specified policies, not unrestricted optimality. Decision and observation-value
labels average actual executed future reward over compatible worlds; observation
costs are subtracted and already observed prefixes are sunk.

Require all outcome categories, uncertain forecasts under identical observations,
current evidence resolving ambiguity, matching byte/relationship snapshots on
independent replay, and independent verifier controls for collateral changes,
wrong link relationships and permissions. Demonstrate that changing costs or
user goals changes the preferred action. Menu-only and contents-only strategies
must not solve the evidence-dependent decision subset. Exact menu coverage and
receipt lineage must reproduce after serialization; reject missing/duplicated
branches or model-supplied labels. Save a rejection record on failure and never
overwrite the rejected collection.

This first slice does not add symlinks, concurrent writers, process crashes or
new application tools. The publication trajectory diagnostic supplies actual
failed commands and recovery histories separately; do not misstate those as
filesystem-family capabilities. Record native runtime versions and qualify a
bounded cross-platform replay in the existing pinned Linux image before using
this family for training. Reuse ToolSandbox for application workflows, and keep
arbitrary proposed commands isolated through a separately qualified Harbor stage.
