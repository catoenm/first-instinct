# Paired public-contract inference diagnostic

Test the qualified constant public tool contract on all 308 exposed report
forecasts, with original and augmented renderings for each fixed checkpoint:
Laya typed-decisions, unmodified Qwen3.5-9B, and original supervised step 2742.
Reuse the pinned weights, runtime, temperature, labels, options and numerical
methods from the [completed baseline](laya-forecast-v1-protocol.md). Preserve its
artifacts and results. The augmented contract is frozen separately in the
[local qualification](report-contract-v1-results.md).

There are **1,848 primary predictions**: 308 questions × two renderings × three
checkpoints. Each pair uses identical state/history, goal, costs, choices and
conditional labels; the sole change is the same appended public tool contract.
Interleave the two variants, alternating which appears first by the sorted original
question index. Keep the same pair order for all models. No batches or padding,
temperature fitting, checkpoint search, option reordering, new labels or training.
Every input must fit completely. Stop on numerical, identity or coverage failure;
do not adapt the experiment after observing scores.

The primary diagnostic is the existing supervised model, because its forecast
errors motivated this stage. The foundation and Laya are fixed controls. Report
all three regardless of outcome. Predeclare descriptive support for a checkpoint
as: expected Brier error improves by at least 0.02 overall and 0.05 on ambiguous
questions; overall log loss does not worsen; deterministic Brier error worsens
by at most 0.02 and deterministic expected outcome accuracy drops by at most two
percentage points. These are interpretation checks, not early selection gates,
statistical significance claims or release criteria. The 308 related questions
remain one mechanism with two physical worlds and four world/goal tasks.

Use the same GPU for both variants and all models. Before scoring, verify offline
checkpoint loading and repeat the same three synthetic menu checks. Native latency
uses the same three original questions selected in the previous baseline by Qwen
input length, with both renderings: three warmups and five measured calls per
rendering. Record those 144 timing forwards and 18 qualification forwards separately
from primary predictions. Keep instrumented timing separate. The correct minimum
usable memory is 44 GiB on a nominal 48-GB-or-larger device; record actual memory.
No foundation model may load on the Mac.

Locally qualify the exact bundle, source closure, numerical audit and paired-label
invariants before renting. Allow one inference rental at no more than $1.60/hour,
with a 90-minute absolute deadline, 30-minute setup and 30-minute model-work bounds,
and time reserved for verified recovery. Reserve at most $6 from the existing
remaining original cumulative authorization; this is not a fresh budget. Install
independent remote/host stop guards before setup. Preserve partial failures and
delete the owned GPU only after hash-verified recovery. No automatic retry,
extension, additional training or recurring hosting follows this diagnostic.

If the contract helps, the supported conclusion concerns information supplied at
inference on these exposed examples. It does not make the existing training better,
prove arbitrary-question calibration, establish fresh-mechanism transfer or promote
a checkpoint. If it does not help, record the negative result and reconsider the
interface hypothesis before expanding the same examples.

Before renting or scoring, final review replaced exact floating-point equality
between independent numerical implementations with the existing scalar auditor's
1e-9 absolute tolerance. Coverage, interpretation checks and all nonnumeric fields
must still match exactly. The initial local input archive is retained; the revised
input-v2 archive binds this audit correction and unchanged question bytes. This
changes no model computation, labels or improvement thresholds.
