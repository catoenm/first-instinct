# Broader application execution: local qualification

This is a prospective, bounded qualification of AppWorld as a data substrate,
not a learned-model evaluation or a training collection. It follows the negative
[expanded comparison](expanded-decisions-v1-results.md) and
[forecast-selection diagnostic](forecast-selector-v1-results.md).

Pin the upstream public repository at
`42b5bcf3cd334fee33f0c37c02070a9f5807add5`, a separate Python 3.11 environment,
and the public 0.2.0 data bundle (34,908,601 bytes; SHA256
`c9299e6cafe92bce4592a3c117c047c973d1554a667c21dd81537e78ab2f532e`).
The local inventory inspected only upstream training metadata: 90 task instances
from 30 generator programs. No development/test task contents or model predictions
enter selection. Freeze the installed dependency versions separately.

Reserve every training-split generator whose tasks require the simulated `venmo`
application as a new mechanism holdout. Inspecting authored qualification and
ownership metadata is allowed; do not execute those tasks or use their derived
rows in training. This reserves seven programs / 21 instances; the remaining
23 programs / 69 instances are candidates, not already prepared training data.
An internal development partition still needs to be declared before learning.

Before task execution, choose four candidate instances with different required
application sets and different generator IDs. Sort by the upstream reference
call count, then generator and task identity; take the first qualifying instance
per application set, with reference call count at most 32. This intentionally
qualifies small tasks first and makes no claim about harder tasks.

For each, create three isolated worlds in separate processes: no-op, upstream
reference solution, independent reference replay. Cap at twelve world instances,
one execution interaction each, 64 application calls each, 768 total calls and
60 seconds per interaction. Use fixed random seed and the dataset's frozen time.
Use local in-process application clients and block outgoing socket connections.
There are zero model calls, new training questions, optimizer steps or rentals.
Never send messages or make financial transactions outside the simulated apps.

Require a nonempty evaluator, failed no-op, passing reference solution, unchanged
no-op databases, identical initial databases, and exact reference/replay database
hashes, request traces and evaluation counts. An exception or unavailable evaluator
fails qualification. Preserve all partial receipts and failures; never replace
an exception with a negative task label. Ground truth and reference code are
verifier-only. A later model-facing proposal stage must build candidates from
public observations and tool schemas, not copy private solution arguments.

This checks executability and a basic non-vacuity control. It does not certify
all task assertions, arbitrary commands, side-effect detection, counterfactual
uncertainty, or an information-respecting decision menu. Those are subsequent
local qualification gates before any data is consumed by the 9B model.

AppWorld's [official project](https://github.com/StonyBrookNLP/appworld) describes
its executable apps and task evaluation. Its protected code/data and derivatives
must remain encrypted for public redistribution; training and served model outputs
are explicitly distinguished. Keep unpacked material and raw derived receipts
private locally. Publish only aggregate qualification measurements, hashes and
our generic adapter code. This is a restricted substrate check, not an official
AppWorld benchmark run.

## Preserved qualification correction

The first no-op reached the evaluator but its expected failed assertion escaped
because the wrapper requested unsuppressed errors. Preserve that execution, log,
freeze and source. The retry uses a new experiment/output identity and lets the
upstream tracker record **only assertion failures**. A narrow wrapper propagates
all other exception types instead of using upstream's broader exception suppression.
Three regression tests check this boundary and reject empty/incomplete evaluations.
No task label or training example from the first attempt was accepted.
