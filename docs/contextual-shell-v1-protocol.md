# Context-dependent shell supervision

This is a new prospective experiment after the cancelled shell-supervised-v1
pilot. That pilot's executed labels were valid, but a normalized command lookup
scored 93.66% on validation without reading either the task or the files. We
cancelled it during baseline evaluation, before any optimizer updates, recovered
all 16 artifacts and deleted the H200. Estimated compute: $0.57313427, excluding
storage. The old freeze remains unchanged; its four training arms never ran.

## Dataset qualification

The new dataset uses three authored mechanisms. Configuration tasks add one
worker to the eligible service with the higher or lower queue-per-worker ratio,
respecting overlays and protected settings. Database tasks repair the eligible
invoice with the higher or lower subtotal discrepancy, preserving everything
else. Report tasks aggregate settled January sales and output the region with
the higher or lower net total, respecting quoting, filtering and negative sums.
These are explicit task priorities, not claims that choosing a lower priority
is generally the best operational policy.

Each quartet crosses two initial file states with two opposite instructions.
All four contexts receive the same four candidate commands, two implementations
for each possible target. Every candidate succeeds in two contexts and fails in
two. Changing only the files reverses each answer; changing only the instruction
also reverses each answer. Correct command patterns therefore cannot substitute
for considering the context. Menus share one shuffled order within a quartet.

Commands execute from fresh files in the same pinned, network-disabled Docker
boundary used by the preceding pilot. A no-op additionally executes as a
qualification control, never as a training example. Independent final-state
checks enforce complete results and preservation. A separate audit recomputes
the public arithmetic/ranking rules, checks visible files against actual initial
snapshots, reconstructs every label, and verifies complete quartet/split ownership.

Three feature bits per family vary overlay precedence/worker normalization/scope,
empty aggregates/fees/scope, and CSV commas/newlines/negative totals. Even-parity
combinations train; 001/010 validate; 100/111 test. All variants of a combination
stay together. There are 12 training, six validation and six test structural
groups, not thousands of independent skills or held-out domains.

Generate 32 quartets per training combination and eight per validation/test
combination: 480 quartets, 1,920 contexts, 9,600 executed branches. Four forecasts
and one four-option choice per context yield 7,680 training, 960 validation and
960 test questions. Both valid implementations are accepted. No command selection
is mislabeled as a probability of future task success.

Data gates passed before renting: reference semantics and no-op controls;
independent reconstruction; zero token exclusions at 3,072; normalized command
lookup at 50% on validation. Even an oracle choosing the most frequent answer
per identical ablated input is limited to 50% if either the goal or the files are
removed. Check this separately by task and split, and again after option shuffling.
Use `contextual_shell_prepare`; the general preparer's per-row shuffle would
break the identical-menu condition. These gates remove specified shortcuts;
they do not prove the dataset is free of every shortcut or broadly representative.

A preregistration engineering probe used the existing local original 9B service:
24 forecasts from six validation quartets, one fixed command per quartet. It
scored 14/24 and solved no entire quartet. This opened validation sample is not
a test result, a population estimate, or independent attestation of resident
parameter bytes. No test-model scores have been examined for this new study.

## Training and evaluation

Use the unchanged shell_experiment trainer, starting each arm from the original
Qwen3.5-9B supervised step-2,742 adapter. Two seeds, 907 and 1709, each compare
37,680 general examples with 30,000 of those examples plus 7,680 context examples.
Internal rank-16 adapters train; foundation matrices remain frozen. Optimizer
state starts fresh. Do not describe this as language-model pretraining.

Recipe unchanged: two epochs maximum, batch 8 with eight accumulation steps,
learning rate 0.00001, validation every 100 updates, two-hour cap per arm.
The common validation set has six contextual task aggregates and one general
retention aggregate (up to 12 examples per old validation task). Select lowest
macro log loss only if general accuracy drops at most 1.5 percentage points and
general log loss rises at most 0.04 from the original. Require 0.0001 improvement;
stop after three scheduled checks without an eligible improvement. Step zero is
eligible. Immutable prediction/adapter receipts bind selection to exact weights.

Evaluate original and all four selected checkpoints on the new 960-question
test and existing 1,128-question general-transfer set. The latter has been opened
in previous studies. Test does not select weights, learning rates or stopping.
Report both seeds, regressions, selected step, actual row/token consumption and
compute. The actual caps are shared, not a guarantee of equal realized compute.
The demo stays on its original checkpoint. This is supervised preparation;
live Harbor reinforcement learning still requires a separate integration.

## Compute boundary

One newly frozen, corrected experiment within the user's existing authorization:
one personal H200 at no more than $5.40/hour, ten-hour independent provider stop.
The two shell pilots together remain within $65, including a $10 storage reserve:
$0.57313427 already spent plus at most $54 new compute. Prior cumulative compute
is approximately $133.65440257 excluding storage; cumulative authorization is
$500. This is a deliberate data redesign following qualification, not an
automatic retry of the rejected data. No further rental, new arms or paid model
API automatically. Recover and verify all artifacts before deleting this rental.

Runtime setup, eight unit checks, baseline evaluation, four arms, evaluation and
archiving run remotely without depending on the Mac connection. Local recovery
can resume after disconnection; independent shutdown retains the volume for
recovery if necessary. No Phantom compute is used.
