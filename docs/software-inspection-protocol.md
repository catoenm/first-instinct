# Software inspection: development protocol

This is an open development experiment, not a final sealed benchmark or a
reconstruction of TypeSafe's private training recipe. The protocol is written
before collecting outcomes. Changes and failed attempts must remain disclosed.

## Sources and independent units

Read TheAlgorithms/Python at revision
`a381578994d545e44f26afabbd2303746a2dc358`, under its MIT license. Parse source as
data on the host; run it only in isolated Docker workers. Extract functions with
at least four distinct literal-input, literal-output doctests. Remove docstrings
from candidate code. Retain the human-written description preceding examples.
Allow a bounded set of standard-library imports and literal module constants.
Discard module entry points. Unsupported modules are logged, not silently fixed.

Keep every function, mutation, input perturbation, evidence view, and episode
from a source module in the same split. Reserve the complete strings and ciphers
directories as unfamiliar families. Other modules use a fixed path hash:
80% training, 10% validation, 10% test. Check normalized-code duplication across
splits. Report source modules and functions, accepted candidates, executed tests,
and replayed interactions separately. Do not call mutations independent tasks.
One upstream repository does not support a claim of new-repository transfer.

**Pre-training split repair:** a source audit found identical prime-checking
helpers in different Project Euler modules. The raw build retains its initial
split fields for provenance. Before any training, `curate.py` joins modules that
share a function body (ignoring its name and annotations), and joins all solution
files for the same Project Euler problem. It assigns complete connected groups
using a hash of their sorted module paths, retaining the same 80/10/10 rule and
moving any group touching strings/ciphers to the family holdout. This catches
exact function clones and known problem variants, not all semantic duplicates.

## Verification

Include unchanged source plus at most 32 deterministic, syntax-tree mutations
per function, chosen without seeing outcomes. Use upstream doctests to check the
original implementation and verifier. Generate up to 48 perturbed literal calls;
retain only calls on which the original returns a supported deterministic value.
These additional targets mean compatibility with the pinned implementation,
not independently established mathematical correctness.

Execute twice in fresh subprocesses, with distinct hash seeds. Quarantine
timeouts, unsupported originals, source-test disagreement, and unstable results.
Keep deterministic candidate exceptions as failed checks when the original
returns a value. Use strict typed equality; no approximate floating-point tests.
The root process has a fixed wall-time limit and the container has no network,
no host mounts, a read-only root, dropped capabilities, a non-root user, and
bounded memory, processes and processor time. This is containment for this
reviewable corpus, not a claim of a perfect hostile-code sandbox.

## Episode

The event is: **this candidate passes the complete fixed suite for this task**.
The suite includes initial, purchasable and permanently hidden checks. One known
failure therefore implies failure of this event; a passing revealed subset does
not establish success. Tests and mutation ancestry are never model inputs until
explicitly revealed (ancestry never is). Return only visible state and available
actions to the policy; keep receipts and outcomes in the environment controller.

Start with the contract, candidate code and one check. Offer two distinct groups
of additional checks, and an explicitly redundant copy of the initial check.
Each purchase costs a declared amount. Permit at most two purchases, then require
a terminal probability report on a 21-point grid from 0 to 1. The reward is
`1 - (report - outcome)^2 - total inspection cost`, without discounting. This
elicits probabilities of a fixed event in expectation; it does not magically
make a finite, misspecified model calibrated. Costs vary independently of the
underlying outcome. Copy and cost interventions should preserve event forecasts.

## Comparisons and measurements

First run small, processor-based policies to validate the environment and
algorithm before renting another accelerator. Compare Proximal Policy
Optimization with terminal outcome rewards, the same method with continued
forecast exercises on randomly sampled evidence states, and direct supervised
outcome learning. Include a fourth control with equally many extra forecast
exercises at the terminal states its policy selected. Both exercise methods use
the root episode's candidate; the broader exercise chooses among all seven
allowed evidence states. Costs also come from an independent exercise stream in
the broader condition, so state coverage includes price coverage. Extra exercise
counts match, although actual policy-update counts may differ under the same
divergence stopping rule. Log all counts and any difference in reward
presentations. Evaluation uses fixed common states as well as each
policy's own trajectories. Do not compare calibration only on selected states.

Report Brier score (mean squared probability error against realized outcomes),
log loss, calibration bins, inspection cost, duplicate purchases, and total reward.
There is no known true per-program probability, so do not report error against
an imaginary analytic probability. Report held-out module and family results
separately. Later language training must distinguish engineered-feature policy
results, frozen language representations, and a trained language backbone.

The research question is whether continued, representative forecast practice
helps a policy avoid losing calibration as it learns what evidence to inspect.
A useful negative result or verifier failure is evidence, not a reason to
silently regenerate the test set.

**Development correction:** the initial trainer used the same value estimate
for a full interaction and a forecast-only exercise with identical visible
evidence. The exercises remove inspection actions, so their future returns have
a different meaning. Before reporting results, that pilot was stopped and the
value network was given the available-inspection mask. The outcome data and
splits were unchanged; the replacement run is `software-inspection-training-v2`.

## Exploratory hybrid follow-up

The early reward-only runs sometimes collapsed to a nearly constant report.
The follow-up is explicitly motivated by that observation, not a preregistered
confirmatory comparison. Initialize a forecast network from each seed's selected
supervised checkpoint, and train a separate acquisition policy using Proximal
Policy Optimization. These networks share the same visible input features, not
trainable weights. Round the forecast to the existing report grid when acting.

Compare freezing that predictor, continuing outcome-label training on terminal
states, repeating those terminal labels twice, and using the second equal-sized
label batch on random evidence/price states of the same candidate. All have the
same starting predictor and interaction candidate stream; the last two have
equal additional label presentations. Select checkpoints by validation workflow
reward, keeping full common-state forecast measurements. This objective differs
from the original report-policy study's forecast-based selection. Do not claim
this isolates one change relative to that study: initialization, report
parameterization, supervision and separation of networks all differ.
