# Reservation language baseline, before new training

Declare this protocol before requesting model predictions. Keep the completed
reservation environment, small-policy study, main 9B weights, and old transfer
studies unchanged. This is a baseline on one authored mechanism, not a claim
of generality or a calibration-training result.

## Fixed model and cases

Use the already resident supervised Qwen3.5-9B checkpoint at step 2742 through
the loopback demo, with its existing 1,536-token limit. Bind the released
adapter's disk hashes, reported model identity, tokenizer files, runtime and
source hashes before inference. Disk hashes plus service metadata do not
attest in-memory tensors. No paid service or new model load is required.

Use two public price schedules: cheap queries `[1,1,16,4,12,12,12,8,0]` and
cheap attempts `[8,8,4,4,12,12,12,8,0]`, in quarter credits. Cross them with
horizons 3 and 6, and either a uniform prior over the four single-fault/ready
worlds or a uniform prior over the two combined-fault worlds. Enumerate every
supported world: 16 familiar and eight combined cases.

Run each of these 24 cases with three fixed renderings:

1. Original action descriptions and original menu order.
2. Exactly the same state/question/descriptions with reversed menu order.
3. Independently written equivalent action descriptions and question, with
   original menu order and unchanged state wording.

Interleave the three renderings within each case. Actions are greedy; each
rendering follows its own trajectory from the same initial world. The maximum
is 72 episodes and 324 model questions. Two hour wall-time cap; one serial
request at a time, 60-second HTTP timeout, zero automatic retries. Count an
attempt before dispatch. Preserve failed requests/partial trajectories and
stop on identity changes, invalid probabilities, errors or exhausted limits.
Do not edit prompts or replace cases after opening results.

## Public information and execution

Provide the goal, transaction semantics, fees, terminal reward rules, initial
prior, current public observation, and a concise action/result history. Do not
provide the sampled world, unobserved inventory, verifier output, policy labels
or prior reward. All nine actions remain available. The two familiar shortages
and missing-account cases are already known from the environment study; call
this mechanism familiar and reserve claims about new domains.

Use the unchanged C environment, previously qualified against actual SQLite.
Every selected action changes that state, including failed transactions and
no-ops. Record public before/after observations, exact submitted text, ordered
options, serialized chat messages, locally reproduced input token IDs, all
probabilities, selected semantic action, verifier state and reward. Gold/private
state belongs only in the execution journal, never in a model request.

Preflight the renderer and cached tokenizer without model forwards. Check all
recorded qualification histories plus a synthetic longest allowed history.
Reject overlength inputs without truncation; repeat exact token validation
before each adaptive request. Freeze the source only after this preparation.

## Analysis and controls

Report success and normalized return separately for each rendering and cohort,
plus the two horizons. Weight cases equally within each cohort. Report paired
initial-action agreement and matched-public-state action/probability differences
across renderings; divergent trajectories are not identical decision contexts.
Do not pool duplicates as independent statistical observations.

Execute finish-only and the existing public inspection/repair continuation on
the same cases. Compare with the already trained small policies only on exactly
matching profiles; do not claim a fair architecture comparison between numeric
and textual inputs. Evaluation histories use different information encodings.

Reconstruct the saved trajectories offline and recompute aggregates. Source,
checkpoint and artifact hashes must still match. No checkpoint selection,
training, temperature fitting or prompt search is permitted in this baseline.
Any later outcome-forecasting versus reward-training comparison needs its own
frozen data split, settings and compute budget, informed by these results but
evaluated on separate cases.
