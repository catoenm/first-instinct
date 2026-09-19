# Supervised warm-up for executable shell decisions

Declared before any model evaluation on this new data. This experiment follows
the mixed-game study and leaves its source, data, results, and demo unchanged.
It is the supervised preparation stage for a future real-shell reinforcement
experiment. It does not perform Proximal Policy Optimization, run a live command
proposer, or claim that replayed demonstrations are on-policy trajectories.

## Data and execution

Three authored families: layered JSON configuration repair, scoped SQLite
invoice recalculation, and quoted CSV aggregation. Each family has three binary
features: override/multiple-setting/boolean requirements; empty aggregate/date
filter/fee calculation; and quoted comma/embedded newline/negative totals.
Even-parity combinations train; 001 and 010 validate; 100 and 111 test.
All instances, candidate branches, and question views of a combination stay in
one split. This is a composition holdout within three known mechanisms, not
unseen domains or an unlimited supply of independent skills.

Generate 128 unique fixtures for each training combination and 16 for each
validation/test combination: 1,728 fixtures total. Each has six authored shell
commands, including two reference implementations, near misses, and a no-op.
Commands are templates, not proposals from a language model. Execute every
candidate from identical initial files in a pinned Python Docker image with
network disabled, no host credentials, a read-only root, and a writable temporary
directory. Keep the expected answers outside that container. This uses the
container execution boundary directly; it is not a Harbor orchestration run.

A host verifier checks final semantics and preservation of unrelated files,
rows, schema, and typed values. Exit status alone does not define success.
All reference branches must pass and all no-op branches fail; a data or execution
error aborts generation rather than silently becoming a negative training label.
Preserve exact commands, initial and final snapshots, outputs and errors, and
hashes. These internally authored commands qualify this data generator; they
are not a security qualification for unrestricted adversarial commands.

Produce six binary completion questions and two four-command choice menus per
fixture, with an explicit none-of-these option. Every question specifies one
execution with no later repairs. All verified successful options are acceptable;
do not mark another correct implementation wrong. Commands that merely inspect
are not being labelled globally bad: these questions specifically ask for
one-command completion. No labels claim optimal multi-step policy behavior or
calibrated long-horizon probabilities. Future trajectories must add evidence
gathering, recovery, partial observations, and explicit continuations.

The expected dataset has 12,288 training, 768 validation and 768 test questions,
with 12/6/6 structural groups. Tokenize without truncation at 3,072 tokens; any
exclusion must be resolved before freezing. Shuffling option order is independent
of correctness. Only public task, initial state, question and commands enter
model inputs. Reference identities, verifier outcomes and future snapshots do not.

## Paired training

All four arms start from the original Qwen3.5-9B supervised adapter at step 2,742,
not a game or reinforcement checkpoint. Two seeds, 907 and 1709; each compares:

- 42,288 existing general examples.
- The same first 30,000 general examples plus 12,288 new shell examples.

Two-epoch maximum, effective batch 64 (microbatch 8, accumulation 8), learning
rate 0.00001, validation every 100 updates, maximum two hours per arm. Fresh
optimizer state: this is not an exact resume. Existing internal rank-16 adapters
train; original foundation matrices remain frozen. Report actual rows, tokens,
updates and durations; neither equal caps nor equal row counts imply equal compute.

Use the same validation examples for all arms. Validation macro log loss averages
six shell question types and one aggregate general-retention task. The general
retention sample takes up to twelve examples from each existing validation task.
Candidates must preserve aggregate general accuracy within 1.5 percentage points
and log loss within 0.04 of their starting checkpoint. Among eligible candidates,
select the lowest macro log loss, requiring improvement of at least 0.0001.
Step zero is eligible. Stop after three scheduled checks without improvement.
Immutable adapter snapshots bind every validation record to its exact weights.
An unscheduled final check caused by stopping does not become a selection chance.

Evaluate original and four selected checkpoints on the frozen shell test and
the existing 1,128-question non-game transfer benchmark. The latter was opened
in previous research. Test results never alter training, selection or stopping.
Report selection and completion separately, both seeds, negative results and
general regressions. Six held-out shell composition groups support only a small
pilot, despite hundreds of questions. Keep the current demo checkpoint.

## Operation

One personal Runpod H200, at most $5.50/hour, ten-hour independent provider stop
deadline. Allocation cap $65 including a $10 storage allowance; cumulative
authorization remains $500. Prior compute estimate $133.0812683 excludes storage.
No paid model API, replacement rental, or additional training arms automatically.
Prepare and audit data locally before renting. Install authenticated remote stop
guard before runtime setup. Run training/evaluation/archiving independently of
the Mac connection. Recover and verify every artifact before deleting the bound
rental; preserve all failures. This experiment's completion is not completion of
the planned Harbor reinforcement-training integration.
