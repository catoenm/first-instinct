# A qualified data stage, not another trained model

The next local curriculum passed execution, read-back and tokenizer audits.
The active `evidence-decisions-v2` 9B comparison remains unchanged. No v3
questions have reached that run or any other 9B optimizer, and no additional
graphics processor was rented for this work.

The useful addition is an actual filesystem publication workflow: stage an
artifact, validate its bytes, then atomically replace the live artifact.
Restaging can invalidate validation. A premature commit fails and permits
recovery; a completed wrong publication ends the attempt. Parent-side checks
compare the published content and protect the source files and executable.
Configuration and SQLite repair train; the entire CSV-report family validates;
publication is reserved for transfer. This means withheld from the proposed
post-training mixture, not unknown to the foundation's pretraining.

All four families cover missing, fresh, duplicated, historical and contradictory
evidence; expensive observations; an offline inspection service; preparation;
and actual failed writes. Contradictory reports here have explicitly obsolete
provenance. Noisy competing current sources are still a gap.

## What was collected

| Quantity | Accepted collection |
| --- | ---: |
| Underlying authored families | 4 |
| Root fixtures | 4 |
| Distinct initial file worlds | 8 |
| World-and-goal tasks | 16 |
| World/goal/cost/prefix variants | 144 |
| Executed alternative-action/continuation branches | 3,024 |
| Actual commands, including menu-construction prefixes | 8,892 |
| Distinct visible decision contexts | 88 |
| Empirical consequence questions | 2,016 |
| Next-action questions | 88 |
| Observation-value questions | 88 |
| Questions consumed by a trained 9B model | **0** |

These 2,192 questions are not 2,192 independent tasks. The corpus retains
120 groups where identical forecast inputs have different legitimate outcomes.
Its first qualification deliberately has just one root per family; this is a
mechanics and data-contract check, not a sufficiently broad generality study.

Every alternative runs in a fresh copy of the same initial files, including
execution of the visible prefix. Forecasts distinguish executing one command
and stopping from following the named adaptive continuation. Next-action labels
average the executed returns across compatible worlds. They never reveal the
best action for a hidden world the model could not identify.

Observation value compares an executed inspection-and-continuation plan with
the best executed plan forbidden to obtain more observations. For the
configuration fixture, this changes with the actual context:

| Visible situation | Inspection's advantage in reward units |
| --- | ---: |
| Evidence missing; inspection costs 0.04 | +0.94 |
| Current evidence already available | −0.04 |
| Two copies of the same current evidence | −0.04 |
| Evidence missing; inspection costs 1.20 | −0.22 |

These are exact values for the declared finite worlds and continuation, not
learned predictions, dollar prices, or a universal value-of-information oracle.
The no-inspection plans include stopping.

## Qualification and a caught defect

The read-only auditor reconstructs all 2,192 questions and their receipt
dependencies, public histories, terminal checks and cost totals. Reference
continuations complete every affordable task and stop in the expensive hidden
cases. On the fresh-evidence quartets, a lookup table seeing only the menu,
only the goal without observations, or observations without the goal can reach
at most **50%** next-action accuracy. Both pieces of context are necessary.

The pinned tokenizer found a maximum of 967 tokens for these questions and
1,088 across 1,698 distinct executed actor histories, below the existing 1,536
limit. No truncation or model inference was used. Thirteen environment and
accounting tests and 25 related regression tests passed.

An earlier local collection failed the read-back audit: JSON serialization
could reorder a command menu, and publication base metadata shared a mutable
file mapping with the environment. The fixes canonicalize menu construction,
separate the base files, and test serialization and protected scripts. That
collection is **rejected and untrained**; its receipts and source remain local.
It also executed 3,024 branches and 8,892 commands, separately from the accepted
counts above. Small development probes and integration tests are excluded from
both production totals. Failed qualification is not erased from the accounting.

The small-network learning check also passed: four real episodes, seven policy
transitions and 36 commands; nonzero language gradients from the actor and
forecasts; zero language gradient from the detached value estimator; and an
actual optimizer update. This is a randomly initialized test network, **not a
9B performance result**.

## Application evidence and the next training gate

The existing ToolSandbox cohort is now exposed as 720 forecast, 144 decision
and 144 inspection-value diagnostic questions, joined by the same visible
history. All labels derive from the previously verified executions. This
required zero new tool calls. Its 48 roots, 192 database worlds and 1,152 unique
world/program labels are old evidence, not newly generated data. It remains
diagnostic-only because earlier model results on it have already been examined.

For the next application training mechanism, the installed ToolSandbox messaging
and device-setting APIs are promising: sending writes a local simulated message
database, requires cellular service and a unique self contact, while low-battery
mode blocks enabling cellular service. Enabling low-battery mode also disables
other services, creating preservation constraints. A bounded new collection can
exercise prerequisites, already-satisfied requests and harmful duplicate sends
without external communication. This candidate has been source-inspected but
has **not** been collected or trained in this stage.

Before another rental, finish the current comparison and reconcile spending;
broaden and freeze the actual training mixture; qualify this application
mechanism; and prepare the controlled 9B launch. Keep the same original
supervised checkpoint, general replay and paired seeds across forecast-only,
reward-only Proximal Policy Optimization and combined arms. The
[prospective protocol](decision-curriculum-v3-protocol.md) gives the improvement,
retention and stopping checks. A new run needs its own bounded allocation
within the remaining original compute budget.

The new live actor controls its own subsequent actions. Forecast questions
instead name their fixed continuation. Their prompts and execution records
remain distinct. The new categorical Brier score sums errors across three
outcomes; it must not be compared numerically with v2's single binary component.

Dynamic command proposals are still a separate unimplemented stage. These
menus are authored. A future Harbor-based proposer evaluation must record the
exact menu, execute candidates, and separate missing useful proposals from bad
selection. This work does not establish broad Jev-like performance or recover
Jev's training method.

## Reproduce and inspect

[Published receipts, questions, freezes and audits](../results/decision-curriculum-v3/)
include the entire accepted collection as compressed JSON lines. Files named
`contexts-private` contain verifier-only targets; “private” refers to exclusion
from model inputs, not restricted publication.

To unpack the accepted artifacts without running any environment:

```sh
python - <<'PY'
import gzip, shutil
from pathlib import Path
src = Path('results/decision-curriculum-v3')
dst = Path('output/decision-curriculum-v3-reproduced')
dst.mkdir(parents=True, exist_ok=False)
for p in src.glob('*.jsonl.gz'):
    with gzip.open(p, 'rb') as i, (dst / p.name.removesuffix('.gz')).open('wb') as o:
        shutil.copyfileobj(i, o)
for name in ('pre-execution-freeze.json', 'qualification.json'):
    shutil.copyfile(src / name, dst / name)
PY
python -m tool_lab.decision_audit \
  --folder output/decision-curriculum-v3-reproduced \
  --output output/decision-curriculum-v3-reproduced/audit.json
```

The audit requires the matching committed sources. `--tokenize` optionally uses
the locally cached pinned Qwen tokenizer. New execution is separate:

```sh
FIRST_INSTINCT_DOCKER_TESTS=1 python -m unittest test_decision_curriculum -v
python -m tool_lab.decision_collect --output output/a-new-local-qualification
```

The Docker commands use the pinned, network-disabled executor. No Harbor run,
paid model API, graphics processor, or language-model judge supplies these labels.
