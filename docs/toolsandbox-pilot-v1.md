# ToolSandbox exact-state pilot, 2026-09-18

The local pilot completed **64 fixture roots, 384 distinct program outcomes, and 768 branch executions including repeat checks**. It uses actual pinned ToolSandbox contact/reminder functions and their Polars execution context. It does not train or query a model. These are authored fixtures running public upstream tool implementations, **not the official ToolSandbox benchmark or its scenario corpus**.

Source: [ToolSandbox at c8571d7854316d2e1c5f288e59fe1e34e53f6dd1](https://github.com/apple-aiml-research/ToolSandbox/tree/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1). The adapter checks SHA-256 values for all 17 downloaded source/license/manifest files before importing. The ordinary, unchanged `tool_sandbox.tools` initializer executes. No substitute modules, rewritten tool functions, altered dependency pins, model clients, official scenarios, or paid services are used.

## Actual result

| Outcome | Distinct program labels |
| --- | ---: |
| Exact success | 64 |
| No change / no-op | 72 |
| Invalid call without mutation | 64 |
| Wrong target | 32 |
| Collateral state change | 64 |
| Incorrect query result | 32 |
| Other failure, including duplicates or a later exception | 56 |

The collection took 15.465 seconds on the local CPU. It executed 832 program tool calls, 4,608 fixture-setup calls, and 1,536 public observation calls, including repeats. Every repeated branch agreed on the full database snapshots, result, exception type/message after the normalization described below, public input, and program. Counts describe the final collection; small integration tests also exercised the tools separately.

The eight tests passed in the isolated runtime in 3.957 seconds. Six dependency-free tests also pass in the ordinary project environment; two integration tests skip only when optional dependencies are absent. A present but broken runtime, missing pinned source, or mismatched source hash fails those integration tests.

The authored inputs, outcome labels, replay receipts and manifest are published in
[results/toolsandbox-pilot-v1](../results/toolsandbox-pilot-v1/). They reproduce
the original local artifact hashes. Downloaded upstream source, runtime, wheels,
and dependency notices remain ignored under `.local/`; none is redistributed in
this repository. Frozen experiment sources, protocols, requirements, and README
were not edited.

## Mechanisms and prospective split

The on-disk `prospective-plan.json` is written before importing upstream code or executing fixtures. The **requested goal's operation group** determines the split; fixture seeds and labels do not choose it. This does not hold out every tool primitive executed inside candidate programs.

| Split | Task families | Roots |
| --- | --- | ---: |
| Train | Contact creation; reminder creation; exact phone lookup; bounded timestamp lookup | 32 |
| Validation | Contact phone update; reminder rescheduling | 16 |
| Test | Contact deletion; reminder deletion | 16 |

These are **eight task families across four operation groups**, not eight independent mechanisms. Relevant behavior includes mutable keyed rows, creation that allows duplicates, argument/type validation, exact and interval queries, deletion of missing entries, and reminder updates that reset the creation timestamp. The tools use their actual upstream implementations: [contact functions](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/tools/contact.py), [reminder functions](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/tools/reminder.py), and [execution context](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/common/execution_context.py).

Each family has eight roots and six finite candidate programs per root. Fixture values and UUID namespaces are disjoint across families. Candidate order is deterministically shuffled. The source seed is 1907. The module rejects more than 512 roots or more than 4,096 branch executions; because it always uses six candidates with two executions, a requested root count must also fit the tighter branch budget. Root counts must be multiples of eight. Candidate programs have a hard eight-call bound; this pilot uses at most two calls.

The creation/query training negatives already execute `remove_contact` and `remove_reminder` as collateral actions. Consequently deletion is a held-out **requested goal**, but its underlying removal primitives are not unseen. This pilot cannot support an unseen-tool or novel-mechanism generalization claim. It tests a shift in requested goal groups, with overlapping tool primitives, and is too small for a generalization claim. There is no validation/test-driven selection and no fitted predictor in this pilot.

An independent post-execution review reproduced all 384 stored labels, public candidate lists, receipt digests, and manifest file hashes. A separate check derived exact-success conditions directly from the public requests and observed states, including all namespace preservation and documented reminder timestamp side effects; it agreed on all 384 rows and 64 successes. This review clarified the split interpretation above. It did not alter the executed adapter, prospective registry, original manifest, or collected receipts.

## Public inputs and exact verification

Every root starts a new real `ExecutionContext`. Six real add calls initialize three contacts and three reminders. Two real search calls produce the full visible observation history. `visible_candidates(public)` accepts only the request, those returned records, and the declared continuation. It has no verifier, private database, hidden target, fixture seed, or expected-outcome argument. Candidate IDs are neutral `choice_N` values.

For update/deletion requests, the target ID is explicitly present in the public query response. Every existing-row candidate ID is obtained from those visible records; the deliberately invalid deletion uses the public literal `missing-public-literal`. The candidate builder has a handcrafted narrow grammar. Its ability to include a successful action is not evidence that a general agent can discover that action.

The continuation is explicit: execute the finite candidate program, stop at its first exception, then stop. There is no undisclosed controller, rollout policy, or replanning. The example contains that continuation and the entire proposed program.

Verification compares **all five namespaces**: contacts, reminders, settings, messaging, and sandbox. Success requires the exact requested target change or exact returned query rows, preservation of unrelated rows/namespaces, no unrequested target-field changes, and no exception. Creation requires exactly one new row with all requested fields; two successful add calls therefore fail the single-creation goal. Query success also requires unchanged state. A good edit followed by an unrelated deletion fails. Deleting the desired row and then raising an error on a second deletion does not receive success.

The seven outcome labels form an ordered, exhaustive partition. Success comes first only when the goal, frame condition, and absence of errors all hold. Unchanged state is then classified as invalid call, no-op, or wrong query result. An off-target mutation without changing the requested row is wrong-target; other frame violations are collateral changes. Remaining changed states are other failures. Separate `goal_satisfied`, `frame_preserved`, `database_unchanged`, and `exception_occurred` fields preserve the underlying checks. Target-field corruption may fall under other failure, but never success.

The verifier receipt includes setup calls, initial/final full snapshots, private goal, complete program trace, and exception details. `examples.jsonl` omits that verifier receipt. Forecast inputs are **only `public_input`**; the adjacent `label` is a training target and must never be concatenated into the prompt. Root/family metadata is for grouping, not prediction input.

## Determinism and containment

The adapter patches only the contact/reminder modules' imported UUID functions and the reminder module's clock reference. UUIDs come from UUID5 over root identity and call index. Setup reminder creation occurs at timestamp 1,799,992,800; program execution occurs at 1,800,000,000 UTC. Each repeated branch rebuilds the fixture through the same real calls. Database snapshots are canonically sorted.

An upstream missing-row exception includes a Polars expression's process memory address. Exception messages replace hexadecimal address tokens with `<address>`; other text, exception type, all tool results, and complete database snapshots are retained and compared. This normalization is recorded on each exception. It does not normalize entity IDs, timestamps, phone numbers, or database contents.

The runtime is launched with an explicit clean environment and isolated HOME. During import and replay, Python socket creation, DNS helpers, connection helpers, subprocess creation, and common `os` process-launch functions are intercepted. Four active probes—socket, connection, DNS, and process creation—were blocked. One additional socket creation was blocked during package imports. **Zero additional attempts occurred during actual tool replay.** The manifest records the attempt counts and phase boundary.

This is a **Python socket/subprocess guard, not an OS/network sandbox guarantee**. It does not claim to contain arbitrary native syscalls or malicious extensions. The actual code used here was pinned, read, and restricted to eight local contact/reminder functions. RapidAPI helpers are imported by the pristine package initializer but never selected or executed.

## What the labels mean

Each label is one exact, deterministic execution outcome, plus the observed number of executed program calls. `outcome_one_hot` encodes that outcome category. It is not an empirical estimate of a nontrivial stochastic probability and it is not a distribution over which action the model prefers.

These public inputs expose the local records through real queries. Consequently this pilot does not demonstrate value of information, uncertainty over hidden database states, general arbitrary questions, user-interaction reasoning, or learned calibration. A future predictor could assign epistemic probabilities and be scored against held-out labels with multiclass Brier/log loss. A stochastic ground-truth distribution would require a declared prior over hidden states, conditioning on the same visible history, and repeated counterfactual execution under a fixed continuation. None is claimed here.

The cheap next use is a separately declared supervised outcome/control experiment on these finite programs, with grouped splits and a retained general-language audit. Scaling names or seeds alone would not add mechanisms. No pilot data has yet entered model training or the frozen GPU experiment.

## Reproduce and inspect

Published artifacts are copied byte-for-byte from these local execution outputs:

- `output/toolsandbox-pilot-v1/manifest.json`: actual counts, pins, source hashes, installed versions, adapter hash, guard proof, and file hashes.
- `output/toolsandbox-pilot-v1/prospective-plan.json`: root/family registry and bounds.
- `output/toolsandbox-pilot-v1/examples.jsonl`: public inputs plus deterministic labels.
- `output/toolsandbox-pilot-v1/receipts.jsonl`: full verifier and replay receipts.
- `.local/toolsandbox-upstream/`: unchanged pinned source/license files.
- `.local/toolsandbox-setup/download-receipt.json`: exact wheel URLs, versions, sizes, and SHA-256 values.
- `.local/toolsandbox-setup/dependency-notices/`: license texts retained from installed wheels.
- `.local/toolsandbox-setup/test-final.log` and `pilot-run.log`: test/execution logs.

Run from the repository root with a fresh empty output directory:

```sh
env -i PATH=/usr/bin:/bin HOME="$PWD/.local/toolsandbox-setup" TZ=UTC PYTHONNOUSERSITE=1 \
  .local/toolsandbox-venv/bin/python -m general_lab.toolsandbox_pilot \
  --source .local/toolsandbox-upstream --output output/toolsandbox-pilot-replay --roots 64 --seed 1907

env -i PATH=/usr/bin:/bin HOME="$PWD/.local/toolsandbox-setup" TZ=UTC PYTHONNOUSERSITE=1 \
  .local/toolsandbox-venv/bin/python -m unittest test_toolsandbox_pilot -v
```

The isolated interpreter is CPython 3.11.16 from [Astral's 20260901 standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901), macOS ARM64 install-only stripped archive, SHA-256 `768f05cf200273bbdda9a5955a5a6892a4b22f2a0b1e4b0a9160f5c7fce86816`. It is under `.local/toolsandbox-python/`; the venv is `.local/toolsandbox-venv/`. The initially available Python 3.12 was rejected because [upstream-pinned ccy 1.3.1 requires Python below 3.12](https://pypi.org/project/ccy/1.3.1/). No compatibility requirement was overridden.

Direct pins are `ccy==1.3.1`, `decorator==5.1.1`, `dill==0.3.8`, `geopy==2.4.1`, `holidays==0.51`, `phonenumbers==8.13.39`, `pint==0.23`, `polars==0.20.31`, `rapidfuzz==3.9.3`, `requests==2.32.3`, `StrEnum==0.4.15`, and `typing_extensions==4.12.2`. All come from the pinned upstream manifest. Eight resolved transitive distributions are recorded by version and hash in the local download receipt; use those wheel files with `pip install --no-index --no-deps .local/toolsandbox-setup/wheels/*.whl` to avoid a new resolution. No model packages were needed.

The Python archive was 26,961,472 bytes; the 20 wheels total 32,058,571 bytes. Including pip's resolver download and the retained second wheel download, bulk transfers total 91,078,614 bytes. Small source/index/metadata requests are additional; this is an artifact-size accounting rather than packet-level metering. The setup stayed comfortably below the 2 GB cap. Later replay and tests need no network. The adapter itself contains no download path.

## Separate upstream notices and license scope

**ToolSandbox: Copyright (C) 2024 Apple Inc. All Rights Reserved.** The full [root license](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/LICENSE) was inspected before execution. It grants a personal, non-exclusive copyright license to use, reproduce, modify, and redistribute source/binary forms, with or without modifications. Redistribution of the entire unmodified package must retain its notice, terms, and disclaimers. Apple names/marks cannot be used for endorsement without permission; no patent rights are granted, and the software carries warranty/liability disclaimers. This is an Apple license, not MIT or Apache. The complete original text remains alongside the downloaded sources.

The [upstream acknowledgments](https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/ACKNOWLEDGEMENTS) and relevant installed-wheel license texts were inspected. Direct components include BSD-style terms for ccy, decorator, dill, and Pint; MIT terms for geopy, Polars, RapidFuzz, StrEnum, and Holidays; Apache 2.0 for phonenumbers and requests; and the Python/PSF license history supplied with typing_extensions. Holidays and typing_extensions are accounted for from their wheel notices rather than assuming the acknowledgment list is exhaustive. Preserve each component's actual notices when redistributing it. Transitive certifi includes a Mozilla Public License 2.0 certificate bundle; transitive dependencies are not all under one common permissive license.

This work leaves upstream sources, distributions, runtime, and dependency notices
ignored. It publishes our authored fixtures and execution receipts, no upstream
scenario corpus and no trained model. The local notice inventory records full
files and provenance, not a claim that every embedded native dependency or
standalone-interpreter component has received an exhaustive redistribution audit.
