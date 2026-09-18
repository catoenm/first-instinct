# Executable data after outcome-v2

Research date: **2026-09-18**. This is a prospective data recommendation, separate
from the frozen, running experiment. **First integrate a small offline,
exact-state subset of ToolSandbox. Choose AppWorld as the more ambitious next
environment.** Neither recommendation depends on the current experiment's test
results.

The five candidates below execute actual Python tools, database mutations or
browser state transitions in simulated applications. They do not supply real
customer outcomes, general counterfactual labels, or arbitrary-question truth.
Their value is additional executable mechanisms with inspectable contracts.

## Ranked choices

| Priority | Candidate and useful new mechanisms | Offline feasibility / incremental service cost | License and immediate limitation |
| --- | --- | --- | --- |
| 1 | **ToolSandbox local tools:** contact lookup, reminder creation/editing/removal, settings dependencies, preservation of unrelated records | Local subset can avoid model services and external search; $0 API calls in the proposed adapter. Full conversational benchmark is different. | Custom Apple software license, not MIT/Apache; retain applicable notices and dependency terms. Aggregate benchmark similarity is not an exact success label. |
| 2 | **AppWorld:** cross-app API workflows, authentication, database consistency and unintended side effects | Default unified execution uses local FastAPI TestClient and simulated app APIs. Setup requires package/bundle downloads. | Public code Apache-2.0; protected app/task material has additional encrypted-public-redistribution requirements. Richest harder follow-on. |
| 3 | **Current τ repository, telecom solo subset:** coordinated agent/user device state, policy restrictions, troubleshooting | Telecom supports solo mode. Standard conversational evaluation uses a user model; voice/knowledge add further dependencies. Avoid those paths initially. | MIT; current checkout is τ³/version 1.0.1. Filtering for genuinely active executable criteria is essential. |
| 4 | **BrowserGym + pinned MiniWoB++:** DOM selection, forms, toggles, submission and browser interaction errors | Chromium plus locally served HTML; no model service is intrinsic to stepping the environment. | BrowserGym Apache-2.0; pinned MiniWoB++ MIT. Positive partial credit is converted to success by the adapter. |
| 5 | **OpenApps:** mutable calendar/todo/messenger apps with configurable UI and cross-app tasks | Local Python apps and vendored frontend assets; skip maps/optional webshop until network dependence is audited. | **CC-BY-NC-4.0**, so keep separate from an unrestricted commercial-use data/model release. Default comparisons intentionally ignore or normalize some state. |

License claims come from the actual [ToolSandbox][ts-license],
[AppWorld public][aw-license], [AppWorld protected-material explanation][aw-readme],
[τ][tau-license], [BrowserGym][bg-license], [MiniWoB++][mw-license] and
[OpenApps][oa-license] sources. This inspection is not a complete transitive
dependency license audit. ToolSandbox also ships separate
[dependency acknowledgements][ts-ack].

## What was inspected, and what the verifier really means

### 1. ToolSandbox: cheapest useful third-party execution substrate

**Inspected:** reminder tools, execution-context serialization/history, scoring
functions, external-search wrapper, dependency manifest and license. The context
stores Polars database snapshots and supports serialization/deep copying;
reminder tools actually add, modify and remove records. They also use wall-clock
timestamps and generated UUIDs, which must be frozen or canonicalized for replay.
These are new tool/data operations, not another wording of the sensor world.
[Execution context][ts-context], [reminder implementation][ts-reminder].

**Verifier distinction:** the scoring module mixes exact columns, numerical
closeness, tool-trace comparisons and ROUGE-L text similarity. Snapshot
constraints are combined into milestone scores; minefield matches can nullify
the score. Even a deterministic similarity value is not a probability or an
exact task-completion label. For our subset, verify explicit database changes
and preservation of untouched rows/fields; keep the original similarity only as
separate metadata. [Scoring implementation][ts-eval].

**Dependencies/cost:** Python >=3.9, with old exact pins including Polars 0.20.31
and model SDKs; use an isolated environment rather than installing into the
training runtime. RapidAPI search tools make actual HTTP requests and require
`RAPID_API_KEY`. Omit these tools and replace conversational-user participation
with a declared finite local script. This is a ToolSandbox-derived subset, not
an unchanged official benchmark run. [Manifest][ts-project], [search code][ts-search],
[benchmark interface][ts-readme].

**Leakage boundary:** keep milestone definitions, minefields, complete context
snapshots and target rows verifier-only. Menus may use IDs returned by visible
queries, not IDs obtained from hidden target rows. Group all initial-state,
entity-name, tool-name and wording variants of one task program before splitting.
No install or replay smoke test was performed in this research pass.

### 2. AppWorld: highest-value harder integration

**Inspected:** public environment/configuration, evaluator, task grouping,
parallelization guide and packaging metadata. The evaluator loads start/end
model collections and invokes a task-specific evaluation program; strict task
success requires all tracked tests to pass. It also groups related task instances
by generator for scenario-level completion. Do not relabel the percentage of
passed assertions as the probability that the full task succeeded.
[Evaluator][aw-eval], [task identifiers][aw-task].

**Important limit:** app implementations and task-specific evaluators are in
protected bundles. They were **not downloaded, unpacked or inspected** here.
The public wrapper is executable-test based and contains no model judge in the
inspected evaluation path; that is not a line-by-line certification of every
bundled task or every assertion's semantic sufficiency.

**Offline operation:** unified mode uses FastAPI TestClient without actual HTTP
network communication. Use a separate process for each live world and unique
experiment/task output identities: the maintainers explicitly warn that
threads/async within one process do not isolate process-wide mocked time and
database state. Python >=3.11; FastAPI/SQLModel and the bundled apps add more setup
than the ToolSandbox slice. Public downloads are provided; no paid agent API is
required if we supply our own finite controller. Download size and CPU throughput
were not measured. [Parallelization guide][aw-parallel], [environment][aw-env],
[dependencies][aw-project], [downloader][aw-download].

**License/access:** the public portion is Apache-2.0. Maintainers describe the
protected portion as Apache-2.0 plus encrypted public redistribution of it and
derivatives; they explicitly distinguish model training/served outputs from
redistribution. Do not publish decrypted app/task artifacts or derived raw
trajectory dumps as ordinary plaintext without resolving that condition.
[Published licensing explanation][aw-readme].

**Leakage/splits:** respect train/dev/test-normal/test-challenge ownership; train
on train, tune on dev, use test only for final aggregate evaluation. The upstream
rules also prohibit hardcoded API-call agent logic for official benchmark
comparisons. A restricted scripted-continuation data adapter therefore needs
its own clearly named evaluation, not an official AppWorld leaderboard claim.
Keep `ground_truth`, solutions and evaluation programs out of model inputs.
Group custom data by generator and underlying world/entities, not merely task
instance IDs. [Development restrictions][aw-readme], [task loader][aw-task].

### 3. τ³/τ² repository: useful, but select criteria and mode from code

**Current version:** the existing `tau2-bench` repository now presents τ³; its
manifest says version 1.0.1, Python >=3.12,<3.14 and MIT. The README documents task
fixes that changed scores. Pin code **and** task files; do not mix historical
leaderboard numbers with regraded tasks. [Release notes in README][tau-readme],
[manifest][tau-project].

**Inspected:** evaluator dispatcher, environment evaluator, natural-language
assertion module, retail/telecom constructors, Gym guide and retail split file.
`ALL` may include LLM-judged natural-language assertions depending on each task's
`reward_basis`. Environment scoring instead replays tools, compares agent/user
database hashes and runs explicit environment assertions. Action matching tests
whether specified calls occurred; it is not interchangeable with final-state
correctness. [Dispatcher][tau-eval], [environment scorer][tau-env-eval],
[NL assertion evaluator][tau-nl].

**Concrete failure gates:** no evaluation criteria can return reward 1. The
environment scorer also returns 1 for missing actions/assertions and only
multiplies checks selected by the task's reward basis. Consequently, switching
an NL-only task to environment scoring can produce a vacuous pass. Golden-action
exceptions are logged and execution continues. Require a nonempty active
DB/assertion contract, successful golden replay, a failing no-op/known-wrong
trajectory where appropriate, and strict replay-output checking before accepting
any label. [Dispatcher][tau-eval], [environment scorer][tau-env-eval].

**Offline gap:** the Gym guide shows a retail solo example, but the retail
constructor explicitly rejects solo mode. The telecom constructor supports
solo policies and both toolkits. Use telecom solo, or deliberately build a
different scripted-user retail adapter. Do not assume a Gym interface makes
the standard user simulator free/offline. Voice and knowledge extras include
additional provider SDKs; no provider pricing or full offline run was verified.
[Gym guide][tau-gym], [retail code][tau-retail], [telecom code][tau-telecom].

**Leakage:** retain official task ownership and group persona/issue variants,
shared customers/orders and initial databases. Published transcripts, golden
actions, private user goals and evaluation assertions must not become features.
The inspected retail file has train/test assignments; these alone do not prove
independent mechanisms. [Split file][tau-split].

### 4. BrowserGym/MiniWoB++: cheap browser mechanics, narrower semantics

**Inspected:** BrowserGym task adapter, core dependencies/setup, and the pinned
MiniWoB checkbox task. Setup documents a fixed MiniWoB commit and local `file://`
HTML or HTTP serving. BrowserGym needs Playwright/Chromium; its demo model API is
an optional agent implementation, not a requirement for environment stepping.
[Setup][bg-miniwob], [core requirements][bg-requirements].

**Verified pitfall:** `validate` converts any positive raw reward into 1.
`click-checkboxes` scores the number of correct versus incorrect checkbox
states; a majority-correct submission can therefore count as success despite
wrong boxes. Preserve raw reward and use a separately specified exact predicate
for tasks that require all fields correct. Read page task logic individually;
the adapter is not itself the truth contract. [Adapter][bg-base],
[checkbox task][mw-checkboxes].

**Offline/leakage gates:** whitelist local assets, freeze browser and JavaScript
RNG/time, replay from seed plus action history, and assert restored state before
branching. Exclude reward/reason globals and hidden page variables from model
observations; expose only the declared rendered/DOM observation. Split by task
family/template and interaction mechanism as well as seeds. A new seed or font
does not establish a new reasoning mechanism. The finite-action adapter still
needs a bounded visible-element/argument menu; it will not measure unconstrained
browser operation. This review did not audit every page for outbound assets.

### 5. OpenApps: worthwhile separate research track, with restrictions

**Inspected:** task/state-comparison code, HTTP state probe, setup/dependencies,
frontend-vendoring notes and license. Tasks build target state and compare actual
application state without an LLM judge. But comparison removes IDs and some
fields, ignores code-editor state, normalizes text and permits geographic
tolerance; selected messenger auto-replies are excluded. A target-state match
therefore proves only that normalized contract. Use exact calendar/todo
predicates for an initial subset. [Task verifiers][oa-tasks].

**Another concrete gate:** the state probe returns an empty list after request
failure. Treat an unavailable endpoint as an infrastructure failure, never as an
empty database; otherwise an empty-target/deletion task could be mislabeled.
This is a risk inferred from the inspected helper, not an observed benchmark
failure. [State probe][oa-state].

Local frontend assets are vendored specifically to prevent silent offline UI
failures. Optional webshop setup adds Java, search indexes, data and a spaCy
model download; avoid it and maps initially. The Python manifest includes a
broad stack even for a small subset. Runtime offline feasibility still needs a
blocked-egress smoke test. [Vendoring][oa-assets], [installation][oa-install],
[dependencies][oa-project].

The license is **Attribution-NonCommercial 4.0**, not a permissive software
license. Do not silently merge this data into a commercially reusable release.
Commercial-use permission and treatment of resulting derivatives were not
established here. Group splits by task program and initial world; the inspected
task identifier hashes goal text, so it cannot by itself prevent paraphrase or
shared-state leakage. [License][oa-license], [task identity code][oa-tasks].

## Converting execution into probability targets

This is a proposed common adapter design, not a capability these projects already
provide. For visible history `h`, offered action `a`, fixed continuation `π`, and
deadline/horizon `H`, the target is the conditional distribution of terminal
outcome and remaining cost after executing `a` then `π`.

1. **Make truth explicit.** Use mutually exclusive outcomes such as correct
   mutation, no requested mutation, wrong-record mutation, or unintended
   additional mutation. Define precedence for overlapping failures or use a
   joint categorical outcome. A bare API HTTP-success flag is insufficient.
2. **Keep actions and forecasts separate.** For each action, ask an independent
   categorical outcome question and cost question. Two actions can both succeed
   with probability 0.9; their success probabilities must not sum to one.
3. **Branch from a valid state.** Restore the same initial world and observed
   action/result history before each offered action. For partial information,
   sample hidden worlds conditional on that complete visible history. Randomly
   replacing a database after an observed query is not conditional sampling.
4. **Execute labels.** Run the offered action and declared continuation; record
   terminal predicates, exceptions, side effects and total remaining calls/cost.
   Keep failed branches. Logged trajectories alone do not identify outcomes of
   actions that were never taken. Never substitute a teacher's confidence for
   an executed label. Separate infrastructure failures from task failures unless
   those faults are explicitly part of the declared environment distribution.
5. **Name the uncertainty.** A fully observed deterministic program has a
   degenerate conditional outcome. Repeating it does not manufacture stochastic
   calibration data. Any hidden-state prior, randomized continuation or injected
   fault distribution must be specified and logged. Monte Carlo reference
   frequencies have sampling error; call them exact only when enumerated.
6. **Compose utility honestly.** With additive terminal utility minus cost,
   separate marginals suffice: `Σ p(outcome) × utility − Σ p(cost) × cost`.
   Their independence is unnecessary. Use a joint target for interaction terms
   or tail-risk constraints. Benchmark call counts are not real monetary costs;
   any synthetic call penalty is an explicitly authored utility.
7. **Audit the deployed policy separately.** Fixed-continuation forecasts are
   not the final success probability of a controller that replans after each
   observation. Evaluate that controller through complete actual trajectories,
   alongside independent forecast draws. Keep related roots together for splits
   and uncertainty estimates; macro-average mechanism families.

These adapters can support many user-described questions **within declared
executable predicate families**. They cannot verify arbitrary natural-language
questions without an additional truth source. A finite menu also introduces a
candidate-generator bottleneck that should be reported independently of the
predictor's quality.

Public benchmarks may already occur in a foundation model's training corpus.
Fresh seeds and renamed entities do not rule out memorized task logic. Report
that contamination uncertainty separately from the correctness of our own
train/validation/test grouping.

## Concrete next integration and acceptance gates

**Cheap first step — ToolSandbox-derived exact-state slice.** Implement a new,
isolated data adapter for contact lookup plus reminder create/edit/delete,
including ambiguous matches, absent targets and preservation of unrelated rows.
Reuse actual upstream tools; author the finite menus, continuation, predicate
partition and costs explicitly. Do not call its output the original benchmark
score. Keep all external-search/model tools unavailable.

Start with 64 replay fixtures, then cap the data-only pilot at **512 roots and
4,096 executed branches including audits**, horizon at most eight. This caps
underlying tool calls at 32,768 before setup checks. Proposed incremental service
spend is **$0**: local CPU, no model endpoint and no extra GPU rental. This is a
resource plan, not a measured throughput claim or permission for new spending.

Acceptance gates before any model training:

- No-op, wrong-target, duplicate-operation and unrelated-row-corruption fixtures
  produce the specified outcomes; at least some tasks require multiple actions.
- Repeated reset/replay produces identical canonical state hashes and receipts;
  clocks, UUIDs and process-global context are controlled.
- Every candidate action comes only from public observations. Hidden database,
  target and validator fields never enter a serialized model input.
- No network requests or model calls occur in blocked-egress execution.
- Generator/entity families are assigned splits before collection; failures and
  abstention examples are retained; per-mechanism label balance is reported.
- Independent audit labels use new conditional executions, with their sampling
  rule and continuation disclosed. Reused labels are counted once.

**Higher-value harder step — AppWorld training-split cross-app workflows.** First
verify a small train-only set using supplied solutions and inspect those tasks'
unpacked assertion code locally. Add one-world-per-process replay and a compact
public API-result observation interface; enforce task and API-call limits far
below the permissive defaults. Only then collect alternative finite action
branches. Resolve protected-artifact packaging before publishing a dataset.
This path adds richer API composition and state consistency than UI skins, while
keeping an execution-based evaluator. It is not yet installation- or
throughput-validated here.

## Source provenance

Small public source files, split-membership metadata and GitHub repository/ref
metadata were read. No
benchmark corpus, protected bundle, model, browser or dependency stack was
downloaded/installed; no environment, model API or provider was run. Repository
refs were checked through GitHub's `git/ref/heads/main` endpoint. Pinned refs and
the inspected-file inventory are in
[executable-data-sources.json](executable-data-sources.json). A package's moving
`main`, old leaderboard score and pinned source revision are not interchangeable.

[ts-license]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/LICENSE
[ts-ack]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/ACKNOWLEDGEMENTS
[ts-context]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/common/execution_context.py
[ts-reminder]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/tools/reminder.py
[ts-eval]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/common/evaluation.py
[ts-search]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/tool_sandbox/tools/rapid_api_search_tools.py
[ts-project]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/pyproject.toml
[ts-readme]: https://github.com/apple-aiml-research/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/README.md
[aw-license]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/LICENSE
[aw-readme]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/README.md
[aw-eval]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/evaluator.py
[aw-task]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/task.py
[aw-env]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/environment.py
[aw-parallel]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/guides/parallelizing_worlds.md
[aw-project]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/pyproject.toml
[aw-download]: https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/download.py
[tau-license]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/LICENSE
[tau-readme]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/README.md
[tau-project]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/pyproject.toml
[tau-eval]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/evaluator/evaluator.py
[tau-env-eval]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/evaluator/evaluator_env.py
[tau-nl]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/evaluator/evaluator_nl_assertions.py
[tau-gym]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/gym/README.md
[tau-retail]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/domains/retail/environment.py
[tau-telecom]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/domains/telecom/environment.py
[tau-split]: https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/data/tau2/domains/retail/split_tasks.json
[bg-license]: https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/LICENSE
[bg-miniwob]: https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/miniwob/README.md
[bg-requirements]: https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/core/requirements.txt
[bg-base]: https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/miniwob/src/browsergym/miniwob/base.py
[mw-license]: https://github.com/Farama-Foundation/miniwob-plusplus/blob/7fd85d71a4b60325c6585396ec4f48377d049838/LICENSE
[mw-checkboxes]: https://github.com/Farama-Foundation/miniwob-plusplus/blob/7fd85d71a4b60325c6585396ec4f48377d049838/miniwob/html/miniwob/click-checkboxes.html
[oa-license]: https://github.com/facebookresearch/OpenApps/blob/c7e08cb269bf6a4ce370cd66c69e28c7c6b28243/LICENSE
[oa-tasks]: https://github.com/facebookresearch/OpenApps/blob/c7e08cb269bf6a4ce370cd66c69e28c7c6b28243/src/open_apps/tasks/tasks.py
[oa-state]: https://github.com/facebookresearch/OpenApps/blob/c7e08cb269bf6a4ce370cd66c69e28c7c6b28243/src/open_apps/state.py
[oa-assets]: https://github.com/facebookresearch/OpenApps/blob/c7e08cb269bf6a4ce370cd66c69e28c7c6b28243/src/open_apps/apps/assets/vendor/README.md
[oa-install]: https://github.com/facebookresearch/OpenApps/blob/c7e08cb269bf6a4ce370cd66c69e28c7c6b28243/docs/installation.md
[oa-project]: https://github.com/facebookresearch/OpenApps/blob/c7e08cb269bf6a4ce370cd66c69e28c7c6b28243/pyproject.toml
