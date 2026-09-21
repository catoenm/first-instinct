# First Instinct 9B: a usable release candidate

The next milestone is a downloadable decision model and a working developer
interface. Keep the nine-billion-parameter foundation and build one coherent
release dataset. Complete a bounded supervised run, then measure reinforcement
learning as an optional improvement to that same candidate. A useful supervised
release does not need to claim that the calibrated-outcome research is solved.

This plan supersedes the previous priority of adding another small environment
study before any release work. It preserves all experiments, data ownership and
the original cumulative $500 authorization. It is a release roadmap, not a claim
that the production dataset, launcher or deployment is ready.

## Product contract

Callers supply text or structured state, natural-language questions and their
own answer definitions. One model supports:

- Choice among caller-defined answers, including supplied tools or commands.
- Binary judgments and explicit outcome forecasts, with a probability of yes.
- Ordered scores using caller-defined levels and probabilities over those levels.
- Multiple independent questions about the same state.

These are the observable interface ideas described by
[TypeSafe's primitives](https://docs.typesafe.ai/primitives). They are not evidence
that we know its architecture, training data or private learning method.

The existing interface already accepts 1–32 questions and 2–36 options. Preserve
that public shape initially. The serving demo currently limits complete prompts
to 1,536 tokens. A release target of 4,096 tokens needs a matched baseline and
training/serving checks; do not silently advertise it before those pass. Reject
overlength requests instead of silently dropping evidence or choices.

The model selects supplied tools; the caller executes them. It does not have to
invent arbitrary command arguments to be useful. A dynamic proposer remains a
separate component with its own coverage and error measurements. An event forecast
must state the event and, where needed, the continuation/horizon. A choice's
probability is not the probability that its selected tool will succeed.

Keep these supported use cases visible throughout evaluation: support/intent
routing, supplied policy checks, evidence-based binary judgments, ordered
urgency/severity, selection among described tools, and verified consequences.
User-defined instructions remain the interface; these are evaluation slices, not
a fixed classifier taxonomy.

## One release dataset, with three kinds of evidence

1. **General decision competence.** Reuse the original 350,857 admitted training
   examples with their original source attribution and grouping. The raw mixture
   contained 170,935 public instruction examples, 160,000 executable reasoning
   examples and 20,000 evidence/action examples before length filtering. Preserve
   the previous evaluation partitions. A repeated visit is not a new example.
2. **Broader tool behavior.** Convert already-audited public trajectories to
   contemporaneous-history → supplied-tool decisions. The three local TOUCAN
   shards contain 35,227 trajectories, 33,961 normalized question strings and
   104,552 recorded calls. Their structural audit found 29,161 eligible first
   calls before combined deduplication, stronger argument validation and split
   assignment. These are candidates, not admitted training examples. ToolACE and
   Glaive are already pinned supplementary sources; re-audit inherited ownership
   and annotation limitations before using them. Do not start another giant
   download while this existing material is unused.
3. **Verified outcomes and decisions.** Reuse the five-mechanism pool of 2,280
   distinct forecasts, the qualified AppWorld training slice of 484 questions,
   eligible contextual shell decisions and the new retail witnesses once their
   paired-question admission passes. Keep hard annotations, acceptable answer
   sets and fractional outcome distributions as distinct target contracts.

TOUCAN is behavior imitation. Its model judges and linked tool responses do not
supply independently verified success probabilities. Do not execute recorded
external commands or contact real services. Never transform a judge score into
an outcome reward. Preserve failed/recovery behavior where its observed action
is valid for imitation; scope the claim accordingly.

The existing shards are contiguous samples from three teacher subsets. Related
questions across teachers must stay together. Their server co-occurrence graph
has a giant connected component; a random row split is not a novel-tool test.
Reserve source/tool groups first, discard cross-partition compositions, and audit
shared schemas and question lineage. If meaningful separation collapses, disclose
that result and use a separately sourced evaluation rather than weakening the
ownership rule. Existing development/reserved mechanisms must never move into
training to reach a size target.

For planning, aim for the existing general corpus plus roughly 20,000–50,000
newly admitted tool/general decision examples and the smaller verified pool.
This range is not a launch requirement or a claim that those new examples already
exist. Actual independent groups, accepted examples, rejected examples and tokens
determine the run. Cap contribution by source and underlying task so a narrow
verified family cannot dominate through repetition. Count each pool's prepared
examples and consumed presentations separately.

Complete the retail paired-question admission as one bounded component of this
dataset. The 72-variant ceiling remains; it is not the production curriculum by
itself. Telecom stays evaluation-only under its corrected digest-bound usage
manifest. Calendar, publication, phone development and payment transfer retain
their prior roles and exposure status. The latest complete [inventory](decision-source-registry-v1-results.md)
and [TOUCAN audit](toucan-data-audit.md) provide source evidence.

## Training and selection

Use Qwen/Qwen3.5-9B at revision c202236235762e1c871ad0ccb60c8ee5ba337b9a.
Start the first release continuation from the original supervised step-2742
adapter, file SHA-256
882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a.
Use the existing internal rank-16 adapters, not a new classifier. Increasing the
foundation size or performing full-parameter pretraining is not part of this run.

Before renting, freeze a source manifest, exact tokens/targets, ownership graph,
pool weights, parent hashes and a representative development evaluation. Preserve
the original checkpoint as an always-eligible baseline. Mixed hard/soft targets
need validated losses and weights; do not feed outcome distributions into an
acceptable-answer-set loss.

Run one bounded pilot first. Use one H200 to measure actual examples/tokens per
second, memory, gradients, general retention and tool selection under the intended
context lengths. Qualify exact restart and recovery behavior. The full run may use
one to four H200s only when measured throughput and the total spending deadline
support it. More GPUs change throughput, not the data or acceptance criteria.

The main supervised continuation is capped at two scheduled visits to its frozen
mixture and the allocated spending/time limit, whichever ends first. Check
development periodically; stop after two checks without an eligible improvement,
and stop immediately on numerical failure or breached retention. Record actual
presentations rather than claiming an epoch completed at a time limit. Exact
update intervals and learning settings belong in the later executable run freeze,
after the data/token census and CPU loss checks, before model scores.

The candidate must improve the new tool-decision development macro accuracy by
at least three percentage points while retaining general macro accuracy within
one point and general log loss within 0.02 of the start. No declared product slice
may lose more than three accuracy points. Report per-slice support; small or
unqualified slices cannot earn a broad capability claim. Record ordered-score
error separately. Proper outcome probability scores must not worsen on the
declared development outcome suite. Fix the exact cohorts and metrics before the
pilot; do not loosen gates to justify the main rental.

Then compare forecast-only supervision, reward-only Proximal Policy Optimization
and their combination from the same selected supervised weights, using the
qualified real-tool runtime, matched schedules and general replay. This is a
small, bounded upgrade experiment, not a requirement to call the base release
usable. Promotion requires better verified return and better outcome probability
quality on entire unseen mechanisms, with retention. Preserve the earlier joint
thresholds of +0.03 return and −0.02 expected Brier error in both paired seeds.
If the upgrade fails, retain the supervised candidate and publish the result.

## Release evaluation and serving

Known tests are regression sets, not newly untouched evidence. Freeze a fresh
source/mechanism-held-out release set before candidate selection; report its
coverage and any foundation-data contamination limits. Use separate development,
calibration and final evaluation groups. If remaining data do not support all
three, reduce the claim rather than recycling final examples for calibration.

Measure original versus candidate on useful developer requests, option-order
changes, missing evidence, changed user rules and costs, irrelevant distractors,
none/clarify choices when supported by source labels, and binary/ordered outputs.
Report accuracy, log loss, outcome Brier score and empirical probability bins with
counts. A scalar temperature may be fit on calibration data if it helps; it is not
a guarantee on arbitrary new requests. Threshold/abstention rules must be tuned
and evaluated with their coverage, not advertised from maximum probability alone.

Package versioned weights/adapters, tokenizer and formatter hashes, a model card,
a Python client, a stable HTTP endpoint and the existing simple example demo.
The current localhost server serializes requests and is not a public server.
The public runtime needs bounded concurrency/queues, authentication or rate limits,
request limits, timeouts, checkpoint identity and content-free operational metrics.

Benchmark single-question and 8-question requests, short and long contexts, with
actual end-to-end median and 95th-percentile latency, throughput and memory on the
chosen serving hardware. Initial engineering targets are a warm single short
question within two seconds and eight same-state questions within five seconds
at concurrency one; these are goals, not current service promises or Jev parity.
Measure higher concurrency separately before opening a public demo.

The shared-prefix prototype is promising but needs a broader prompt-order quality
check. On one existing Mac fixture it was 3.45 times faster than a complete batched
forward, with matching outputs; that is not a general serving guarantee. Keep the
old formatter as a control. Only ship caching or quantization after matched-input
quality checks. Test a 4-bit artifact to lower serving cost; retain the full
reference and report any answer/probability regression. Do not assume a particular
Mac mini can serve the quantized model until measured there.

## Remaining compute, not a new budget

A fresh read-only provider reconciliation found no pods and $208.71 in reported
charges. Retaining previous conservative commitments leaves **$242.64** of the
original $500 authorization. Do not release old holds solely because a provider
total is lower. Proposed maximum allocations within that remaining ceiling:

| Work | Maximum allocation |
| --- | ---: |
| Common-data pilot and throughput qualification | $25 |
| Main supervised continuation | $110 |
| Bounded reinforcement comparison | $35 |
| Final evaluation and quantized-artifact checks | $35 |
| Storage, recovery and billing reserve | $37.63 |

These are spending ceilings, not current hardware price quotes or promises of
completed training. Reconcile billing and actual rental rates again at launch.
All active rentals need provider-side deadlines and verified artifact recovery.
No recurring production hosting charge or automatic top-up is authorized by this
plan. Failed gates stop progression; money remaining is not evidence to scale.

## Immediate execution order

First build the release source registry and TOUCAN imitation converter from the
already-cached shards, with explicit ownership and rejection records. Alongside
that work, complete the small retail admission and define the product evaluation
cohorts. Then freeze the combined pack and qualify mixed-target training locally.
Launch the bounded pilot without another permission request once those gates pass.
No additional standalone game or toy-environment study is on this release path.
