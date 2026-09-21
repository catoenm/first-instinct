# The first combined release training corpus

The private release pack now contains **371,278 training questions and
133,796,202 input tokens**. Its saved-data audit matched every question to the
frozen source and verified its target contract and inherited usage. This is
prepared data; new training presentations and optimizer updates remain zero.

| Training source | Eligible questions | Interpretation |
| :--- | ---: | :--- |
| Original general corpus | 350,857 | Previously trained examples, available for replay |
| New TOUCAN tool choices | 17,786 | Teacher-action imitation across 283 server groups |
| Existing verified mechanism pool | 2,280 | Execution-derived outcome distributions |
| Existing AppWorld pool, within the context limit | 288 | 174 forecasts and 114 decisions |
| Newly admitted retail pairs | 67 | Decisions, explicit-horizon forecasts and inspection value |
| **Total** | **371,278** | **368,757 acceptable-set and 2,521 distribution targets** |

The 2,521 distribution targets include the retail protocol's decision/tie targets;
they are not all independent probabilistic forecasts. A question is a distinct
token input. The pack preserves source references and groups; neither question
count nor group count is a count of independent real-world tasks.

The AppWorld source has 484 training questions, but **196 exceed 4,096 tokens**.
They remain in their original corpus and are excluded whole from this release
pack. No history, procedure or scope requirement was cropped. Another 75 AppWorld
development questions exceed the same bound, for 271 total exclusions across
all roles. Longer-context learning and serving remain a separate experiment.

| Evaluation role | Questions | Status |
| :--- | ---: | :--- |
| Development | 16,764 | Existing general/verified/AppWorld development plus new tool development |
| Reserved tool transfer | 2,431 | Held-out tool/server groups; no model evaluated in this stage |
| Known general regression | 24,325 | Previously exposed tests/challenges |
| Known calendar transfer | 576 | Previously evaluated mechanism |
| Publication diagnostic | 728 | Previously exposed diagnostic |

These roles are preserved explicitly. Known regression and transfer are not
advertised as untouched evaluation. Telecom remains outside the learning pack
under its corrected evaluation-only manifest. Protected application contents
remain private, as do the combined token rows and source witnesses.

The independent saved-pack audit checks **416,373 source rows**: 416,102 retained
across all roles and 271 rejected whole. It rechecks exact token sequences, option
alignment, source hashes, source roles, target contracts, lineage and counts.
There were no exact duplicates to collapse, no conflicting token inputs and no
inherited source groups crossing roles. These exact checks do not establish
semantic independence across every public source or absence from foundation
pretraining.

The first-action teacher labels remain acceptable-action annotations. Verified
outcome probabilities remain distributions. Separate local loss checks verify
that those targets have different objectives and that legitimate uncertainty
does not become an arbitrary hard class. Five assembly fixtures and two saved-row
audit fixtures pass, in addition to the earlier source and objective checks.

**No GPU has been rented for this pack, and no model has consumed it.** The next
gate is a prospective bounded pilot: fix exact sampling/task weights, development
cohorts, original step-2742 parent hashes, context/padding checks, restart and
recovery behavior, and the remaining-budget deadline. Only then launch training.
The original demo and checkpoint remain unchanged.

See the [protocol](release-mixture-v1-protocol.md),
[aggregate census](../results/release-mixture-v1/summary.json),
[saved-data admission](../results/release-mixture-v1/data-admission.json),
[assembler](../release_lab/mixture.py) and
[independent auditor](../release_lab/mixture_audit.py).
