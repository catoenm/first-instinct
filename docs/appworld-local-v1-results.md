# AppWorld opens a broader source, with verifier work still needed

AppWorld is installed locally at the pinned revision in the
[prospective protocol](appworld-local-v1-protocol.md). Its training inventory
contains **90 task instances from 30 generator programs**. We reserve all seven
programs involving its simulated payment application (21 instances) from training
consideration. No reserved task was executed and no model was evaluated.

Of the four small preselected candidate workflows, **two passed our complete
no-op/reference/replay contract; two are quarantined**. All four reference
solutions and their independent replays pass the upstream task assertions, with
identical saved database hashes, application request logs and assertion counts.
Passing a reference solution alone is insufficient to accept a verifier.

| Application combination | Assertions per evaluation | Calls per reference | No-op check | Status |
| --- | ---: | ---: | --- | --- |
| Music library | 2 | 5 | Clean task failure; state unchanged | Execution substrate qualified |
| Notes | 8 | 7 | Dependent-variable exception | Quarantined |
| Filesystem application | 8 | 9 | Clean task failure; state unchanged | Execution substrate qualified |
| Phone and notes | 8 | 10 | Dependent-variable exception | Quarantined |

These are four generator programs with one selected instance each, not four
whole applications proven correct. Application call counts exclude environment
initialization and internal database queries. Two successful substrate checks
do not constitute qualified decision menus or forecast training data.

## Why the negative controls matter

The upstream evaluation tracker ordinarily catches general exceptions inside
an assertion block and records failed checks. Our stricter wrapper records only
assertion failures; other exception types propagate. In two no-op evaluations,
a later assertion accesses a variable that an earlier failed check never assigned.
That raises `UnboundLocalError`. The default upstream behavior would record a
failed task, but it does not meet our stronger requirement for a clean executable
truth contract. We did not turn these exceptions into training labels.

The first wrapper attempt also requested fully unsuppressed errors, which let an
expected no-op assertion escape. That single execution and its source remain
preserved. The corrected wrapper distinguishes ordinary assertion failures from
other exceptions; three regression tests check that boundary, nonempty evaluators
and complete assertion counts. Remaining independent preselected checks continued
only after reviewing the first dependent-variable failure; no task was replaced
with an easier one.

## Actual work and the next data gate

The main qualification attempted 12 isolated worlds: four no-ops, four reference
executions and four independent reference replays. The eight reference executions
made **62 simulated application calls**. Including the earlier wrapper failure,
there were 13 world attempts. Every accepted reference replay was exact. No real
messages or financial operations occurred, and outgoing socket connections were
blocked during execution.

There were **zero model calls, GPU rentals, optimizer steps, prepared training
questions or consumed training questions**. The
[aggregate receipt](../results/appworld-local-v1/summary.json) separates accepted
substrates from failed attempts. Protected task material and raw derived receipts
remain private locally, consistent with the upstream redistribution condition.

Next, add independent state predicates and wrong-target/collateral-change controls
before accepting additional programs. Quarantined programs remain excluded until
their contracts are repaired and requalified prospectively. Build public-observation
menus and branch alternatives only after that gate, keeping private reference
arguments out of model inputs. The [broader supervised plan](broader-supervision-next.md)
uses this source alongside the existing qualified environments, with the same 9B
foundation and a bounded supervised pilot before additional reinforcement learning.

The [official AppWorld project](https://github.com/StonyBrookNLP/appworld) provides
the app implementations, task generators and evaluation framework. This restricted
qualification is not an official benchmark run or evidence of Jev-like performance.
