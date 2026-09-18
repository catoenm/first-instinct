# Outcome-v2: first shared-state divergence

The completed forecast-controller traces point mainly to errors in terminal
expectations at the first differing decision. Across six distinct
checkpoint/trace pairs, replacing terminal expectations with the stored exact
ones leaves an average local gap of **0.25–0.82 cents**. Replacing only expected
costs leaves **15.24–58.96 cents**. These are local arithmetic substitutions,
not measured improvements from executing a corrected policy.

This read-only follow-up used the existing
[first-divergence analyzer and interpretation rules](outcome-first-divergence.md).
The original selection, training outputs and test receipts were not changed.
No inference, tool/environment execution, provider access or service call ran.

All 12 original cohort roles are recorded. Ten roles have complete finalized
test traces; exact equality of adapter weights, config, pinned model identity
**and complete trace bytes** reduces those to six analyses. Every analysis has
the same 256 root/scenario/tape identities: 128 retry and 128 workflow roots.
These are repeated evaluations of the same roots, not 1,536 independent tasks.
Both hybrid-77 roles remain excluded, with best at update 40 and latest at
update 58 and no final test traces. The launcher marked hybrid-83 `timed_out`
while recording return code 0, despite its complete update-60 trainer and test
receipts. That launcher failure propagated cancellation to hybrid-77 at about
6,703 elapsed seconds. Hybrid-77's receipt says `bounded_stop`, but this was
propagated cancellation, not exhaustion of its own time allowance. The study
remains partial even though hybrid-83's finalized roles are available.

| Original role(s) | Update | First value-loss difference / 256 | Mean local gap, cents | Exact terminal substitution | Exact cost substitution |
|---|---:|---:|---:|---:|---:|
| outcome-77 best and latest | 60 | 166 | 16.272 | 0.287 | 15.241 |
| outcome-83 best and latest | 60 | 178 | 17.940 | 0.255 | 15.478 |
| reward-77 best | 40 | 246 | 58.763 | 0.750 | 58.859 |
| reward-77 latest | 60 | 241 | 58.997 | 0.753 | 58.909 |
| reward-83 best and latest | 60 | 236 | 60.284 | 0.819 | 58.961 |
| hybrid-83 best and latest | 60 | 181 | 19.762 | 0.318 | 18.819 |
| hybrid-77 best | 40 | unavailable | — | — | — |
| hybrid-77 latest | 58 | unavailable | — | — | — |

The gap is the exact-menu value of its best action minus that of the model's
action at the first state where their choices differ. The menu evaluates an
offered action followed by the declared fixed continuation. It is not a
globally optimal planner. All roots remain in the denominator; roots with
identical paths contribute zero. Means weight environments equally. No first
differences were exact-value ties in these traces. Substituting both exact
components yields zero gap throughout, as the arithmetic requires.

Workflow exposes a particularly clear pattern. Reward-only roles differ from
the exact menu on 126/128 roots, with mean local gaps of 75.54–76.72 cents.
Their largest action pairs are `submit` versus `inspect` or `acquire`: together
94–101 roots per role. Correcting only costs makes none of the 126 value-loss
roots menu-optimal. Correcting terminal expectations makes 107–109 menu-optimal,
leaving a small mean gap on the remainder. For outcome-only and hybrid-83,
the largest workflow pair is `inspect` versus `acquire` on 36–40 roots; their
workflow mean local gaps are 12.18–14.26 cents. The retry environment also
shows terminal substitutions removing much more of the local gap than cost
substitutions. Per-environment counts, action pairs and signed margin
decompositions remain in each full diagnostic.

The terminal component is recovered as stored net expectation plus expected
cost. This depends on the original net-value receipts; the analysis does not
independently reverify terminal-outcome probabilities. Each substitution ranks
all candidate actions at that one shared state. Later policy states can change,
so the reported gap reductions are neither causal return attribution nor
executed policy improvements. This analysis does not select checkpoints,
establish improvement over the starting adapter, or establish a hybrid gain
over outcome-only training. Hybrid-77 is unavailable, and hybrid has additional
forecast labels/work. Any follow-up coverage must use fresh training roots;
the held-out roots, labels and tapes here are not training data.

The [public artifact bundle](../results/outcome-v2-divergence/) contains
[every role](../results/outcome-v2-divergence/roles.json), the
[six aggregate comparisons](../results/outcome-v2-divergence/summary.json),
[common root identity checks](../results/outcome-v2-divergence/coverage-verification.json),
and [source and recovery provenance](../results/outcome-v2-divergence/recovery-provenance.json). Each
SHA-named group directory contains the unchanged analyzer's `diagnostic.json`
and `report.md`, with its caller provenance under `provenance/`. `commands.json`
records the six successful module invocations, and `artifact-hashes.json`
hashes every other bundle file.

Provenance anchors:

- Original archive: `54c35c5a98b83a1311f4ece94c52c95ea03948cf94d1980477eadc815eafd78e`.
- Recovered 294-file manifest: `33ba8bb4c887f426edcf6ca5cc292e1b57c4bb78bd1f5f9120db31b8ec9c847b`.
- Verified cohort manifest: `2bcc770c4037fcd82fe1551faab1be11b9f209ead577f5d00679f70ba3adbf56`.
- Frozen training protocol: `22d68712a79054b7f87227de7383d8ec3954d4eb649612f79823ae4c2b364daa`.
- Analysis summary: `1ea34b77afa12be045592b677fae3bdee40d0b067729e61574c5abdf5c3526ac`.
- Analysis bundle manifest: `7a9fadf78a2ce21251e97b158298e842ed632efdff13a716f19fffa2d6b0c1e0`.

The archive byte checksum, cohort content, recovered-manifest and collection
hashes were rechecked. Each used adapter, original run receipt and trace was
rehashed against the verified recovered manifest before analysis, and again
afterward. The completed recovery receipt records the pod as deleted. Only
complete eligible final test traces were analyzed; no missing role was replaced
with a validation trace or another checkpoint.
