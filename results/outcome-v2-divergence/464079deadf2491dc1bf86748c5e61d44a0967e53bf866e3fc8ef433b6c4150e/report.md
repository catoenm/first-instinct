# First shared-state decision divergence

Local gaps below are cents under the declared fixed continuation, not realized reward losses.

| Environment | Roots | Same path | Tie difference | Value-loss difference | Mean local gap, all roots | Terminal corrected | Costs corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| retry | 128 | 20 | 0 | 108 | 20.362 | 0.362 | 19.592 |
| workflow | 128 | 70 | 0 | 58 | 12.182 | 0.212 | 10.889 |

The last two columns replace one component for **all** actions, choose again, and report the new exact-menu gap. Same-path roots contribute zero local gap.

## retry: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / lookup | 32 | 0 | 23.688 | 36.526 | 2.005 |
| abstain / retry | 2 | 0 | 24.700 | 38.926 | -0.003 |
| abstain / wait | 10 | 0 | 20.970 | 33.686 | -0.418 |
| lookup / abstain | 1 | 0 | 77.700 | 93.215 | -0.697 |
| lookup / wait | 8 | 0 | 44.212 | 53.406 | 0.389 |
| retry / abstain | 2 | 0 | 44.525 | 54.299 | -0.732 |
| retry / lookup | 19 | 0 | 24.576 | 53.971 | 0.226 |
| retry / wait | 1 | 0 | 51.425 | 105.704 | -7.346 |
| wait / abstain | 14 | 0 | 6.336 | 17.067 | 0.896 |
| wait / lookup | 18 | 0 | 23.817 | 38.413 | 1.003 |
| wait / retry | 1 | 0 | 33.050 | 72.940 | 3.860 |

## workflow: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / inspect | 1 | 0 | 18.900 | 19.785 | 3.000 |
| abstain / prepare | 1 | 0 | 63.750 | 103.234 | -8.186 |
| acquire / assemble | 8 | 0 | 85.312 | 92.940 | 3.492 |
| inspect / acquire | 40 | 0 | 19.041 | 43.498 | -0.813 |
| inspect / assemble | 2 | 0 | 0.775 | 16.436 | 1.870 |
| prepare / assemble | 6 | 0 | 5.167 | 15.331 | 4.285 |

## Interpretation limits

- One first action difference per root, compared only along the shared action prefix. Exact-value ties stop comparison too.
- The exact menu maximizes expected return for one offered action plus the fixed continuation; it is not an omniscient or globally optimal planner.
- Terminal expectations are stored net expectations plus expected costs. This relies on the original net-value receipts, not an independent terminal-outcome verifier.
- Component substitutions re-rank every candidate at this one state. Their exact-menu gaps are arithmetic diagnostics, not executed adaptive-policy returns or causal return attribution.
- Roots with identical action paths remain in denominators. Roots are equally weighted within environment; macro rates and means equally weight environments.
- No confidence intervals or independent-mechanism claim. Caller-supplied checkpoint provenance is retained, not independently certified here.
- Use patterns to design fresh training-only coverage. Do not copy held-out roots, labels, or hidden tapes into training.
