# First shared-state decision divergence

Local gaps below are cents under the declared fixed continuation, not realized reward losses.

| Environment | Roots | Same path | Tie difference | Value-loss difference | Mean local gap, all roots | Terminal corrected | Costs corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| retry | 128 | 13 | 0 | 115 | 21.621 | 0.283 | 19.592 |
| workflow | 128 | 65 | 0 | 63 | 14.259 | 0.227 | 11.365 |

The last two columns replace one component for **all** actions, choose again, and report the new exact-menu gap. Same-path roots contribute zero local gap.

## retry: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / lookup | 27 | 0 | 25.878 | 39.884 | 1.862 |
| abstain / retry | 1 | 0 | 7.900 | 18.781 | -1.275 |
| abstain / wait | 9 | 0 | 27.400 | 38.839 | -1.026 |
| lookup / abstain | 1 | 0 | 77.700 | 90.601 | -0.812 |
| lookup / wait | 9 | 0 | 53.800 | 62.065 | 0.932 |
| retry / abstain | 2 | 0 | 44.525 | 52.673 | -0.707 |
| retry / lookup | 15 | 0 | 15.340 | 41.691 | -0.072 |
| retry / wait | 2 | 0 | 75.588 | 103.373 | -5.245 |
| wait / abstain | 17 | 0 | 7.135 | 17.261 | 1.291 |
| wait / lookup | 29 | 0 | 20.248 | 35.802 | 1.632 |
| wait / retry | 3 | 0 | 24.517 | 43.678 | 4.670 |

## workflow: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / inspect | 1 | 0 | 18.900 | 28.556 | 3.349 |
| abstain / prepare | 1 | 0 | 63.750 | 97.486 | -8.959 |
| acquire / assemble | 6 | 0 | 61.333 | 64.093 | 4.734 |
| assemble / abstain | 1 | 0 | 30.000 | 33.007 | 3.143 |
| assemble / acquire | 2 | 0 | 11.963 | 25.138 | 1.872 |
| assemble / inspect | 4 | 0 | 48.081 | 61.013 | -0.905 |
| assemble / prepare | 1 | 0 | 145.000 | 143.709 | 1.529 |
| inspect / acquire | 36 | 0 | 19.239 | 40.539 | -0.945 |
| inspect / assemble | 2 | 0 | 0.775 | 8.936 | 2.948 |
| prepare / abstain | 1 | 0 | 6.000 | 5.467 | 0.949 |
| prepare / assemble | 5 | 0 | 5.600 | 9.941 | 4.038 |
| prepare / inspect | 3 | 0 | 85.033 | 85.829 | 3.082 |

## Interpretation limits

- One first action difference per root, compared only along the shared action prefix. Exact-value ties stop comparison too.
- The exact menu maximizes expected return for one offered action plus the fixed continuation; it is not an omniscient or globally optimal planner.
- Terminal expectations are stored net expectations plus expected costs. This relies on the original net-value receipts, not an independent terminal-outcome verifier.
- Component substitutions re-rank every candidate at this one state. Their exact-menu gaps are arithmetic diagnostics, not executed adaptive-policy returns or causal return attribution.
- Roots with identical action paths remain in denominators. Roots are equally weighted within environment; macro rates and means equally weight environments.
- No confidence intervals or independent-mechanism claim. Caller-supplied checkpoint provenance is retained, not independently certified here.
- Use patterns to design fresh training-only coverage. Do not copy held-out roots, labels, or hidden tapes into training.
