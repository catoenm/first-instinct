# First shared-state decision divergence

Local gaps below are cents under the declared fixed continuation, not realized reward losses.

| Environment | Roots | Same path | Tie difference | Value-loss difference | Mean local gap, all roots | Terminal corrected | Costs corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| retry | 128 | 8 | 0 | 120 | 40.802 | 0.983 | 39.724 |
| workflow | 128 | 2 | 0 | 126 | 76.724 | 0.516 | 77.994 |

The last two columns replace one component for **all** actions, choose again, and report the new exact-menu gap. Same-path roots contribute zero local gap.

## retry: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / lookup | 7 | 0 | 15.086 | 28.247 | -2.765 |
| abstain / retry | 1 | 0 | 48.571 | 50.662 | -0.572 |
| abstain / wait | 1 | 0 | 23.400 | 46.810 | -4.371 |
| lookup / abstain | 7 | 0 | 43.986 | 56.827 | 7.854 |
| lookup / wait | 6 | 0 | 50.400 | 65.065 | -0.403 |
| retry / abstain | 7 | 0 | 113.654 | 134.180 | 6.054 |
| retry / lookup | 39 | 0 | 18.312 | 46.045 | -0.721 |
| retry / wait | 22 | 0 | 117.683 | 147.207 | -4.471 |
| wait / abstain | 16 | 0 | 7.450 | 16.241 | 8.962 |
| wait / lookup | 13 | 0 | 14.131 | 25.906 | 1.688 |
| wait / retry | 1 | 0 | 33.050 | 29.072 | 7.181 |

## workflow: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / acquire | 4 | 0 | 43.888 | 78.288 | 1.574 |
| abstain / inspect | 16 | 0 | 45.741 | 77.723 | -4.696 |
| assemble / inspect | 1 | 0 | 21.800 | 33.781 | -10.682 |
| inspect / acquire | 1 | 0 | 18.600 | 47.861 | -1.166 |
| inspect / assemble | 1 | 0 | 0.150 | 1.750 | 1.961 |
| prepare / acquire | 1 | 0 | 18.850 | 35.267 | 0.075 |
| submit / acquire | 34 | 0 | 91.948 | 145.245 | -3.397 |
| submit / assemble | 1 | 0 | 104.300 | 139.744 | 6.163 |
| submit / inspect | 67 | 0 | 83.931 | 114.453 | -3.800 |

## Interpretation limits

- One first action difference per root, compared only along the shared action prefix. Exact-value ties stop comparison too.
- The exact menu maximizes expected return for one offered action plus the fixed continuation; it is not an omniscient or globally optimal planner.
- Terminal expectations are stored net expectations plus expected costs. This relies on the original net-value receipts, not an independent terminal-outcome verifier.
- Component substitutions re-rank every candidate at this one state. Their exact-menu gaps are arithmetic diagnostics, not executed adaptive-policy returns or causal return attribution.
- Roots with identical action paths remain in denominators. Roots are equally weighted within environment; macro rates and means equally weight environments.
- No confidence intervals or independent-mechanism claim. Caller-supplied checkpoint provenance is retained, not independently certified here.
- Use patterns to design fresh training-only coverage. Do not copy held-out roots, labels, or hidden tapes into training.
