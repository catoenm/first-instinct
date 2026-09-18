# First shared-state decision divergence

Local gaps below are cents under the declared fixed continuation, not realized reward losses.

| Environment | Roots | Same path | Tie difference | Value-loss difference | Mean local gap, all roots | Terminal corrected | Costs corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| retry | 128 | 13 | 0 | 115 | 42.263 | 0.955 | 40.583 |
| workflow | 128 | 2 | 0 | 126 | 75.732 | 0.552 | 77.235 |

The last two columns replace one component for **all** actions, choose again, and report the new exact-menu gap. Same-path roots contribute zero local gap.

## retry: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / lookup | 3 | 0 | 15.667 | 25.079 | -1.928 |
| abstain / retry | 1 | 0 | 48.571 | 55.300 | -0.568 |
| abstain / wait | 1 | 0 | 23.400 | 47.265 | -4.404 |
| lookup / abstain | 10 | 0 | 43.720 | 65.379 | 6.416 |
| lookup / wait | 5 | 0 | 50.220 | 67.264 | -0.410 |
| retry / abstain | 7 | 0 | 102.011 | 125.750 | 6.077 |
| retry / lookup | 41 | 0 | 20.371 | 50.906 | -0.525 |
| retry / wait | 24 | 0 | 117.923 | 152.331 | -4.371 |
| wait / abstain | 14 | 0 | 6.786 | 21.788 | 7.861 |
| wait / lookup | 8 | 0 | 14.662 | 29.047 | 1.404 |
| wait / retry | 1 | 0 | 10.600 | 10.965 | 2.060 |

## workflow: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / acquire | 5 | 0 | 49.270 | 88.926 | -1.003 |
| abstain / inspect | 21 | 0 | 51.290 | 82.039 | -3.364 |
| assemble / inspect | 2 | 0 | 17.750 | 26.067 | -6.139 |
| inspect / acquire | 2 | 0 | 23.500 | 60.619 | -0.730 |
| inspect / assemble | 1 | 0 | 0.150 | 5.133 | 2.080 |
| submit / acquire | 33 | 0 | 92.148 | 146.981 | -2.717 |
| submit / assemble | 1 | 0 | 104.300 | 144.197 | 6.845 |
| submit / inspect | 61 | 0 | 84.302 | 114.584 | -3.676 |

## Interpretation limits

- One first action difference per root, compared only along the shared action prefix. Exact-value ties stop comparison too.
- The exact menu maximizes expected return for one offered action plus the fixed continuation; it is not an omniscient or globally optimal planner.
- Terminal expectations are stored net expectations plus expected costs. This relies on the original net-value receipts, not an independent terminal-outcome verifier.
- Component substitutions re-rank every candidate at this one state. Their exact-menu gaps are arithmetic diagnostics, not executed adaptive-policy returns or causal return attribution.
- Roots with identical action paths remain in denominators. Roots are equally weighted within environment; macro rates and means equally weight environments.
- No confidence intervals or independent-mechanism claim. Caller-supplied checkpoint provenance is retained, not independently certified here.
- Use patterns to design fresh training-only coverage. Do not copy held-out roots, labels, or hidden tapes into training.
