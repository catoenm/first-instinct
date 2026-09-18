# First shared-state decision divergence

Local gaps below are cents under the declared fixed continuation, not realized reward losses.

| Environment | Roots | Same path | Tie difference | Value-loss difference | Mean local gap, all roots | Terminal corrected | Costs corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| retry | 128 | 18 | 0 | 110 | 45.031 | 0.892 | 42.386 |
| workflow | 128 | 2 | 0 | 126 | 75.537 | 0.746 | 75.537 |

The last two columns replace one component for **all** actions, choose again, and report the new exact-menu gap. Same-path roots contribute zero local gap.

## retry: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / lookup | 3 | 0 | 15.733 | 37.867 | -3.338 |
| abstain / wait | 2 | 0 | 22.400 | 52.756 | -5.360 |
| lookup / abstain | 7 | 0 | 43.571 | 67.784 | 8.486 |
| lookup / wait | 2 | 0 | 75.100 | 87.050 | -1.760 |
| retry / abstain | 6 | 0 | 150.183 | 174.550 | 5.911 |
| retry / lookup | 42 | 0 | 30.180 | 57.541 | -1.174 |
| retry / wait | 25 | 0 | 111.808 | 137.794 | -4.476 |
| wait / abstain | 13 | 0 | 8.608 | 26.393 | 9.842 |
| wait / lookup | 9 | 0 | 11.989 | 31.894 | 3.454 |
| wait / retry | 1 | 0 | 33.050 | 41.602 | 7.199 |

## workflow: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / acquire | 3 | 0 | 44.867 | 85.813 | -1.062 |
| abstain / assemble | 1 | 0 | 134.000 | 140.565 | -1.859 |
| abstain / inspect | 20 | 0 | 48.938 | 80.621 | -7.407 |
| assemble / inspect | 2 | 0 | 37.750 | 40.502 | -1.232 |
| inspect / acquire | 1 | 0 | 18.600 | 54.982 | -0.996 |
| inspect / assemble | 1 | 0 | 0.150 | 2.411 | 1.758 |
| prepare / acquire | 1 | 0 | 18.850 | 33.418 | 0.350 |
| submit / abstain | 1 | 0 | 5.000 | 7.085 | 0.000 |
| submit / acquire | 35 | 0 | 90.548 | 142.382 | -4.262 |
| submit / assemble | 1 | 0 | 104.300 | 136.920 | 3.083 |
| submit / inspect | 59 | 0 | 83.182 | 114.386 | -6.133 |
| submit / prepare | 1 | 0 | 122.000 | 161.767 | -0.511 |

## Interpretation limits

- One first action difference per root, compared only along the shared action prefix. Exact-value ties stop comparison too.
- The exact menu maximizes expected return for one offered action plus the fixed continuation; it is not an omniscient or globally optimal planner.
- Terminal expectations are stored net expectations plus expected costs. This relies on the original net-value receipts, not an independent terminal-outcome verifier.
- Component substitutions re-rank every candidate at this one state. Their exact-menu gaps are arithmetic diagnostics, not executed adaptive-policy returns or causal return attribution.
- Roots with identical action paths remain in denominators. Roots are equally weighted within environment; macro rates and means equally weight environments.
- No confidence intervals or independent-mechanism claim. Caller-supplied checkpoint provenance is retained, not independently certified here.
- Use patterns to design fresh training-only coverage. Do not copy held-out roots, labels, or hidden tapes into training.
