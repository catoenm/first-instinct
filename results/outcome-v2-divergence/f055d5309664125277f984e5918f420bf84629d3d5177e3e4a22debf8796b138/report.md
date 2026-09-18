# First shared-state decision divergence

Local gaps below are cents under the declared fixed continuation, not realized reward losses.

| Environment | Roots | Same path | Tie difference | Value-loss difference | Mean local gap, all roots | Terminal corrected | Costs corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| retry | 128 | 13 | 0 | 115 | 25.761 | 0.322 | 24.900 |
| workflow | 128 | 62 | 0 | 66 | 13.763 | 0.314 | 12.738 |

The last two columns replace one component for **all** actions, choose again, and report the new exact-menu gap. Same-path roots contribute zero local gap.

## retry: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / lookup | 23 | 0 | 25.600 | 42.436 | 1.134 |
| abstain / retry | 1 | 0 | 7.900 | 19.101 | -1.486 |
| abstain / wait | 9 | 0 | 28.733 | 41.951 | -0.572 |
| lookup / abstain | 1 | 0 | 77.700 | 95.143 | -0.705 |
| lookup / wait | 8 | 0 | 58.862 | 68.756 | 0.660 |
| retry / abstain | 2 | 0 | 103.075 | 115.597 | 1.519 |
| retry / lookup | 19 | 0 | 25.071 | 54.317 | 0.416 |
| retry / wait | 6 | 0 | 93.400 | 112.107 | -2.179 |
| wait / abstain | 18 | 0 | 7.583 | 20.268 | 1.561 |
| wait / lookup | 25 | 0 | 17.624 | 33.241 | 1.489 |
| wait / retry | 3 | 0 | 24.517 | 40.626 | 5.055 |

## workflow: first differing action pairs

| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |
|---|---:|---:|---:|---:|---:|
| abstain / inspect | 1 | 0 | 18.900 | 31.706 | 4.077 |
| abstain / prepare | 1 | 0 | 63.750 | 104.419 | -8.635 |
| acquire / assemble | 8 | 0 | 78.344 | 85.257 | 3.452 |
| inspect / acquire | 37 | 0 | 18.764 | 40.382 | -1.509 |
| inspect / assemble | 2 | 0 | 0.775 | 31.720 | 3.128 |
| prepare / acquire | 2 | 0 | 21.700 | 30.623 | 0.809 |
| prepare / assemble | 10 | 0 | 5.100 | 16.544 | 3.235 |
| prepare / inspect | 5 | 0 | 52.420 | 56.490 | 4.142 |

## Interpretation limits

- One first action difference per root, compared only along the shared action prefix. Exact-value ties stop comparison too.
- The exact menu maximizes expected return for one offered action plus the fixed continuation; it is not an omniscient or globally optimal planner.
- Terminal expectations are stored net expectations plus expected costs. This relies on the original net-value receipts, not an independent terminal-outcome verifier.
- Component substitutions re-rank every candidate at this one state. Their exact-menu gaps are arithmetic diagnostics, not executed adaptive-policy returns or causal return attribution.
- Roots with identical action paths remain in denominators. Roots are equally weighted within environment; macro rates and means equally weight environments.
- No confidence intervals or independent-mechanism claim. Caller-supplied checkpoint provenance is retained, not independently certified here.
- Use patterns to design fresh training-only coverage. Do not copy held-out roots, labels, or hidden tapes into training.
