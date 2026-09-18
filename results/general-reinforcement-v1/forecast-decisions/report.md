# Post-hoc forecast-to-action diagnostic

This is a fixed **stop now** controller, not the learned sequential policy. It uses saved event forecasts to choose affirm, deny, or abstain under the declared payoffs. It never buys another check; past check costs are sunk. This analysis does not select or retrain a model.

Each sampled world contributes four equally weighted evidence masks. Rewards integrate the latent event exactly conditional on that evidence; they do not integrate every possible sensor report or follow policy visitation frequencies. The terminal-only oracle knows the exact posterior, not the realized event.

| Run / split / checkpoint | Mean terminal reward | Mean terminal regret | Oracle-action mismatches / states | Affirm / deny / abstain |
|---|---:|---:|---:|---:|
| rl-s47-reward / validation / baseline | 0.467894 | 0.006065 | 14 / 256 | 109 / 109 / 38 |
| rl-s47-reward / validation / best (supervised update zero) | 0.467894 | 0.006065 | 14 / 256 | 109 / 109 / 38 |
| rl-s47-reward / validation / latest | 0.459120 | 0.014839 | 24 / 256 | 123 / 106 / 27 |
| rl-s47-reward / test / baseline | 0.485001 | 0.012317 | 18 / 256 | 102 / 122 / 32 |
| rl-s47-reward / test / best (supervised update zero) | 0.485001 | 0.012317 | 18 / 256 | 102 / 122 / 32 |
| rl-s47-reward / test / latest | 0.474054 | 0.023265 | 26 / 256 | 113 / 123 / 20 |
| rl-s47-reward / shift / baseline | 0.397549 | 0.036135 | 23 / 256 | 118 / 131 / 7 |
| rl-s47-reward / shift / best (supervised update zero) | 0.397549 | 0.036135 | 23 / 256 | 118 / 131 / 7 |
| rl-s47-reward / shift / latest | 0.335884 | 0.097800 | 38 / 256 | 127 / 127 / 2 |
| rl-s47-reward / new_domain / baseline | 0.526963 | 0.010868 | 9 / 256 | 126 / 107 / 23 |
| rl-s47-reward / new_domain / best (supervised update zero) | 0.526963 | 0.010868 | 9 / 256 | 126 / 107 / 23 |
| rl-s47-reward / new_domain / latest | 0.520151 | 0.017681 | 20 / 256 | 140 / 104 / 12 |
| rl-s47-hybrid / validation / baseline | 0.467894 | 0.006065 | 14 / 256 | 109 / 109 / 38 |
| rl-s47-hybrid / validation / best (supervised update zero) | 0.467894 | 0.006065 | 14 / 256 | 109 / 109 / 38 |
| rl-s47-hybrid / validation / latest | 0.468394 | 0.005566 | 15 / 256 | 108 / 105 / 43 |
| rl-s47-hybrid / test / baseline | 0.485001 | 0.012317 | 18 / 256 | 102 / 122 / 32 |
| rl-s47-hybrid / test / best (supervised update zero) | 0.485001 | 0.012317 | 18 / 256 | 102 / 122 / 32 |
| rl-s47-hybrid / test / latest | 0.475171 | 0.022147 | 24 / 256 | 98 / 119 / 39 |
| rl-s47-hybrid / shift / baseline | 0.397549 | 0.036135 | 23 / 256 | 118 / 131 / 7 |
| rl-s47-hybrid / shift / best (supervised update zero) | 0.397549 | 0.036135 | 23 / 256 | 118 / 131 / 7 |
| rl-s47-hybrid / shift / latest | 0.404471 | 0.029212 | 25 / 256 | 119 / 128 / 9 |
| rl-s47-hybrid / new_domain / baseline | 0.526963 | 0.010868 | 9 / 256 | 126 / 107 / 23 |
| rl-s47-hybrid / new_domain / best (supervised update zero) | 0.526963 | 0.010868 | 9 / 256 | 126 / 107 / 23 |
| rl-s47-hybrid / new_domain / latest | 0.525487 | 0.012345 | 15 / 256 | 129 / 101 / 26 |
| rl-s53-reward / validation / baseline | 0.467894 | 0.006065 | 14 / 256 | 109 / 109 / 38 |
| rl-s53-reward / validation / best | 0.464368 | 0.009592 | 22 / 256 | 114 / 109 / 33 |
| rl-s53-reward / validation / latest | 0.464368 | 0.009592 | 22 / 256 | 114 / 109 / 33 |
| rl-s53-reward / test / baseline | 0.485001 | 0.012317 | 18 / 256 | 102 / 122 / 32 |
| rl-s53-reward / test / best | 0.482385 | 0.014933 | 20 / 256 | 104 / 131 / 21 |
| rl-s53-reward / test / latest | 0.482385 | 0.014933 | 20 / 256 | 104 / 131 / 21 |
| rl-s53-reward / shift / baseline | 0.397549 | 0.036135 | 23 / 256 | 118 / 131 / 7 |
| rl-s53-reward / shift / best | 0.349404 | 0.084280 | 32 / 256 | 119 / 132 / 5 |
| rl-s53-reward / shift / latest | 0.349404 | 0.084280 | 32 / 256 | 119 / 132 / 5 |
| rl-s53-reward / new_domain / baseline | 0.526963 | 0.010868 | 9 / 256 | 126 / 107 / 23 |
| rl-s53-reward / new_domain / best | 0.528943 | 0.008888 | 12 / 256 | 130 / 108 / 18 |
| rl-s53-reward / new_domain / latest | 0.528943 | 0.008888 | 12 / 256 | 130 / 108 / 18 |
| rl-s53-hybrid / validation / baseline | 0.467894 | 0.006065 | 14 / 256 | 109 / 109 / 38 |
| rl-s53-hybrid / validation / best (supervised update zero) | 0.467894 | 0.006065 | 14 / 256 | 109 / 109 / 38 |
| rl-s53-hybrid / validation / latest | 0.467802 | 0.006157 | 13 / 256 | 105 / 108 / 43 |
| rl-s53-hybrid / test / baseline | 0.485001 | 0.012317 | 18 / 256 | 102 / 122 / 32 |
| rl-s53-hybrid / test / best (supervised update zero) | 0.485001 | 0.012317 | 18 / 256 | 102 / 122 / 32 |
| rl-s53-hybrid / test / latest | 0.486435 | 0.010883 | 15 / 256 | 95 / 128 / 33 |
| rl-s53-hybrid / shift / baseline | 0.397549 | 0.036135 | 23 / 256 | 118 / 131 / 7 |
| rl-s53-hybrid / shift / best (supervised update zero) | 0.397549 | 0.036135 | 23 / 256 | 118 / 131 / 7 |
| rl-s53-hybrid / shift / latest | 0.401232 | 0.032452 | 22 / 256 | 114 / 131 / 11 |
| rl-s53-hybrid / new_domain / baseline | 0.526963 | 0.010868 | 9 / 256 | 126 / 107 / 23 |
| rl-s53-hybrid / new_domain / best (supervised update zero) | 0.526963 | 0.010868 | 9 / 256 | 126 / 107 / 23 |
| rl-s53-hybrid / new_domain / latest | 0.523900 | 0.013931 | 13 / 256 | 122 / 111 / 23 |

| Run / split / comparison | Mean regret change [95% world-bootstrap interval] | Actions changed |
|---|---:|---:|
| rl-s47-reward / validation / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-reward / validation / baseline_to_latest | +0.008774 [+0.001518, +0.017085] | 20 |
| rl-s47-reward / test / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-reward / test / baseline_to_latest | +0.010948 [+0.002790, +0.021912] | 15 |
| rl-s47-reward / shift / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-reward / shift / baseline_to_latest | +0.061665 [+0.024414, +0.109980] | 28 |
| rl-s47-reward / new_domain / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-reward / new_domain / baseline_to_latest | +0.006813 [-0.000100, +0.013853] | 15 |
| rl-s47-hybrid / validation / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-hybrid / validation / baseline_to_latest | -0.000500 [-0.005490, +0.004223] | 11 |
| rl-s47-hybrid / test / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-hybrid / test / baseline_to_latest | +0.009830 [+0.002242, +0.019728] | 15 |
| rl-s47-hybrid / shift / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-hybrid / shift / baseline_to_latest | -0.006922 [-0.034381, +0.011988] | 12 |
| rl-s47-hybrid / new_domain / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s47-hybrid / new_domain / baseline_to_latest | +0.001477 [-0.004687, +0.007173] | 14 |
| rl-s53-reward / validation / baseline_to_best | +0.003526 [-0.003114, +0.011326] | 18 |
| rl-s53-reward / validation / baseline_to_latest | +0.003526 [-0.003114, +0.011326] | 18 |
| rl-s53-reward / test / baseline_to_best | +0.002616 [-0.004889, +0.010449] | 16 |
| rl-s53-reward / test / baseline_to_latest | +0.002616 [-0.004889, +0.010449] | 16 |
| rl-s53-reward / shift / baseline_to_best | +0.048145 [+0.013361, +0.093500] | 24 |
| rl-s53-reward / shift / baseline_to_latest | +0.048145 [+0.013361, +0.093500] | 24 |
| rl-s53-reward / new_domain / baseline_to_best | -0.001980 [-0.006645, +0.001774] | 11 |
| rl-s53-reward / new_domain / baseline_to_latest | -0.001980 [-0.006645, +0.001774] | 11 |
| rl-s53-hybrid / validation / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s53-hybrid / validation / baseline_to_latest | +0.000092 [-0.001545, +0.001469] | 7 |
| rl-s53-hybrid / test / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s53-hybrid / test / baseline_to_latest | -0.001434 [-0.006374, +0.003747] | 11 |
| rl-s53-hybrid / shift / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s53-hybrid / shift / baseline_to_latest | -0.003683 [-0.011145, +0.002578] | 11 |
| rl-s53-hybrid / new_domain / baseline_to_best | +0.000000 [+0.000000, +0.000000] | 0 |
| rl-s53-hybrid / new_domain / baseline_to_latest | +0.003063 [+0.000530, +0.006325] | 6 |

Changes are checkpoint minus baseline; negative regret change is better. Whole-world resampling keeps all four correlated states together. Intervals cover sampled-world variation only, not training-seed uncertainty or multiple-comparison corrections.

Ties deterministically prefer affirm, then deny, then abstain. Oracle-action mismatch counts use that same rule; strictly-suboptimal counts in report.json require regret greater than 1e-12. All named splits, input hashes, and reconstruction checks are recorded; no incomplete subset is silently accepted.
