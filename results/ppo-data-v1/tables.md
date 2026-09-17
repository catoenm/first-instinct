# Every method, width, training-data regime, and seed

Five-seed means, sample standard deviations and full ranges. Paired differences use the same seed and test states. Negative Brier/cost differences are improvements. These ranges are not confidence intervals.

## in_distribution

| Recipe | Brier mean [min, max] | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| ppo-accuracy-narrow-w128 | 0.215629 [0.213950, 0.216870] | 0.244099 | 77.475% | 0.049788 |
| ppo-accuracy-narrow-w128-temperature | 0.158935 [0.158184, 0.159285] | 0.053656 | 77.475% | 0.001972 |
| ppo-accuracy-narrow-w32 | 0.212146 [0.210916, 0.212742] | 0.236860 | 77.507% | 0.046965 |
| ppo-accuracy-narrow-w32-temperature | 0.157891 [0.157681, 0.158055] | 0.042985 | 77.507% | 0.001619 |
| ppo-forecast-expanded-w128 | 0.162217 [0.159016, 0.164426] | 0.077546 | 76.576% | 0.003071 |
| ppo-forecast-narrow-w128 | 0.169432 [0.167878, 0.171374] | 0.115614 | 75.017% | 0.003933 |
| ppo-forecast-narrow-w32 | 0.165662 [0.160638, 0.175137] | 0.094588 | 76.139% | 0.005056 |
| reinforce-accuracy-narrow-w128 | 0.214154 [0.213270, 0.214963] | 0.241062 | 77.486% | 0.048607 |
| reinforce-accuracy-narrow-w128-temperature | 0.158931 [0.158525, 0.159595] | 0.053654 | 77.486% | 0.001982 |
| reinforce-accuracy-narrow-w32 | 0.209542 [0.209323, 0.209649] | 0.231302 | 77.491% | 0.044751 |
| reinforce-accuracy-narrow-w32-temperature | 0.158193 [0.157978, 0.158376] | 0.046359 | 77.491% | 0.001714 |
| reinforce-forecast-expanded-w128 | 0.161649 [0.159687, 0.164085] | 0.074025 | 76.496% | 0.002957 |
| reinforce-forecast-narrow-w128 | 0.169924 [0.167954, 0.171562] | 0.117697 | 75.223% | 0.004940 |
| reinforce-forecast-narrow-w32 | 0.168298 [0.165287, 0.172124] | 0.110262 | 76.092% | 0.006118 |
| supervised-labels-expanded-w128 | 0.156372 [0.156198, 0.156498] | 0.017960 | 77.469% | 0.000119 |
| supervised-labels-narrow-w128 | 0.156213 [0.156179, 0.156266] | 0.013067 | 77.495% | 0.000076 |
| supervised-labels-narrow-w32 | 0.156235 [0.156203, 0.156252] | 0.013917 | 77.490% | 0.000079 |
| supervised_continue-labels-narrow-w128 | 0.156247 [0.156184, 0.156275] | 0.014290 | 77.484% | 0.000075 |
| supervised_continue-labels-narrow-w32 | 0.156252 [0.156180, 0.156364] | 0.014329 | 77.482% | 0.000077 |

### Individual runs

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised-labels-narrow-w32-s11 | 0.156203 | 0.012733 | 77.494% | 0.000070 |
| supervised-labels-narrow-w32-s23 | 0.156252 | 0.014526 | 77.484% | 0.000080 |
| supervised-labels-narrow-w32-s37 | 0.156226 | 0.013612 | 77.498% | 0.000074 |
| supervised-labels-narrow-w32-s53 | 0.156245 | 0.014275 | 77.488% | 0.000083 |
| supervised-labels-narrow-w32-s71 | 0.156250 | 0.014440 | 77.486% | 0.000086 |
| reinforce-forecast-narrow-w32-s11 | 0.168414 | 0.111234 | 76.398% | 0.007003 |
| reinforce-forecast-narrow-w32-s23 | 0.172124 | 0.126820 | 76.199% | 0.007466 |
| reinforce-forecast-narrow-w32-s37 | 0.167218 | 0.105721 | 75.206% | 0.005669 |
| reinforce-forecast-narrow-w32-s53 | 0.165287 | 0.096155 | 76.235% | 0.003487 |
| reinforce-forecast-narrow-w32-s71 | 0.168447 | 0.111382 | 76.421% | 0.006965 |
| ppo-forecast-narrow-w32-s11 | 0.167991 | 0.109315 | 76.161% | 0.007052 |
| ppo-forecast-narrow-w32-s23 | 0.175137 | 0.138190 | 76.190% | 0.009666 |
| ppo-forecast-narrow-w32-s37 | 0.162939 | 0.083056 | 75.900% | 0.003446 |
| ppo-forecast-narrow-w32-s53 | 0.160638 | 0.067800 | 76.223% | 0.002310 |
| ppo-forecast-narrow-w32-s71 | 0.161603 | 0.074578 | 76.222% | 0.002806 |
| supervised-labels-narrow-w128-s11 | 0.156188 | 0.012106 | 77.492% | 0.000075 |
| supervised-labels-narrow-w128-s23 | 0.156266 | 0.015005 | 77.485% | 0.000075 |
| supervised-labels-narrow-w128-s37 | 0.156203 | 0.012730 | 77.494% | 0.000071 |
| supervised-labels-narrow-w128-s53 | 0.156179 | 0.011729 | 77.504% | 0.000076 |
| supervised-labels-narrow-w128-s71 | 0.156230 | 0.013765 | 77.500% | 0.000084 |
| reinforce-forecast-narrow-w128-s11 | 0.171028 | 0.122423 | 75.017% | 0.004918 |
| reinforce-forecast-narrow-w128-s23 | 0.171562 | 0.124584 | 75.017% | 0.004213 |
| reinforce-forecast-narrow-w128-s37 | 0.169209 | 0.114751 | 75.017% | 0.004193 |
| reinforce-forecast-narrow-w128-s53 | 0.167954 | 0.109149 | 76.047% | 0.007113 |
| reinforce-forecast-narrow-w128-s71 | 0.169866 | 0.117579 | 75.017% | 0.004265 |
| ppo-forecast-narrow-w128-s11 | 0.167878 | 0.108797 | 75.017% | 0.004209 |
| ppo-forecast-narrow-w128-s23 | 0.171374 | 0.123825 | 75.017% | 0.003387 |
| ppo-forecast-narrow-w128-s37 | 0.169284 | 0.115076 | 75.017% | 0.003999 |
| ppo-forecast-narrow-w128-s53 | 0.169737 | 0.117032 | 75.017% | 0.004181 |
| ppo-forecast-narrow-w128-s71 | 0.168887 | 0.113338 | 75.017% | 0.003888 |
| supervised-labels-expanded-w128-s11 | 0.156198 | 0.012520 | 77.512% | 0.000086 |
| supervised-labels-expanded-w128-s23 | 0.156391 | 0.018707 | 77.495% | 0.000138 |
| supervised-labels-expanded-w128-s37 | 0.156498 | 0.021380 | 77.435% | 0.000137 |
| supervised-labels-expanded-w128-s53 | 0.156415 | 0.019326 | 77.449% | 0.000116 |
| supervised-labels-expanded-w128-s71 | 0.156360 | 0.017864 | 77.455% | 0.000118 |
| reinforce-forecast-expanded-w128-s11 | 0.164085 | 0.089686 | 76.962% | 0.004152 |
| reinforce-forecast-expanded-w128-s23 | 0.162961 | 0.083185 | 75.888% | 0.003276 |
| reinforce-forecast-expanded-w128-s37 | 0.161517 | 0.074001 | 77.140% | 0.002981 |
| reinforce-forecast-expanded-w128-s53 | 0.159687 | 0.060379 | 76.209% | 0.002150 |
| reinforce-forecast-expanded-w128-s71 | 0.159995 | 0.062877 | 76.284% | 0.002224 |
| ppo-forecast-expanded-w128-s11 | 0.164426 | 0.091567 | 76.654% | 0.004214 |
| ppo-forecast-expanded-w128-s23 | 0.162684 | 0.081506 | 76.240% | 0.003253 |
| ppo-forecast-expanded-w128-s37 | 0.163358 | 0.085539 | 76.648% | 0.003510 |
| ppo-forecast-expanded-w128-s53 | 0.159016 | 0.054541 | 77.348% | 0.001383 |
| ppo-forecast-expanded-w128-s71 | 0.161603 | 0.074579 | 75.990% | 0.002995 |
| supervised_continue-labels-narrow-w32-s11 | 0.156185 | 0.012001 | 77.499% | 0.000063 |
| supervised_continue-labels-narrow-w32-s23 | 0.156180 | 0.011779 | 77.495% | 0.000063 |
| supervised_continue-labels-narrow-w32-s37 | 0.156285 | 0.015606 | 77.476% | 0.000081 |
| supervised_continue-labels-narrow-w32-s53 | 0.156245 | 0.014275 | 77.488% | 0.000083 |
| supervised_continue-labels-narrow-w32-s71 | 0.156364 | 0.017981 | 77.455% | 0.000094 |
| reinforce-accuracy-narrow-w32-s11 | 0.209323 | 0.230830 | 77.497% | 0.044618 |
| reinforce-accuracy-narrow-w32-s23 | 0.209545 | 0.231309 | 77.498% | 0.044785 |
| reinforce-accuracy-narrow-w32-s37 | 0.209649 | 0.231534 | 77.503% | 0.044864 |
| reinforce-accuracy-narrow-w32-s53 | 0.209578 | 0.231380 | 77.480% | 0.044727 |
| reinforce-accuracy-narrow-w32-s71 | 0.209613 | 0.231456 | 77.476% | 0.044759 |
| ppo-accuracy-narrow-w32-s11 | 0.212715 | 0.238063 | 77.502% | 0.047436 |
| ppo-accuracy-narrow-w32-s23 | 0.212365 | 0.237327 | 77.500% | 0.047127 |
| ppo-accuracy-narrow-w32-s37 | 0.210916 | 0.234254 | 77.511% | 0.045960 |
| ppo-accuracy-narrow-w32-s53 | 0.211989 | 0.236534 | 77.507% | 0.046805 |
| ppo-accuracy-narrow-w32-s71 | 0.212742 | 0.238121 | 77.512% | 0.047499 |
| supervised_continue-labels-narrow-w128-s11 | 0.156250 | 0.014461 | 77.486% | 0.000085 |
| supervised_continue-labels-narrow-w128-s23 | 0.156266 | 0.015005 | 77.485% | 0.000075 |
| supervised_continue-labels-narrow-w128-s37 | 0.156258 | 0.014729 | 77.470% | 0.000076 |
| supervised_continue-labels-narrow-w128-s53 | 0.156184 | 0.011960 | 77.509% | 0.000061 |
| supervised_continue-labels-narrow-w128-s71 | 0.156275 | 0.015294 | 77.472% | 0.000077 |
| reinforce-accuracy-narrow-w128-s11 | 0.213587 | 0.239887 | 77.481% | 0.048121 |
| reinforce-accuracy-narrow-w128-s23 | 0.214355 | 0.241484 | 77.514% | 0.048868 |
| reinforce-accuracy-narrow-w128-s37 | 0.213270 | 0.239225 | 77.442% | 0.047721 |
| reinforce-accuracy-narrow-w128-s53 | 0.214594 | 0.241977 | 77.504% | 0.049001 |
| reinforce-accuracy-narrow-w128-s71 | 0.214963 | 0.242738 | 77.488% | 0.049324 |
| ppo-accuracy-narrow-w128-s11 | 0.215942 | 0.244747 | 77.421% | 0.049852 |
| ppo-accuracy-narrow-w128-s23 | 0.216870 | 0.246636 | 77.442% | 0.050713 |
| ppo-accuracy-narrow-w128-s37 | 0.213950 | 0.240643 | 77.506% | 0.048486 |
| ppo-accuracy-narrow-w128-s53 | 0.215517 | 0.243878 | 77.488% | 0.049750 |
| ppo-accuracy-narrow-w128-s71 | 0.215867 | 0.244593 | 77.517% | 0.050139 |
| reinforce-accuracy-narrow-w32-s11-temperature | 0.158376 | 0.048324 | 77.497% | 0.001874 |
| reinforce-accuracy-narrow-w32-s23-temperature | 0.158283 | 0.047347 | 77.498% | 0.001715 |
| reinforce-accuracy-narrow-w32-s37-temperature | 0.158289 | 0.047411 | 77.503% | 0.001715 |
| reinforce-accuracy-narrow-w32-s53-temperature | 0.157978 | 0.044007 | 77.480% | 0.001632 |
| reinforce-accuracy-narrow-w32-s71-temperature | 0.158040 | 0.044705 | 77.476% | 0.001633 |
| ppo-accuracy-narrow-w32-s11-temperature | 0.158055 | 0.044875 | 77.502% | 0.001707 |
| ppo-accuracy-narrow-w32-s23-temperature | 0.157891 | 0.043007 | 77.500% | 0.001599 |
| ppo-accuracy-narrow-w32-s37-temperature | 0.158012 | 0.044398 | 77.511% | 0.001647 |
| ppo-accuracy-narrow-w32-s53-temperature | 0.157681 | 0.040500 | 77.507% | 0.001554 |
| ppo-accuracy-narrow-w32-s71-temperature | 0.157817 | 0.042146 | 77.512% | 0.001590 |
| reinforce-accuracy-narrow-w128-s11-temperature | 0.158829 | 0.052806 | 77.481% | 0.001947 |
| reinforce-accuracy-narrow-w128-s23-temperature | 0.158689 | 0.051461 | 77.514% | 0.001926 |
| reinforce-accuracy-narrow-w128-s37-temperature | 0.158525 | 0.049840 | 77.442% | 0.001759 |
| reinforce-accuracy-narrow-w128-s53-temperature | 0.159595 | 0.059616 | 77.504% | 0.002245 |
| reinforce-accuracy-narrow-w128-s71-temperature | 0.159016 | 0.054547 | 77.488% | 0.002032 |
| ppo-accuracy-narrow-w128-s11-temperature | 0.159285 | 0.056960 | 77.421% | 0.002060 |
| ppo-accuracy-narrow-w128-s23-temperature | 0.158997 | 0.054368 | 77.442% | 0.001943 |
| ppo-accuracy-narrow-w128-s37-temperature | 0.158184 | 0.046294 | 77.506% | 0.001695 |
| ppo-accuracy-narrow-w128-s53-temperature | 0.159273 | 0.056850 | 77.488% | 0.002151 |
| ppo-accuracy-narrow-w128-s71-temperature | 0.158936 | 0.053809 | 77.517% | 0.002009 |
| oracle_posterior | 0.156041 | 0.000000 | 77.520% | 0.000000 |
| constant_half | 0.250000 | 0.306527 | 49.745% | 0.021153 |

## weaker_sensor

| Recipe | Brier mean [min, max] | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| ppo-accuracy-narrow-w128 | 0.305486 [0.303287, 0.309561] | 0.314026 | 67.797% | 0.079862 |
| ppo-accuracy-narrow-w128-temperature | 0.210933 [0.209665, 0.211792] | 0.063540 | 67.797% | 0.001203 |
| ppo-accuracy-narrow-w32 | 0.301321 [0.299028, 0.302840] | 0.307336 | 67.884% | 0.076693 |
| ppo-accuracy-narrow-w32-temperature | 0.209519 [0.208917, 0.209864] | 0.051457 | 67.884% | 0.000789 |
| ppo-forecast-expanded-w128 | 0.216969 [0.213874, 0.220836] | 0.099648 | 64.568% | 0.002292 |
| ppo-forecast-narrow-w128 | 0.251549 [0.241477, 0.260103] | 0.210878 | 54.838% | 0.007478 |
| ppo-forecast-narrow-w32 | 0.224829 [0.222284, 0.228810] | 0.133642 | 61.012% | 0.003775 |
| reinforce-accuracy-narrow-w128 | 0.302415 [0.301029, 0.303764] | 0.309113 | 67.825% | 0.077401 |
| reinforce-accuracy-narrow-w128-temperature | 0.210420 [0.209424, 0.211869] | 0.059223 | 67.825% | 0.001057 |
| reinforce-accuracy-narrow-w32 | 0.298382 [0.297226, 0.299514] | 0.302521 | 67.751% | 0.073766 |
| reinforce-accuracy-narrow-w32-temperature | 0.210553 [0.209477, 0.211378] | 0.060531 | 67.751% | 0.001066 |
| reinforce-forecast-expanded-w128 | 0.217404 [0.214337, 0.220538] | 0.101954 | 63.523% | 0.002509 |
| reinforce-forecast-narrow-w128 | 0.248650 [0.225686, 0.260023] | 0.201752 | 56.031% | 0.007372 |
| reinforce-forecast-narrow-w32 | 0.228499 [0.223163, 0.232736] | 0.146309 | 60.280% | 0.004725 |
| supervised-labels-expanded-w128 | 0.207212 [0.207067, 0.207376] | 0.018495 | 67.896% | 0.000086 |
| supervised-labels-narrow-w128 | 0.207256 [0.207077, 0.207502] | 0.019488 | 67.935% | 0.000129 |
| supervised-labels-narrow-w32 | 0.207466 [0.207078, 0.207868] | 0.023890 | 67.873% | 0.000148 |
| supervised_continue-labels-narrow-w128 | 0.207425 [0.207060, 0.207719] | 0.023136 | 67.929% | 0.000198 |
| supervised_continue-labels-narrow-w32 | 0.207539 [0.207041, 0.208079] | 0.024864 | 67.863% | 0.000169 |

### Individual runs

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised-labels-narrow-w32-s11 | 0.207078 | 0.014744 | 67.942% | 0.000046 |
| supervised-labels-narrow-w32-s23 | 0.207282 | 0.020525 | 67.899% | 0.000080 |
| supervised-labels-narrow-w32-s37 | 0.207602 | 0.027221 | 67.878% | 0.000244 |
| supervised-labels-narrow-w32-s53 | 0.207868 | 0.031737 | 67.795% | 0.000233 |
| supervised-labels-narrow-w32-s71 | 0.207497 | 0.025222 | 67.853% | 0.000135 |
| reinforce-forecast-narrow-w32-s11 | 0.230922 | 0.155115 | 61.087% | 0.005128 |
| reinforce-forecast-narrow-w32-s23 | 0.232736 | 0.160859 | 61.277% | 0.005143 |
| reinforce-forecast-narrow-w32-s37 | 0.223235 | 0.127961 | 56.143% | 0.005112 |
| reinforce-forecast-narrow-w32-s53 | 0.223163 | 0.127679 | 61.374% | 0.003300 |
| reinforce-forecast-narrow-w32-s71 | 0.232439 | 0.159930 | 61.517% | 0.004943 |
| ppo-forecast-narrow-w32-s11 | 0.227654 | 0.144197 | 61.230% | 0.004757 |
| ppo-forecast-narrow-w32-s23 | 0.228810 | 0.148150 | 61.121% | 0.004402 |
| ppo-forecast-narrow-w32-s37 | 0.223066 | 0.127300 | 60.009% | 0.003912 |
| ppo-forecast-narrow-w32-s53 | 0.222330 | 0.124373 | 61.355% | 0.002905 |
| ppo-forecast-narrow-w32-s71 | 0.222284 | 0.124188 | 61.345% | 0.002900 |
| supervised-labels-narrow-w128-s11 | 0.207130 | 0.016389 | 67.953% | 0.000083 |
| supervised-labels-narrow-w128-s23 | 0.207502 | 0.025320 | 67.876% | 0.000171 |
| supervised-labels-narrow-w128-s37 | 0.207077 | 0.014707 | 67.937% | 0.000048 |
| supervised-labels-narrow-w128-s53 | 0.207375 | 0.022670 | 67.952% | 0.000219 |
| supervised-labels-narrow-w128-s71 | 0.207198 | 0.018353 | 67.955% | 0.000123 |
| reinforce-forecast-narrow-w128-s11 | 0.252147 | 0.212804 | 54.838% | 0.007987 |
| reinforce-forecast-narrow-w128-s23 | 0.260023 | 0.230569 | 54.838% | 0.007983 |
| reinforce-forecast-narrow-w128-s37 | 0.252029 | 0.212528 | 54.838% | 0.008002 |
| reinforce-forecast-narrow-w128-s53 | 0.225686 | 0.137203 | 60.799% | 0.004930 |
| reinforce-forecast-narrow-w128-s71 | 0.253368 | 0.215654 | 54.838% | 0.007956 |
| ppo-forecast-narrow-w128-s11 | 0.241477 | 0.186053 | 54.834% | 0.006466 |
| ppo-forecast-narrow-w128-s23 | 0.260103 | 0.230742 | 54.838% | 0.008009 |
| ppo-forecast-narrow-w128-s37 | 0.252008 | 0.212478 | 54.838% | 0.007985 |
| ppo-forecast-narrow-w128-s53 | 0.254645 | 0.218595 | 54.838% | 0.008300 |
| ppo-forecast-narrow-w128-s71 | 0.249511 | 0.206518 | 54.838% | 0.006630 |
| supervised-labels-expanded-w128-s11 | 0.207067 | 0.014341 | 67.929% | 0.000047 |
| supervised-labels-expanded-w128-s23 | 0.207118 | 0.016040 | 67.915% | 0.000048 |
| supervised-labels-expanded-w128-s37 | 0.207376 | 0.022694 | 67.809% | 0.000124 |
| supervised-labels-expanded-w128-s53 | 0.207274 | 0.020328 | 67.883% | 0.000114 |
| supervised-labels-expanded-w128-s71 | 0.207225 | 0.019074 | 67.945% | 0.000097 |
| reinforce-forecast-expanded-w128-s11 | 0.214887 | 0.089587 | 66.988% | 0.001930 |
| reinforce-forecast-expanded-w128-s23 | 0.220538 | 0.116947 | 60.796% | 0.002929 |
| reinforce-forecast-expanded-w128-s37 | 0.214337 | 0.086463 | 67.156% | 0.001598 |
| reinforce-forecast-expanded-w128-s53 | 0.217663 | 0.103935 | 61.274% | 0.002975 |
| reinforce-forecast-expanded-w128-s71 | 0.219593 | 0.112836 | 61.402% | 0.003111 |
| ppo-forecast-expanded-w128-s11 | 0.215072 | 0.090613 | 65.935% | 0.001851 |
| ppo-forecast-expanded-w128-s23 | 0.219541 | 0.112605 | 61.337% | 0.002600 |
| ppo-forecast-expanded-w128-s37 | 0.215522 | 0.093065 | 66.895% | 0.001949 |
| ppo-forecast-expanded-w128-s53 | 0.213874 | 0.083745 | 67.809% | 0.001586 |
| ppo-forecast-expanded-w128-s71 | 0.220836 | 0.118214 | 60.865% | 0.003472 |
| supervised_continue-labels-narrow-w32-s11 | 0.207041 | 0.013402 | 67.961% | 0.000054 |
| supervised_continue-labels-narrow-w32-s23 | 0.207292 | 0.020771 | 67.922% | 0.000117 |
| supervised_continue-labels-narrow-w32-s37 | 0.207413 | 0.023505 | 67.877% | 0.000155 |
| supervised_continue-labels-narrow-w32-s53 | 0.207868 | 0.031737 | 67.795% | 0.000233 |
| supervised_continue-labels-narrow-w32-s71 | 0.208079 | 0.034904 | 67.760% | 0.000286 |
| reinforce-accuracy-narrow-w32-s11 | 0.298470 | 0.302670 | 67.718% | 0.073745 |
| reinforce-accuracy-narrow-w32-s23 | 0.299514 | 0.304389 | 67.671% | 0.074419 |
| reinforce-accuracy-narrow-w32-s37 | 0.297875 | 0.301685 | 67.755% | 0.073367 |
| reinforce-accuracy-narrow-w32-s53 | 0.297226 | 0.300608 | 67.867% | 0.073160 |
| reinforce-accuracy-narrow-w32-s71 | 0.298823 | 0.303252 | 67.746% | 0.074139 |
| ppo-accuracy-narrow-w32-s11 | 0.301669 | 0.307909 | 67.882% | 0.077023 |
| ppo-accuracy-narrow-w32-s23 | 0.302186 | 0.308747 | 67.903% | 0.077490 |
| ppo-accuracy-narrow-w32-s37 | 0.299028 | 0.303591 | 67.862% | 0.074667 |
| ppo-accuracy-narrow-w32-s53 | 0.300881 | 0.306626 | 67.921% | 0.076469 |
| ppo-accuracy-narrow-w32-s71 | 0.302840 | 0.309805 | 67.852% | 0.077818 |
| supervised_continue-labels-narrow-w128-s11 | 0.207719 | 0.029299 | 67.911% | 0.000323 |
| supervised_continue-labels-narrow-w128-s23 | 0.207502 | 0.025320 | 67.876% | 0.000171 |
| supervised_continue-labels-narrow-w128-s37 | 0.207273 | 0.020294 | 67.958% | 0.000178 |
| supervised_continue-labels-narrow-w128-s53 | 0.207572 | 0.026669 | 67.943% | 0.000258 |
| supervised_continue-labels-narrow-w128-s71 | 0.207060 | 0.014095 | 67.958% | 0.000059 |
| reinforce-accuracy-narrow-w128-s11 | 0.301029 | 0.306868 | 67.852% | 0.076329 |
| reinforce-accuracy-narrow-w128-s23 | 0.303764 | 0.311293 | 67.771% | 0.078415 |
| reinforce-accuracy-narrow-w128-s37 | 0.301535 | 0.307692 | 67.894% | 0.076934 |
| reinforce-accuracy-narrow-w128-s53 | 0.303177 | 0.310348 | 67.845% | 0.078016 |
| reinforce-accuracy-narrow-w128-s71 | 0.302568 | 0.309366 | 67.761% | 0.077310 |
| ppo-accuracy-narrow-w128-s11 | 0.305683 | 0.314360 | 67.775% | 0.080035 |
| ppo-accuracy-narrow-w128-s23 | 0.309561 | 0.320469 | 67.678% | 0.082849 |
| ppo-accuracy-narrow-w128-s37 | 0.303765 | 0.311294 | 67.796% | 0.078395 |
| ppo-accuracy-narrow-w128-s53 | 0.303287 | 0.310525 | 67.902% | 0.078417 |
| ppo-accuracy-narrow-w128-s71 | 0.305132 | 0.313482 | 67.832% | 0.079612 |
| reinforce-accuracy-narrow-w32-s11-temperature | 0.210835 | 0.063036 | 67.718% | 0.001122 |
| reinforce-accuracy-narrow-w32-s23-temperature | 0.211378 | 0.067206 | 67.671% | 0.001264 |
| reinforce-accuracy-narrow-w32-s37-temperature | 0.210547 | 0.060715 | 67.755% | 0.001134 |
| reinforce-accuracy-narrow-w32-s53-temperature | 0.209477 | 0.051147 | 67.867% | 0.000771 |
| reinforce-accuracy-narrow-w32-s71-temperature | 0.210528 | 0.060553 | 67.746% | 0.001039 |
| ppo-accuracy-narrow-w32-s11-temperature | 0.209552 | 0.051871 | 67.882% | 0.000779 |
| ppo-accuracy-narrow-w32-s23-temperature | 0.209864 | 0.054801 | 67.903% | 0.000892 |
| ppo-accuracy-narrow-w32-s37-temperature | 0.209605 | 0.052383 | 67.862% | 0.000845 |
| ppo-accuracy-narrow-w32-s53-temperature | 0.208917 | 0.045345 | 67.921% | 0.000606 |
| ppo-accuracy-narrow-w32-s71-temperature | 0.209658 | 0.052885 | 67.852% | 0.000824 |
| reinforce-accuracy-narrow-w128-s11-temperature | 0.209692 | 0.053205 | 67.852% | 0.000800 |
| reinforce-accuracy-narrow-w128-s23-temperature | 0.210280 | 0.058473 | 67.771% | 0.000998 |
| reinforce-accuracy-narrow-w128-s37-temperature | 0.209424 | 0.050627 | 67.894% | 0.000733 |
| reinforce-accuracy-narrow-w128-s53-temperature | 0.211869 | 0.070768 | 67.845% | 0.001524 |
| reinforce-accuracy-narrow-w128-s71-temperature | 0.210835 | 0.063041 | 67.761% | 0.001231 |
| ppo-accuracy-narrow-w128-s11-temperature | 0.211192 | 0.065811 | 67.775% | 0.001241 |
| ppo-accuracy-narrow-w128-s23-temperature | 0.211792 | 0.070224 | 67.678% | 0.001418 |
| ppo-accuracy-narrow-w128-s37-temperature | 0.209665 | 0.052952 | 67.796% | 0.000788 |
| ppo-accuracy-narrow-w128-s53-temperature | 0.210715 | 0.062083 | 67.902% | 0.001177 |
| ppo-accuracy-narrow-w128-s71-temperature | 0.211300 | 0.066628 | 67.832% | 0.001390 |
| oracle_posterior | 0.206861 | 0.000000 | 67.972% | 0.000000 |
| constant_half | 0.250000 | 0.207699 | 50.079% | 0.008661 |

## stronger_sensor

| Recipe | Brier mean [min, max] | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| ppo-accuracy-narrow-w128 | 0.049789 [0.049581, 0.049996] | 0.070989 | 94.993% | 0.004779 |
| ppo-accuracy-narrow-w128-temperature | 0.064710 [0.061562, 0.066138] | 0.141163 | 94.993% | 0.015423 |
| ppo-accuracy-narrow-w32 | 0.049915 [0.049771, 0.049988] | 0.071880 | 94.993% | 0.004917 |
| ppo-accuracy-narrow-w32-temperature | 0.058107 [0.057328, 0.059380] | 0.115543 | 94.993% | 0.013155 |
| ppo-forecast-expanded-w128 | 0.061806 [0.051877, 0.070015] | 0.128223 | 94.984% | 0.014634 |
| ppo-forecast-narrow-w128 | 0.065942 [0.063944, 0.070397] | 0.145387 | 94.977% | 0.015677 |
| ppo-forecast-narrow-w32 | 0.078408 [0.060046, 0.107351] | 0.177081 | 94.563% | 0.022461 |
| reinforce-accuracy-narrow-w128 | 0.049754 [0.049468, 0.049996] | 0.070742 | 94.993% | 0.004771 |
| reinforce-accuracy-narrow-w128-temperature | 0.064718 [0.062926, 0.066399] | 0.141262 | 94.993% | 0.015421 |
| reinforce-accuracy-narrow-w32 | 0.049538 [0.049498, 0.049594] | 0.069208 | 94.993% | 0.004569 |
| reinforce-accuracy-narrow-w32-temperature | 0.059427 [0.058539, 0.060223] | 0.121129 | 94.993% | 0.013545 |
| reinforce-forecast-expanded-w128 | 0.060040 [0.053322, 0.069985] | 0.121001 | 94.993% | 0.014791 |
| reinforce-forecast-narrow-w128 | 0.071782 [0.065082, 0.088756] | 0.162495 | 94.607% | 0.019605 |
| reinforce-forecast-narrow-w32 | 0.086653 [0.075340, 0.094456] | 0.203822 | 94.052% | 0.025502 |
| supervised-labels-expanded-w128 | 0.045120 [0.044943, 0.045379] | 0.018907 | 94.993% | 0.000313 |
| supervised-labels-narrow-w128 | 0.048130 [0.047502, 0.049388] | 0.057910 | 94.993% | 0.002703 |
| supervised-labels-narrow-w32 | 0.048525 [0.047444, 0.048870] | 0.061281 | 94.993% | 0.002964 |
| supervised_continue-labels-narrow-w128 | 0.047871 [0.047119, 0.049253] | 0.055507 | 94.993% | 0.002526 |
| supervised_continue-labels-narrow-w32 | 0.047997 [0.047361, 0.048870] | 0.056814 | 94.993% | 0.002605 |

### Individual runs

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised-labels-narrow-w32-s11 | 0.047444 | 0.051926 | 94.993% | 0.002359 |
| supervised-labels-narrow-w32-s23 | 0.048785 | 0.063535 | 94.993% | 0.003129 |
| supervised-labels-narrow-w32-s37 | 0.048720 | 0.063022 | 94.993% | 0.003074 |
| supervised-labels-narrow-w32-s53 | 0.048870 | 0.064202 | 94.993% | 0.003093 |
| supervised-labels-narrow-w32-s71 | 0.048808 | 0.063720 | 94.993% | 0.003164 |
| reinforce-forecast-narrow-w32-s11 | 0.092723 | 0.219031 | 93.697% | 0.029331 |
| reinforce-forecast-narrow-w32-s23 | 0.094456 | 0.222953 | 94.579% | 0.030123 |
| reinforce-forecast-narrow-w32-s37 | 0.079973 | 0.187685 | 94.865% | 0.021472 |
| reinforce-forecast-narrow-w32-s53 | 0.075340 | 0.174906 | 94.722% | 0.017383 |
| reinforce-forecast-narrow-w32-s71 | 0.090774 | 0.214537 | 92.399% | 0.029201 |
| ppo-forecast-narrow-w32-s11 | 0.091560 | 0.216361 | 94.346% | 0.029335 |
| ppo-forecast-narrow-w32-s23 | 0.107351 | 0.250207 | 94.339% | 0.035568 |
| ppo-forecast-narrow-w32-s37 | 0.067737 | 0.151623 | 94.717% | 0.017035 |
| ppo-forecast-narrow-w32-s53 | 0.060046 | 0.123686 | 94.816% | 0.014143 |
| ppo-forecast-narrow-w32-s71 | 0.065348 | 0.143526 | 94.598% | 0.016225 |
| supervised-labels-narrow-w128-s11 | 0.047758 | 0.054865 | 94.993% | 0.002440 |
| supervised-labels-narrow-w128-s23 | 0.047991 | 0.056952 | 94.993% | 0.002543 |
| supervised-labels-narrow-w128-s37 | 0.048012 | 0.057136 | 94.993% | 0.002575 |
| supervised-labels-narrow-w128-s53 | 0.047502 | 0.052481 | 94.993% | 0.002408 |
| supervised-labels-narrow-w128-s71 | 0.049388 | 0.068118 | 94.993% | 0.003550 |
| reinforce-forecast-narrow-w128-s11 | 0.072529 | 0.166678 | 94.993% | 0.019289 |
| reinforce-forecast-narrow-w128-s23 | 0.065082 | 0.142598 | 94.993% | 0.016612 |
| reinforce-forecast-narrow-w128-s37 | 0.066256 | 0.146655 | 94.917% | 0.016749 |
| reinforce-forecast-narrow-w128-s53 | 0.088756 | 0.209782 | 93.142% | 0.028348 |
| reinforce-forecast-narrow-w128-s71 | 0.066286 | 0.146760 | 94.993% | 0.017027 |
| ppo-forecast-narrow-w128-s11 | 0.070397 | 0.160153 | 94.993% | 0.017500 |
| ppo-forecast-narrow-w128-s23 | 0.063944 | 0.138551 | 94.993% | 0.013918 |
| ppo-forecast-narrow-w128-s37 | 0.064972 | 0.142212 | 94.914% | 0.014144 |
| ppo-forecast-narrow-w128-s53 | 0.065125 | 0.142748 | 94.993% | 0.016134 |
| ppo-forecast-narrow-w128-s71 | 0.065274 | 0.143269 | 94.993% | 0.016692 |
| supervised-labels-expanded-w128-s11 | 0.044943 | 0.013965 | 94.993% | 0.000205 |
| supervised-labels-expanded-w128-s23 | 0.044994 | 0.015689 | 94.993% | 0.000256 |
| supervised-labels-expanded-w128-s37 | 0.045379 | 0.025116 | 94.993% | 0.000447 |
| supervised-labels-expanded-w128-s53 | 0.045157 | 0.020229 | 94.993% | 0.000324 |
| supervised-labels-expanded-w128-s71 | 0.045129 | 0.019534 | 94.993% | 0.000336 |
| reinforce-forecast-expanded-w128-s11 | 0.069985 | 0.158861 | 94.993% | 0.018636 |
| reinforce-forecast-expanded-w128-s23 | 0.063633 | 0.137423 | 94.993% | 0.016074 |
| reinforce-forecast-expanded-w128-s37 | 0.059730 | 0.122401 | 94.993% | 0.014796 |
| reinforce-forecast-expanded-w128-s53 | 0.053322 | 0.092596 | 94.993% | 0.012196 |
| reinforce-forecast-expanded-w128-s71 | 0.053532 | 0.093721 | 94.993% | 0.012254 |
| ppo-forecast-expanded-w128-s11 | 0.070015 | 0.158958 | 94.993% | 0.018663 |
| ppo-forecast-expanded-w128-s23 | 0.063674 | 0.137574 | 94.993% | 0.016094 |
| ppo-forecast-expanded-w128-s37 | 0.063837 | 0.138163 | 94.957% | 0.016103 |
| ppo-forecast-expanded-w128-s53 | 0.051877 | 0.084434 | 94.983% | 0.007541 |
| ppo-forecast-expanded-w128-s71 | 0.059629 | 0.121988 | 94.993% | 0.014767 |
| supervised_continue-labels-narrow-w32-s11 | 0.047537 | 0.052816 | 94.993% | 0.002332 |
| supervised_continue-labels-narrow-w32-s23 | 0.047361 | 0.051119 | 94.993% | 0.002293 |
| supervised_continue-labels-narrow-w32-s37 | 0.048081 | 0.057737 | 94.993% | 0.002616 |
| supervised_continue-labels-narrow-w32-s53 | 0.048870 | 0.064202 | 94.993% | 0.003093 |
| supervised_continue-labels-narrow-w32-s71 | 0.048135 | 0.058198 | 94.993% | 0.002689 |
| reinforce-accuracy-narrow-w32-s11 | 0.049498 | 0.068920 | 94.993% | 0.004530 |
| reinforce-accuracy-narrow-w32-s23 | 0.049594 | 0.069614 | 94.993% | 0.004623 |
| reinforce-accuracy-narrow-w32-s37 | 0.049538 | 0.069212 | 94.993% | 0.004573 |
| reinforce-accuracy-narrow-w32-s53 | 0.049517 | 0.069057 | 94.993% | 0.004549 |
| reinforce-accuracy-narrow-w32-s71 | 0.049541 | 0.069236 | 94.993% | 0.004570 |
| ppo-accuracy-narrow-w32-s11 | 0.049975 | 0.072300 | 94.993% | 0.004964 |
| ppo-accuracy-narrow-w32-s23 | 0.049913 | 0.071869 | 94.993% | 0.004920 |
| ppo-accuracy-narrow-w32-s37 | 0.049771 | 0.070872 | 94.993% | 0.004798 |
| ppo-accuracy-narrow-w32-s53 | 0.049928 | 0.071970 | 94.993% | 0.004930 |
| ppo-accuracy-narrow-w32-s71 | 0.049988 | 0.072387 | 94.993% | 0.004974 |
| supervised_continue-labels-narrow-w128-s11 | 0.047318 | 0.050697 | 94.993% | 0.002213 |
| supervised_continue-labels-narrow-w128-s23 | 0.047991 | 0.056952 | 94.993% | 0.002543 |
| supervised_continue-labels-narrow-w128-s37 | 0.047119 | 0.048697 | 94.993% | 0.002052 |
| supervised_continue-labels-narrow-w128-s53 | 0.047671 | 0.054066 | 94.993% | 0.002538 |
| supervised_continue-labels-narrow-w128-s71 | 0.049253 | 0.067121 | 94.993% | 0.003284 |
| reinforce-accuracy-narrow-w128-s11 | 0.049732 | 0.070600 | 94.993% | 0.004771 |
| reinforce-accuracy-narrow-w128-s23 | 0.049897 | 0.071757 | 94.993% | 0.004910 |
| reinforce-accuracy-narrow-w128-s37 | 0.049468 | 0.068706 | 94.993% | 0.004481 |
| reinforce-accuracy-narrow-w128-s53 | 0.049676 | 0.070203 | 94.993% | 0.004694 |
| reinforce-accuracy-narrow-w128-s71 | 0.049996 | 0.072446 | 94.993% | 0.004997 |
| ppo-accuracy-narrow-w128-s11 | 0.049668 | 0.070145 | 94.993% | 0.004672 |
| ppo-accuracy-narrow-w128-s23 | 0.049581 | 0.069523 | 94.993% | 0.004534 |
| ppo-accuracy-narrow-w128-s37 | 0.049776 | 0.070909 | 94.993% | 0.004761 |
| ppo-accuracy-narrow-w128-s53 | 0.049921 | 0.071924 | 94.993% | 0.004939 |
| ppo-accuracy-narrow-w128-s71 | 0.049996 | 0.072445 | 94.993% | 0.004990 |
| reinforce-accuracy-narrow-w32-s11-temperature | 0.059966 | 0.123363 | 94.993% | 0.013914 |
| reinforce-accuracy-narrow-w32-s23-temperature | 0.059440 | 0.121210 | 94.993% | 0.013501 |
| reinforce-accuracy-narrow-w32-s37-temperature | 0.060223 | 0.124399 | 94.993% | 0.013699 |
| reinforce-accuracy-narrow-w32-s53-temperature | 0.058539 | 0.117436 | 94.993% | 0.013239 |
| reinforce-accuracy-narrow-w32-s71-temperature | 0.058966 | 0.119238 | 94.993% | 0.013374 |
| ppo-accuracy-narrow-w32-s11-temperature | 0.058137 | 0.115711 | 94.993% | 0.013228 |
| ppo-accuracy-narrow-w32-s23-temperature | 0.058076 | 0.115450 | 94.993% | 0.013131 |
| ppo-accuracy-narrow-w32-s37-temperature | 0.059380 | 0.120965 | 94.993% | 0.013470 |
| ppo-accuracy-narrow-w32-s53-temperature | 0.057328 | 0.112162 | 94.993% | 0.012937 |
| ppo-accuracy-narrow-w32-s71-temperature | 0.057614 | 0.113429 | 94.993% | 0.013011 |
| reinforce-accuracy-narrow-w128-s11-temperature | 0.064752 | 0.141436 | 94.993% | 0.015394 |
| reinforce-accuracy-narrow-w128-s23-temperature | 0.064679 | 0.141178 | 94.993% | 0.015577 |
| reinforce-accuracy-narrow-w128-s37-temperature | 0.062926 | 0.134825 | 94.993% | 0.014555 |
| reinforce-accuracy-narrow-w128-s53-temperature | 0.066399 | 0.147144 | 94.993% | 0.015833 |
| reinforce-accuracy-narrow-w128-s71-temperature | 0.064834 | 0.141727 | 94.993% | 0.015746 |
| ppo-accuracy-narrow-w128-s11-temperature | 0.065596 | 0.144388 | 94.993% | 0.015691 |
| ppo-accuracy-narrow-w128-s23-temperature | 0.066138 | 0.146254 | 94.993% | 0.015693 |
| ppo-accuracy-narrow-w128-s37-temperature | 0.061562 | 0.129671 | 94.993% | 0.014132 |
| ppo-accuracy-narrow-w128-s53-temperature | 0.064932 | 0.142070 | 94.993% | 0.015679 |
| ppo-accuracy-narrow-w128-s71-temperature | 0.065321 | 0.143433 | 94.993% | 0.015921 |
| oracle_posterior | 0.044748 | 0.000000 | 94.993% | 0.000000 |
| constant_half | 0.250000 | 0.453048 | 49.987% | 0.055958 |

## extreme_prior

| Recipe | Brier mean [min, max] | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| ppo-accuracy-narrow-w128 | 0.078612 [0.077617, 0.079547] | 0.122705 | 91.961% | 0.013260 |
| ppo-accuracy-narrow-w128-temperature | 0.071044 [0.070560, 0.071543] | 0.086557 | 91.961% | 0.009926 |
| ppo-accuracy-narrow-w32 | 0.077977 [0.077524, 0.078491] | 0.120119 | 91.960% | 0.012740 |
| ppo-accuracy-narrow-w32-temperature | 0.070675 [0.069820, 0.071591] | 0.084355 | 91.960% | 0.010346 |
| ppo-forecast-expanded-w128 | 0.093685 [0.081119, 0.100829] | 0.172177 | 88.540% | 0.016274 |
| ppo-forecast-narrow-w128 | 0.139702 [0.134118, 0.147202] | 0.275840 | 74.845% | 0.024447 |
| ppo-forecast-narrow-w32 | 0.111198 [0.102583, 0.126720] | 0.217388 | 86.452% | 0.023017 |
| reinforce-accuracy-narrow-w128 | 0.077841 [0.075699, 0.079355] | 0.119441 | 91.953% | 0.012587 |
| reinforce-accuracy-narrow-w128-temperature | 0.071667 [0.070300, 0.072778] | 0.089998 | 91.953% | 0.010387 |
| reinforce-accuracy-narrow-w32 | 0.077955 [0.077678, 0.078163] | 0.120033 | 91.955% | 0.012685 |
| reinforce-accuracy-narrow-w32-temperature | 0.071196 [0.070530, 0.072163] | 0.087407 | 91.955% | 0.010393 |
| reinforce-forecast-expanded-w128 | 0.094143 [0.087310, 0.099599] | 0.174509 | 87.249% | 0.016949 |
| reinforce-forecast-narrow-w128 | 0.138262 [0.117176, 0.147160] | 0.272535 | 76.580% | 0.026385 |
| reinforce-forecast-narrow-w32 | 0.114391 [0.104372, 0.123320] | 0.224995 | 88.630% | 0.023665 |
| supervised-labels-expanded-w128 | 0.064298 [0.064058, 0.064642] | 0.027167 | 91.951% | 0.000463 |
| supervised-labels-narrow-w128 | 0.067751 [0.066974, 0.068741] | 0.064591 | 91.911% | 0.002101 |
| supervised-labels-narrow-w32 | 0.067935 [0.066871, 0.069002] | 0.066018 | 91.905% | 0.002193 |
| supervised_continue-labels-narrow-w128 | 0.067834 [0.067150, 0.068345] | 0.065384 | 91.927% | 0.002199 |
| supervised_continue-labels-narrow-w32 | 0.067888 [0.067522, 0.068613] | 0.065817 | 91.915% | 0.002186 |

### Individual runs

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised-labels-narrow-w32-s11 | 0.068396 | 0.069637 | 91.791% | 0.002198 |
| supervised-labels-narrow-w32-s23 | 0.066871 | 0.057663 | 91.962% | 0.001770 |
| supervised-labels-narrow-w32-s37 | 0.069002 | 0.073863 | 91.901% | 0.002663 |
| supervised-labels-narrow-w32-s53 | 0.067522 | 0.063050 | 91.952% | 0.002166 |
| supervised-labels-narrow-w32-s71 | 0.067886 | 0.065878 | 91.919% | 0.002170 |
| reinforce-forecast-narrow-w32-s11 | 0.110220 | 0.216042 | 91.948% | 0.025682 |
| reinforce-forecast-narrow-w32-s23 | 0.123320 | 0.244486 | 84.884% | 0.027051 |
| reinforce-forecast-narrow-w32-s37 | 0.119233 | 0.235979 | 90.386% | 0.020349 |
| reinforce-forecast-narrow-w32-s53 | 0.114809 | 0.226412 | 83.983% | 0.019827 |
| reinforce-forecast-narrow-w32-s71 | 0.104372 | 0.202055 | 91.948% | 0.025415 |
| ppo-forecast-narrow-w32-s11 | 0.115431 | 0.227782 | 91.862% | 0.026331 |
| ppo-forecast-narrow-w32-s23 | 0.126720 | 0.251343 | 84.917% | 0.033407 |
| ppo-forecast-narrow-w32-s37 | 0.106873 | 0.208150 | 87.715% | 0.018889 |
| ppo-forecast-narrow-w32-s53 | 0.102583 | 0.197576 | 83.787% | 0.017761 |
| ppo-forecast-narrow-w32-s71 | 0.104386 | 0.202088 | 83.978% | 0.018698 |
| supervised-labels-narrow-w128-s11 | 0.067043 | 0.059137 | 91.932% | 0.001852 |
| supervised-labels-narrow-w128-s23 | 0.067440 | 0.062404 | 91.951% | 0.001882 |
| supervised-labels-narrow-w128-s37 | 0.066974 | 0.058546 | 91.956% | 0.001744 |
| supervised-labels-narrow-w128-s53 | 0.068558 | 0.070797 | 91.839% | 0.002562 |
| supervised-labels-narrow-w128-s71 | 0.068741 | 0.072071 | 91.875% | 0.002465 |
| reinforce-forecast-narrow-w128-s11 | 0.145559 | 0.286378 | 74.845% | 0.027566 |
| reinforce-forecast-narrow-w128-s23 | 0.147160 | 0.289160 | 74.845% | 0.025503 |
| reinforce-forecast-narrow-w128-s37 | 0.140294 | 0.277034 | 74.845% | 0.025503 |
| reinforce-forecast-narrow-w128-s53 | 0.117176 | 0.231581 | 83.523% | 0.028197 |
| reinforce-forecast-narrow-w128-s71 | 0.141120 | 0.278520 | 74.845% | 0.025155 |
| ppo-forecast-narrow-w128-s11 | 0.134118 | 0.265654 | 74.845% | 0.024731 |
| ppo-forecast-narrow-w128-s23 | 0.147202 | 0.289234 | 74.845% | 0.023432 |
| ppo-forecast-narrow-w128-s37 | 0.140478 | 0.277365 | 74.845% | 0.025503 |
| ppo-forecast-narrow-w128-s53 | 0.140982 | 0.278273 | 74.845% | 0.024788 |
| ppo-forecast-narrow-w128-s71 | 0.135731 | 0.268672 | 74.845% | 0.023781 |
| supervised-labels-expanded-w128-s11 | 0.064228 | 0.026103 | 91.962% | 0.000448 |
| supervised-labels-expanded-w128-s23 | 0.064642 | 0.033110 | 91.910% | 0.000599 |
| supervised-labels-expanded-w128-s37 | 0.064140 | 0.024361 | 91.959% | 0.000356 |
| supervised-labels-expanded-w128-s53 | 0.064058 | 0.022627 | 91.958% | 0.000282 |
| supervised-labels-expanded-w128-s71 | 0.064424 | 0.029633 | 91.966% | 0.000630 |
| reinforce-forecast-expanded-w128-s11 | 0.095689 | 0.179285 | 91.950% | 0.018067 |
| reinforce-forecast-expanded-w128-s23 | 0.099599 | 0.189876 | 83.168% | 0.018425 |
| reinforce-forecast-expanded-w128-s37 | 0.087310 | 0.154156 | 91.952% | 0.015207 |
| reinforce-forecast-expanded-w128-s53 | 0.092442 | 0.169987 | 85.753% | 0.016230 |
| reinforce-forecast-expanded-w128-s71 | 0.095674 | 0.179242 | 83.421% | 0.016819 |
| ppo-forecast-expanded-w128-s11 | 0.095833 | 0.179686 | 91.953% | 0.018341 |
| ppo-forecast-expanded-w128-s23 | 0.099972 | 0.190856 | 83.392% | 0.018472 |
| ppo-forecast-expanded-w128-s37 | 0.090670 | 0.164694 | 91.954% | 0.016292 |
| ppo-forecast-expanded-w128-s53 | 0.081119 | 0.132564 | 91.959% | 0.009381 |
| ppo-forecast-expanded-w128-s71 | 0.100829 | 0.193087 | 83.444% | 0.018884 |
| supervised_continue-labels-narrow-w32-s11 | 0.067534 | 0.063153 | 91.922% | 0.001997 |
| supervised_continue-labels-narrow-w32-s23 | 0.068613 | 0.071182 | 91.828% | 0.002425 |
| supervised_continue-labels-narrow-w32-s37 | 0.068073 | 0.067282 | 91.937% | 0.002238 |
| supervised_continue-labels-narrow-w32-s53 | 0.067522 | 0.063050 | 91.952% | 0.002166 |
| supervised_continue-labels-narrow-w32-s71 | 0.067696 | 0.064419 | 91.934% | 0.002103 |
| reinforce-accuracy-narrow-w32-s11 | 0.078163 | 0.120902 | 91.951% | 0.012825 |
| reinforce-accuracy-narrow-w32-s23 | 0.077941 | 0.119979 | 91.954% | 0.012670 |
| reinforce-accuracy-narrow-w32-s37 | 0.077982 | 0.120148 | 91.953% | 0.012698 |
| reinforce-accuracy-narrow-w32-s53 | 0.078008 | 0.120256 | 91.956% | 0.012751 |
| reinforce-accuracy-narrow-w32-s71 | 0.077678 | 0.118879 | 91.960% | 0.012481 |
| ppo-accuracy-narrow-w32-s11 | 0.077524 | 0.118229 | 91.964% | 0.012379 |
| ppo-accuracy-narrow-w32-s23 | 0.078491 | 0.122249 | 91.957% | 0.013149 |
| ppo-accuracy-narrow-w32-s37 | 0.078012 | 0.120272 | 91.956% | 0.012759 |
| ppo-accuracy-narrow-w32-s53 | 0.078161 | 0.120890 | 91.958% | 0.012884 |
| ppo-accuracy-narrow-w32-s71 | 0.077697 | 0.118956 | 91.965% | 0.012530 |
| supervised_continue-labels-narrow-w128-s11 | 0.068195 | 0.068185 | 91.924% | 0.002426 |
| supervised_continue-labels-narrow-w128-s23 | 0.067440 | 0.062404 | 91.951% | 0.001882 |
| supervised_continue-labels-narrow-w128-s37 | 0.068039 | 0.067027 | 91.913% | 0.002390 |
| supervised_continue-labels-narrow-w128-s53 | 0.068345 | 0.069275 | 91.905% | 0.002391 |
| supervised_continue-labels-narrow-w128-s71 | 0.067150 | 0.060031 | 91.942% | 0.001908 |
| reinforce-accuracy-narrow-w128-s11 | 0.078020 | 0.120307 | 91.961% | 0.012766 |
| reinforce-accuracy-narrow-w128-s23 | 0.077400 | 0.117700 | 91.969% | 0.012287 |
| reinforce-accuracy-narrow-w128-s37 | 0.079355 | 0.125733 | 91.948% | 0.013811 |
| reinforce-accuracy-narrow-w128-s53 | 0.078731 | 0.123225 | 91.955% | 0.013329 |
| reinforce-accuracy-narrow-w128-s71 | 0.075699 | 0.110240 | 91.931% | 0.010740 |
| ppo-accuracy-narrow-w128-s11 | 0.079250 | 0.125316 | 91.952% | 0.013740 |
| ppo-accuracy-narrow-w128-s23 | 0.079547 | 0.126494 | 91.955% | 0.014024 |
| ppo-accuracy-narrow-w128-s37 | 0.078621 | 0.122779 | 91.960% | 0.013250 |
| ppo-accuracy-narrow-w128-s53 | 0.078023 | 0.120320 | 91.967% | 0.012802 |
| ppo-accuracy-narrow-w128-s71 | 0.077617 | 0.118618 | 91.972% | 0.012481 |
| reinforce-accuracy-narrow-w32-s11-temperature | 0.071444 | 0.088871 | 91.951% | 0.010311 |
| reinforce-accuracy-narrow-w32-s23-temperature | 0.070877 | 0.085620 | 91.954% | 0.010195 |
| reinforce-accuracy-narrow-w32-s37-temperature | 0.072163 | 0.092827 | 91.953% | 0.010695 |
| reinforce-accuracy-narrow-w32-s53-temperature | 0.070530 | 0.083569 | 91.956% | 0.010330 |
| reinforce-accuracy-narrow-w32-s71-temperature | 0.070967 | 0.086146 | 91.960% | 0.010435 |
| ppo-accuracy-narrow-w32-s11-temperature | 0.070949 | 0.086039 | 91.964% | 0.010278 |
| ppo-accuracy-narrow-w32-s23-temperature | 0.069820 | 0.079207 | 91.957% | 0.010107 |
| ppo-accuracy-narrow-w32-s37-temperature | 0.071591 | 0.089694 | 91.956% | 0.010641 |
| ppo-accuracy-narrow-w32-s53-temperature | 0.070228 | 0.081741 | 91.958% | 0.010302 |
| ppo-accuracy-narrow-w32-s71-temperature | 0.070787 | 0.085095 | 91.965% | 0.010400 |
| reinforce-accuracy-narrow-w128-s11-temperature | 0.071453 | 0.088917 | 91.961% | 0.010305 |
| reinforce-accuracy-narrow-w128-s23-temperature | 0.071672 | 0.090143 | 91.969% | 0.010473 |
| reinforce-accuracy-narrow-w128-s37-temperature | 0.070300 | 0.082181 | 91.948% | 0.010090 |
| reinforce-accuracy-narrow-w128-s53-temperature | 0.072133 | 0.092667 | 91.955% | 0.010689 |
| reinforce-accuracy-narrow-w128-s71-temperature | 0.072778 | 0.096083 | 91.931% | 0.010378 |
| ppo-accuracy-narrow-w128-s11-temperature | 0.071002 | 0.086346 | 91.952% | 0.009833 |
| ppo-accuracy-narrow-w128-s23-temperature | 0.070560 | 0.083747 | 91.955% | 0.009932 |
| ppo-accuracy-narrow-w128-s37-temperature | 0.070645 | 0.084257 | 91.960% | 0.010276 |
| ppo-accuracy-narrow-w128-s53-temperature | 0.071543 | 0.089426 | 91.967% | 0.009895 |
| ppo-accuracy-narrow-w128-s71-temperature | 0.071469 | 0.089011 | 91.972% | 0.009695 |
| oracle_posterior | 0.063546 | 0.000000 | 91.973% | 0.000000 |
| constant_half | 0.250000 | 0.431803 | 49.951% | 0.050531 |

## reversed_sensor

| Recipe | Brier mean [min, max] | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| ppo-accuracy-narrow-w128 | 0.345913 [0.327246, 0.354386] | 0.435698 | 63.056% | 0.110380 |
| ppo-accuracy-narrow-w128-temperature | 0.229278 [0.220588, 0.233990] | 0.270628 | 63.056% | 0.016771 |
| ppo-accuracy-narrow-w32 | 0.298222 [0.274210, 0.320878] | 0.376371 | 67.759% | 0.086317 |
| ppo-accuracy-narrow-w32-temperature | 0.209643 [0.198033, 0.221283] | 0.230868 | 67.759% | 0.013420 |
| ppo-forecast-expanded-w128 | 0.208546 [0.194089, 0.236818] | 0.226517 | 66.590% | 0.013407 |
| ppo-forecast-narrow-w128 | 0.338543 [0.304307, 0.368259] | 0.426615 | 25.422% | 0.035245 |
| ppo-forecast-narrow-w32 | 0.259776 [0.249684, 0.275402] | 0.321739 | 49.884% | 0.022337 |
| reinforce-accuracy-narrow-w128 | 0.348592 [0.321858, 0.364136] | 0.438515 | 62.382% | 0.110251 |
| reinforce-accuracy-narrow-w128-temperature | 0.232186 [0.218800, 0.239206] | 0.275714 | 62.382% | 0.017195 |
| reinforce-accuracy-narrow-w32 | 0.307883 [0.283129, 0.334998] | 0.388698 | 66.237% | 0.089275 |
| reinforce-accuracy-narrow-w32-temperature | 0.217858 [0.203448, 0.232443] | 0.247624 | 66.237% | 0.015344 |
| reinforce-forecast-expanded-w128 | 0.214456 [0.193635, 0.236850] | 0.239041 | 61.512% | 0.015195 |
| reinforce-forecast-narrow-w128 | 0.333108 [0.252288, 0.368772] | 0.417414 | 30.007% | 0.034980 |
| reinforce-forecast-narrow-w32 | 0.273234 [0.249761, 0.299098] | 0.341216 | 44.193% | 0.025606 |
| supervised-labels-expanded-w128 | 0.156688 [0.156537, 0.156808] | 0.027445 | 77.480% | 0.000322 |
| supervised-labels-narrow-w128 | 0.204261 [0.180248, 0.232713] | 0.213662 | 68.733% | 0.012095 |
| supervised-labels-narrow-w32 | 0.179911 [0.170794, 0.193980] | 0.152328 | 73.626% | 0.006843 |
| supervised_continue-labels-narrow-w128 | 0.215235 [0.180944, 0.251064] | 0.237785 | 66.593% | 0.014412 |
| supervised_continue-labels-narrow-w32 | 0.187215 [0.170794, 0.208878] | 0.171900 | 72.229% | 0.008496 |

### Individual runs

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised-labels-narrow-w32-s11 | 0.193980 | 0.195062 | 70.638% | 0.009797 |
| supervised-labels-narrow-w32-s23 | 0.185424 | 0.171735 | 72.128% | 0.007804 |
| supervised-labels-narrow-w32-s37 | 0.171631 | 0.125300 | 75.787% | 0.005183 |
| supervised-labels-narrow-w32-s53 | 0.170794 | 0.121913 | 75.560% | 0.004917 |
| supervised-labels-narrow-w32-s71 | 0.177725 | 0.147629 | 74.017% | 0.006512 |
| reinforce-forecast-narrow-w32-s11 | 0.289878 | 0.365988 | 47.014% | 0.027116 |
| reinforce-forecast-narrow-w32-s23 | 0.276783 | 0.347637 | 50.451% | 0.025340 |
| reinforce-forecast-narrow-w32-s37 | 0.249761 | 0.306316 | 30.833% | 0.025806 |
| reinforce-forecast-narrow-w32-s53 | 0.250652 | 0.307767 | 51.249% | 0.020839 |
| reinforce-forecast-narrow-w32-s71 | 0.299098 | 0.378374 | 41.418% | 0.028929 |
| ppo-forecast-narrow-w32-s11 | 0.273014 | 0.342173 | 48.207% | 0.024996 |
| ppo-forecast-narrow-w32-s23 | 0.275402 | 0.345646 | 50.238% | 0.023224 |
| ppo-forecast-narrow-w32-s37 | 0.249684 | 0.306191 | 48.407% | 0.021818 |
| ppo-forecast-narrow-w32-s53 | 0.250395 | 0.307350 | 51.308% | 0.020825 |
| ppo-forecast-narrow-w32-s71 | 0.250387 | 0.307336 | 51.261% | 0.020821 |
| supervised-labels-narrow-w128-s11 | 0.227665 | 0.267832 | 64.275% | 0.017254 |
| supervised-labels-narrow-w128-s23 | 0.180944 | 0.158156 | 73.341% | 0.007098 |
| supervised-labels-narrow-w128-s37 | 0.180248 | 0.155939 | 73.070% | 0.006634 |
| supervised-labels-narrow-w128-s53 | 0.232713 | 0.277096 | 63.310% | 0.018279 |
| supervised-labels-narrow-w128-s71 | 0.199733 | 0.209289 | 69.667% | 0.011212 |
| reinforce-forecast-narrow-w128-s11 | 0.347204 | 0.437348 | 24.980% | 0.037748 |
| reinforce-forecast-narrow-w128-s23 | 0.368772 | 0.461347 | 24.980% | 0.037748 |
| reinforce-forecast-narrow-w128-s37 | 0.347367 | 0.437534 | 24.980% | 0.037747 |
| reinforce-forecast-narrow-w128-s53 | 0.252288 | 0.310414 | 50.116% | 0.023907 |
| reinforce-forecast-narrow-w128-s71 | 0.349908 | 0.440428 | 24.980% | 0.037749 |
| ppo-forecast-narrow-w128-s11 | 0.304307 | 0.385196 | 26.559% | 0.031179 |
| ppo-forecast-narrow-w128-s23 | 0.368259 | 0.460791 | 24.980% | 0.037749 |
| ppo-forecast-narrow-w128-s37 | 0.347574 | 0.437770 | 24.980% | 0.037789 |
| ppo-forecast-narrow-w128-s53 | 0.340215 | 0.429283 | 25.611% | 0.037042 |
| ppo-forecast-narrow-w128-s71 | 0.332361 | 0.420036 | 24.980% | 0.032466 |
| supervised-labels-expanded-w128-s11 | 0.156601 | 0.025889 | 77.488% | 0.000291 |
| supervised-labels-expanded-w128-s23 | 0.156791 | 0.029323 | 77.451% | 0.000402 |
| supervised-labels-expanded-w128-s37 | 0.156808 | 0.029609 | 77.437% | 0.000370 |
| supervised-labels-expanded-w128-s53 | 0.156703 | 0.027790 | 77.496% | 0.000267 |
| supervised-labels-expanded-w128-s71 | 0.156537 | 0.024612 | 77.526% | 0.000283 |
| reinforce-forecast-expanded-w128-s11 | 0.194296 | 0.195869 | 77.136% | 0.009297 |
| reinforce-forecast-expanded-w128-s23 | 0.220582 | 0.254265 | 51.197% | 0.015648 |
| reinforce-forecast-expanded-w128-s37 | 0.193635 | 0.194175 | 77.083% | 0.009212 |
| reinforce-forecast-expanded-w128-s53 | 0.226916 | 0.266430 | 51.082% | 0.020900 |
| reinforce-forecast-expanded-w128-s71 | 0.236850 | 0.284463 | 51.061% | 0.020919 |
| ppo-forecast-expanded-w128-s11 | 0.194533 | 0.196474 | 76.458% | 0.009331 |
| ppo-forecast-expanded-w128-s23 | 0.220254 | 0.253620 | 51.250% | 0.015397 |
| ppo-forecast-expanded-w128-s37 | 0.194089 | 0.195341 | 77.195% | 0.009380 |
| ppo-forecast-expanded-w128-s53 | 0.197036 | 0.202744 | 77.300% | 0.011947 |
| ppo-forecast-expanded-w128-s71 | 0.236818 | 0.284406 | 50.745% | 0.020978 |
| supervised_continue-labels-narrow-w32-s11 | 0.208878 | 0.230102 | 67.665% | 0.013057 |
| supervised_continue-labels-narrow-w32-s23 | 0.199258 | 0.208151 | 69.787% | 0.011126 |
| supervised_continue-labels-narrow-w32-s37 | 0.173933 | 0.134170 | 75.074% | 0.005559 |
| supervised_continue-labels-narrow-w32-s53 | 0.170794 | 0.121913 | 75.560% | 0.004917 |
| supervised_continue-labels-narrow-w32-s71 | 0.183211 | 0.165167 | 73.057% | 0.007820 |
| reinforce-accuracy-narrow-w32-s11 | 0.334998 | 0.423163 | 63.444% | 0.102644 |
| reinforce-accuracy-narrow-w32-s23 | 0.333674 | 0.421596 | 63.644% | 0.102149 |
| reinforce-accuracy-narrow-w32-s37 | 0.283129 | 0.356648 | 68.581% | 0.076467 |
| reinforce-accuracy-narrow-w32-s53 | 0.283202 | 0.356750 | 68.818% | 0.077268 |
| reinforce-accuracy-narrow-w32-s71 | 0.304413 | 0.385334 | 66.697% | 0.087847 |
| ppo-accuracy-narrow-w32-s11 | 0.320878 | 0.406136 | 65.492% | 0.097758 |
| ppo-accuracy-narrow-w32-s23 | 0.314907 | 0.398718 | 66.219% | 0.095129 |
| ppo-accuracy-narrow-w32-s37 | 0.274210 | 0.343917 | 69.836% | 0.073142 |
| ppo-accuracy-narrow-w32-s53 | 0.278426 | 0.349993 | 69.802% | 0.076618 |
| ppo-accuracy-narrow-w32-s71 | 0.302691 | 0.383093 | 67.444% | 0.088939 |
| supervised_continue-labels-narrow-w128-s11 | 0.251064 | 0.308437 | 58.869% | 0.021384 |
| supervised_continue-labels-narrow-w128-s23 | 0.180944 | 0.158156 | 73.341% | 0.007098 |
| supervised_continue-labels-narrow-w128-s37 | 0.205530 | 0.222707 | 68.910% | 0.012584 |
| supervised_continue-labels-narrow-w128-s53 | 0.234968 | 0.281136 | 62.972% | 0.018871 |
| supervised_continue-labels-narrow-w128-s71 | 0.203668 | 0.218488 | 68.875% | 0.012125 |
| reinforce-accuracy-narrow-w128-s11 | 0.357605 | 0.449081 | 61.327% | 0.114345 |
| reinforce-accuracy-narrow-w128-s23 | 0.336903 | 0.425408 | 63.715% | 0.104968 |
| reinforce-accuracy-narrow-w128-s37 | 0.321858 | 0.407341 | 65.457% | 0.098216 |
| reinforce-accuracy-narrow-w128-s53 | 0.364136 | 0.456295 | 60.655% | 0.117335 |
| reinforce-accuracy-narrow-w128-s71 | 0.362458 | 0.454452 | 60.756% | 0.116391 |
| ppo-accuracy-narrow-w128-s11 | 0.352229 | 0.443055 | 62.406% | 0.113641 |
| ppo-accuracy-narrow-w128-s23 | 0.341376 | 0.430634 | 63.933% | 0.109520 |
| ppo-accuracy-narrow-w128-s37 | 0.327246 | 0.413902 | 65.179% | 0.101765 |
| ppo-accuracy-narrow-w128-s53 | 0.354325 | 0.445415 | 61.639% | 0.112805 |
| ppo-accuracy-narrow-w128-s71 | 0.354386 | 0.445483 | 62.124% | 0.114167 |
| reinforce-accuracy-narrow-w32-s11-temperature | 0.232443 | 0.276608 | 63.444% | 0.018608 |
| reinforce-accuracy-narrow-w32-s23-temperature | 0.231600 | 0.275079 | 63.644% | 0.018387 |
| reinforce-accuracy-narrow-w32-s37-temperature | 0.206061 | 0.223898 | 68.581% | 0.012963 |
| reinforce-accuracy-narrow-w32-s53-temperature | 0.203448 | 0.217984 | 68.818% | 0.011891 |
| reinforce-accuracy-narrow-w32-s71-temperature | 0.215737 | 0.244553 | 66.697% | 0.014871 |
| ppo-accuracy-narrow-w32-s11-temperature | 0.221283 | 0.255641 | 65.492% | 0.015962 |
| ppo-accuracy-narrow-w32-s23-temperature | 0.217876 | 0.248888 | 66.219% | 0.015221 |
| ppo-accuracy-narrow-w32-s37-temperature | 0.199910 | 0.209711 | 69.836% | 0.011530 |
| ppo-accuracy-narrow-w32-s53-temperature | 0.198033 | 0.205187 | 69.802% | 0.010635 |
| ppo-accuracy-narrow-w32-s71-temperature | 0.211115 | 0.234912 | 67.444% | 0.013752 |
| reinforce-accuracy-narrow-w128-s11-temperature | 0.238100 | 0.286652 | 61.327% | 0.018394 |
| reinforce-accuracy-narrow-w128-s23-temperature | 0.225942 | 0.264596 | 63.715% | 0.016088 |
| reinforce-accuracy-narrow-w128-s37-temperature | 0.218800 | 0.250736 | 65.457% | 0.014967 |
| reinforce-accuracy-narrow-w128-s53-temperature | 0.238882 | 0.288012 | 60.655% | 0.018226 |
| reinforce-accuracy-narrow-w128-s71-temperature | 0.239206 | 0.288574 | 60.756% | 0.018300 |
| ppo-accuracy-narrow-w128-s11-temperature | 0.233350 | 0.278243 | 62.406% | 0.017584 |
| ppo-accuracy-narrow-w128-s23-temperature | 0.224774 | 0.262379 | 63.933% | 0.015896 |
| ppo-accuracy-narrow-w128-s37-temperature | 0.220588 | 0.254277 | 65.179% | 0.015465 |
| ppo-accuracy-narrow-w128-s53-temperature | 0.233990 | 0.279390 | 61.639% | 0.017456 |
| ppo-accuracy-narrow-w128-s71-temperature | 0.233689 | 0.278851 | 62.124% | 0.017454 |
| oracle_posterior | 0.155931 | 0.000000 | 77.545% | 0.000000 |
| constant_half | 0.250000 | 0.306706 | 49.927% | 0.021224 |

## held_out_combination

| Recipe | Brier mean [min, max] | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| ppo-accuracy-narrow-w128 | 0.080350 [0.080350, 0.080350] | 0.130587 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w128-temperature | 0.093381 [0.084349, 0.099411] | 0.172696 | 91.965% | 0.018557 |
| ppo-accuracy-narrow-w32 | 0.080350 [0.080350, 0.080350] | 0.130587 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w32-temperature | 0.082990 [0.080422, 0.088635] | 0.139946 | 91.965% | 0.013890 |
| ppo-forecast-expanded-w128 | 0.166641 [0.140463, 0.214554] | 0.318786 | 77.952% | 0.034531 |
| ppo-forecast-narrow-w128 | 0.350605 [0.320036, 0.371756] | 0.535734 | 19.410% | 0.066599 |
| ppo-forecast-narrow-w32 | 0.215501 [0.190972, 0.227729] | 0.389673 | 59.184% | 0.045161 |
| reinforce-accuracy-narrow-w128 | 0.080350 [0.080349, 0.080350] | 0.130585 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w128-temperature | 0.095334 [0.084045, 0.101101] | 0.178061 | 91.965% | 0.019483 |
| reinforce-accuracy-narrow-w32 | 0.080350 [0.080349, 0.080350] | 0.130586 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w32-temperature | 0.085055 [0.082022, 0.090727] | 0.147175 | 91.965% | 0.014446 |
| reinforce-forecast-expanded-w128 | 0.175559 [0.139136, 0.212332] | 0.331954 | 71.368% | 0.037091 |
| reinforce-forecast-narrow-w128 | 0.323333 [0.215286, 0.364666] | 0.506538 | 30.600% | 0.063328 |
| reinforce-forecast-narrow-w32 | 0.203784 [0.157273, 0.231843] | 0.372499 | 72.594% | 0.041759 |
| supervised-labels-expanded-w128 | 0.065284 [0.064537, 0.066687] | 0.043806 | 91.972% | 0.001143 |
| supervised-labels-narrow-w128 | 0.082593 [0.071916, 0.098183] | 0.134564 | 91.965% | 0.014169 |
| supervised-labels-narrow-w32 | 0.074964 [0.072911, 0.077708] | 0.107643 | 91.965% | 0.010905 |
| supervised_continue-labels-narrow-w128 | 0.090210 [0.074615, 0.107221] | 0.159308 | 91.965% | 0.017188 |
| supervised_continue-labels-narrow-w32 | 0.077558 [0.073423, 0.082023] | 0.118767 | 91.965% | 0.011877 |

### Individual runs

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised-labels-narrow-w32-s11 | 0.076904 | 0.116649 | 91.965% | 0.012334 |
| supervised-labels-narrow-w32-s23 | 0.072911 | 0.098051 | 91.965% | 0.010117 |
| supervised-labels-narrow-w32-s37 | 0.077708 | 0.120047 | 91.965% | 0.012784 |
| supervised-labels-narrow-w32-s53 | 0.073423 | 0.100626 | 91.965% | 0.009455 |
| supervised-labels-narrow-w32-s71 | 0.073873 | 0.102841 | 91.965% | 0.009834 |
| reinforce-forecast-narrow-w32-s11 | 0.179007 | 0.340161 | 75.553% | 0.038545 |
| reinforce-forecast-narrow-w32-s23 | 0.231843 | 0.410543 | 58.396% | 0.048199 |
| reinforce-forecast-narrow-w32-s37 | 0.224833 | 0.401915 | 78.662% | 0.041451 |
| reinforce-forecast-narrow-w32-s53 | 0.225965 | 0.403321 | 58.396% | 0.046169 |
| reinforce-forecast-narrow-w32-s71 | 0.157273 | 0.306554 | 91.965% | 0.034428 |
| ppo-forecast-narrow-w32-s11 | 0.206790 | 0.378805 | 62.447% | 0.045482 |
| ppo-forecast-narrow-w32-s23 | 0.190972 | 0.357315 | 58.396% | 0.042993 |
| ppo-forecast-narrow-w32-s37 | 0.224991 | 0.402112 | 58.396% | 0.045655 |
| ppo-forecast-narrow-w32-s53 | 0.227024 | 0.404632 | 58.396% | 0.045723 |
| ppo-forecast-narrow-w32-s71 | 0.227729 | 0.405502 | 58.286% | 0.045952 |
| supervised-labels-narrow-w128-s11 | 0.088754 | 0.159553 | 91.965% | 0.016370 |
| supervised-labels-narrow-w128-s23 | 0.074615 | 0.106385 | 91.965% | 0.011576 |
| supervised-labels-narrow-w128-s37 | 0.071916 | 0.092837 | 91.965% | 0.009955 |
| supervised-labels-narrow-w128-s53 | 0.098183 | 0.186777 | 91.965% | 0.019929 |
| supervised-labels-narrow-w128-s71 | 0.079495 | 0.127270 | 91.965% | 0.013015 |
| reinforce-forecast-narrow-w128-s11 | 0.347581 | 0.533183 | 24.901% | 0.067132 |
| reinforce-forecast-narrow-w128-s23 | 0.364666 | 0.548970 | 25.012% | 0.065975 |
| reinforce-forecast-narrow-w128-s37 | 0.341246 | 0.527209 | 25.012% | 0.065804 |
| reinforce-forecast-narrow-w128-s53 | 0.215286 | 0.389858 | 53.063% | 0.050914 |
| reinforce-forecast-narrow-w128-s71 | 0.347885 | 0.533468 | 25.012% | 0.066816 |
| ppo-forecast-narrow-w128-s11 | 0.320036 | 0.506694 | 16.540% | 0.063868 |
| ppo-forecast-narrow-w128-s23 | 0.371756 | 0.555390 | 25.009% | 0.067321 |
| ppo-forecast-narrow-w128-s37 | 0.350453 | 0.535870 | 21.824% | 0.067782 |
| ppo-forecast-narrow-w128-s53 | 0.366553 | 0.550686 | 16.507% | 0.069877 |
| ppo-forecast-narrow-w128-s71 | 0.344228 | 0.530029 | 17.171% | 0.064146 |
| supervised-labels-expanded-w128-s11 | 0.065284 | 0.044579 | 91.972% | 0.001174 |
| supervised-labels-expanded-w128-s23 | 0.065286 | 0.044592 | 91.965% | 0.001185 |
| supervised-labels-expanded-w128-s37 | 0.064537 | 0.035209 | 91.965% | 0.000836 |
| supervised-labels-expanded-w128-s53 | 0.064624 | 0.036430 | 91.972% | 0.000853 |
| supervised-labels-expanded-w128-s71 | 0.066687 | 0.058221 | 91.986% | 0.001668 |
| reinforce-forecast-expanded-w128-s11 | 0.139136 | 0.275388 | 91.652% | 0.027114 |
| reinforce-forecast-expanded-w128-s23 | 0.180741 | 0.342701 | 57.082% | 0.036643 |
| reinforce-forecast-expanded-w128-s37 | 0.143229 | 0.282721 | 91.530% | 0.028004 |
| reinforce-forecast-expanded-w128-s53 | 0.202359 | 0.372909 | 57.996% | 0.047122 |
| reinforce-forecast-expanded-w128-s71 | 0.212332 | 0.386050 | 58.581% | 0.046570 |
| ppo-forecast-expanded-w128-s11 | 0.141833 | 0.280242 | 90.211% | 0.027652 |
| ppo-forecast-expanded-w128-s23 | 0.178081 | 0.338798 | 58.183% | 0.036149 |
| ppo-forecast-expanded-w128-s37 | 0.140463 | 0.277787 | 91.760% | 0.027154 |
| ppo-forecast-expanded-w128-s53 | 0.158275 | 0.308185 | 91.786% | 0.034989 |
| ppo-forecast-expanded-w128-s71 | 0.214554 | 0.388917 | 57.819% | 0.046712 |
| supervised_continue-labels-narrow-w32-s11 | 0.082023 | 0.136841 | 91.965% | 0.014182 |
| supervised_continue-labels-narrow-w32-s23 | 0.079554 | 0.127501 | 91.965% | 0.012962 |
| supervised_continue-labels-narrow-w32-s37 | 0.076041 | 0.112890 | 91.965% | 0.012308 |
| supervised_continue-labels-narrow-w32-s53 | 0.073423 | 0.100626 | 91.965% | 0.009455 |
| supervised_continue-labels-narrow-w32-s71 | 0.076748 | 0.115976 | 91.965% | 0.010480 |
| reinforce-accuracy-narrow-w32-s11 | 0.080350 | 0.130586 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w32-s23 | 0.080350 | 0.130586 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w32-s37 | 0.080349 | 0.130582 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w32-s53 | 0.080350 | 0.130587 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w32-s71 | 0.080350 | 0.130587 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w32-s11 | 0.080350 | 0.130588 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w32-s23 | 0.080350 | 0.130588 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w32-s37 | 0.080350 | 0.130586 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w32-s53 | 0.080350 | 0.130588 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w32-s71 | 0.080350 | 0.130588 | 91.965% | 0.014818 |
| supervised_continue-labels-narrow-w128-s11 | 0.107221 | 0.209580 | 91.965% | 0.024662 |
| supervised_continue-labels-narrow-w128-s23 | 0.074615 | 0.106385 | 91.965% | 0.011576 |
| supervised_continue-labels-narrow-w128-s37 | 0.086123 | 0.151083 | 91.965% | 0.015423 |
| supervised_continue-labels-narrow-w128-s53 | 0.102517 | 0.198039 | 91.965% | 0.020951 |
| supervised_continue-labels-narrow-w128-s71 | 0.080577 | 0.131451 | 91.965% | 0.013327 |
| reinforce-accuracy-narrow-w128-s11 | 0.080349 | 0.130581 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w128-s23 | 0.080350 | 0.130587 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w128-s37 | 0.080350 | 0.130587 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w128-s53 | 0.080350 | 0.130585 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w128-s71 | 0.080350 | 0.130586 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w128-s11 | 0.080350 | 0.130587 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w128-s23 | 0.080350 | 0.130588 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w128-s37 | 0.080350 | 0.130588 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w128-s53 | 0.080350 | 0.130586 | 91.965% | 0.014818 |
| ppo-accuracy-narrow-w128-s71 | 0.080350 | 0.130587 | 91.965% | 0.014818 |
| reinforce-accuracy-narrow-w32-s11-temperature | 0.085160 | 0.147861 | 91.965% | 0.014306 |
| reinforce-accuracy-narrow-w32-s23-temperature | 0.083807 | 0.143213 | 91.965% | 0.014336 |
| reinforce-accuracy-narrow-w32-s37-temperature | 0.090727 | 0.165621 | 91.965% | 0.016524 |
| reinforce-accuracy-narrow-w32-s53-temperature | 0.082022 | 0.136837 | 91.965% | 0.013486 |
| reinforce-accuracy-narrow-w32-s71-temperature | 0.083559 | 0.142344 | 91.965% | 0.013579 |
| ppo-accuracy-narrow-w32-s11-temperature | 0.083139 | 0.140861 | 91.965% | 0.013867 |
| ppo-accuracy-narrow-w32-s23-temperature | 0.080422 | 0.130863 | 91.965% | 0.013360 |
| ppo-accuracy-narrow-w32-s37-temperature | 0.088635 | 0.159180 | 91.965% | 0.015606 |
| ppo-accuracy-narrow-w32-s53-temperature | 0.080467 | 0.131032 | 91.965% | 0.013218 |
| ppo-accuracy-narrow-w32-s71-temperature | 0.082284 | 0.137794 | 91.965% | 0.013398 |
| reinforce-accuracy-narrow-w128-s11-temperature | 0.099335 | 0.189835 | 91.965% | 0.020830 |
| reinforce-accuracy-narrow-w128-s23-temperature | 0.093911 | 0.174968 | 91.965% | 0.018201 |
| reinforce-accuracy-narrow-w128-s37-temperature | 0.084045 | 0.144040 | 91.965% | 0.013833 |
| reinforce-accuracy-narrow-w128-s53-temperature | 0.098277 | 0.187028 | 91.965% | 0.021373 |
| reinforce-accuracy-narrow-w128-s71-temperature | 0.101101 | 0.194432 | 91.965% | 0.023177 |
| ppo-accuracy-narrow-w128-s11-temperature | 0.095175 | 0.178545 | 91.965% | 0.019457 |
| ppo-accuracy-narrow-w128-s23-temperature | 0.090567 | 0.165136 | 91.965% | 0.016845 |
| ppo-accuracy-narrow-w128-s37-temperature | 0.084349 | 0.145091 | 91.965% | 0.013778 |
| ppo-accuracy-narrow-w128-s53-temperature | 0.099411 | 0.190037 | 91.965% | 0.021702 |
| ppo-accuracy-narrow-w128-s71-temperature | 0.097400 | 0.184669 | 91.965% | 0.021003 |
| oracle_posterior | 0.063297 | 0.000000 | 91.999% | 0.000000 |
| constant_half | 0.250000 | 0.432091 | 50.110% | 0.050665 |
