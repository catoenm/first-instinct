# Complete calibration results

Expected metrics integrate event uncertainty on sampled states. Ranges span all five training seeds; they are not confidence intervals.

## in_distribution

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.156273 [0.156209, 0.156379] | 0.014228 | 0.774215 | 0.000081 |
| supervised_brier | 0.156271 [0.156222, 0.156294] | 0.014262 | 0.774246 | 0.000107 |
| reward_accuracy | 0.237329 [0.226494, 0.250846] | 0.284418 | 0.756671 | 0.061980 |
| reward_forecast | 0.169754 [0.166583, 0.175119] | 0.116311 | 0.760018 | 0.007572 |
| supervised_continue | 0.156224 [0.156198, 0.156250] | 0.012532 | 0.774323 | 0.000068 |
| accuracy_continue | 0.210207 [0.209735, 0.210943] | 0.232678 | 0.774275 | 0.045147 |
| accuracy_temperature | 0.177263 [0.171273, 0.185002] | 0.144321 | 0.756671 | 0.006208 |
| continued_temperature | 0.158572 [0.158220, 0.158920] | 0.050002 | 0.774275 | 0.001688 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log-seed-11 | 0.156233 | 0.012912 | 0.774317 | 0.000075 |
| supervised_log-seed-23 | 0.156209 | 0.011939 | 0.774307 | 0.000067 |
| supervised_log-seed-37 | 0.156283 | 0.014702 | 0.774178 | 0.000076 |
| supervised_log-seed-53 | 0.156260 | 0.013911 | 0.774237 | 0.000083 |
| supervised_log-seed-71 | 0.156379 | 0.017676 | 0.774033 | 0.000107 |
| supervised_brier-seed-11 | 0.156294 | 0.015088 | 0.774252 | 0.000107 |
| supervised_brier-seed-23 | 0.156222 | 0.012476 | 0.774324 | 0.000088 |
| supervised_brier-seed-37 | 0.156271 | 0.014289 | 0.774183 | 0.000103 |
| supervised_brier-seed-53 | 0.156281 | 0.014638 | 0.774269 | 0.000130 |
| supervised_brier-seed-71 | 0.156286 | 0.014820 | 0.774200 | 0.000109 |
| reward_accuracy-seed-11 | 0.226494 | 0.265381 | 0.761413 | 0.054568 |
| reward_accuracy-seed-23 | 0.250814 | 0.307810 | 0.749008 | 0.070729 |
| reward_accuracy-seed-37 | 0.250846 | 0.307863 | 0.749008 | 0.070729 |
| reward_accuracy-seed-53 | 0.228822 | 0.269732 | 0.761887 | 0.056591 |
| reward_accuracy-seed-71 | 0.229671 | 0.271302 | 0.762037 | 0.057283 |
| reward_forecast-seed-11 | 0.166583 | 0.102549 | 0.768344 | 0.006653 |
| reward_forecast-seed-23 | 0.169504 | 0.115922 | 0.756035 | 0.007169 |
| reward_forecast-seed-37 | 0.167138 | 0.105221 | 0.755589 | 0.004802 |
| reward_forecast-seed-53 | 0.170426 | 0.119832 | 0.757410 | 0.009549 |
| reward_forecast-seed-71 | 0.175119 | 0.138029 | 0.762716 | 0.009684 |
| supervised_continue-seed-11 | 0.156209 | 0.011948 | 0.774362 | 0.000057 |
| supervised_continue-seed-23 | 0.156250 | 0.013547 | 0.774231 | 0.000082 |
| supervised_continue-seed-37 | 0.156198 | 0.011447 | 0.774390 | 0.000059 |
| supervised_continue-seed-53 | 0.156229 | 0.012746 | 0.774324 | 0.000071 |
| supervised_continue-seed-71 | 0.156235 | 0.012970 | 0.774310 | 0.000068 |
| accuracy_continue-seed-11 | 0.210551 | 0.233418 | 0.774236 | 0.045372 |
| accuracy_continue-seed-23 | 0.210016 | 0.232270 | 0.774347 | 0.044986 |
| accuracy_continue-seed-37 | 0.209789 | 0.231781 | 0.774407 | 0.044829 |
| accuracy_continue-seed-53 | 0.210943 | 0.234257 | 0.773998 | 0.045744 |
| accuracy_continue-seed-71 | 0.209735 | 0.231664 | 0.774388 | 0.044806 |
| accuracy_temperature-seed-11 | 0.171273 | 0.123316 | 0.761413 | 0.005033 |
| accuracy_temperature-seed-23 | 0.183210 | 0.164753 | 0.749008 | 0.006769 |
| accuracy_temperature-seed-37 | 0.185002 | 0.170105 | 0.749008 | 0.007527 |
| accuracy_temperature-seed-53 | 0.173444 | 0.131822 | 0.761887 | 0.005856 |
| accuracy_temperature-seed-71 | 0.173387 | 0.131608 | 0.762037 | 0.005857 |
| continued_temperature-seed-11 | 0.158920 | 0.053416 | 0.774236 | 0.001551 |
| continued_temperature-seed-23 | 0.158574 | 0.050076 | 0.774347 | 0.001889 |
| continued_temperature-seed-37 | 0.158469 | 0.049012 | 0.774407 | 0.001867 |
| continued_temperature-seed-53 | 0.158678 | 0.051103 | 0.773998 | 0.001903 |
| continued_temperature-seed-71 | 0.158220 | 0.046403 | 0.774388 | 0.001232 |
| constant_half | 0.250000 | 0.306486 | 0.498793 | 0.021209 |
| oracle_posterior | 0.156066 | 0.000000 | 0.774531 | 0.000000 |
| oracle_grid | 0.156272 | 0.014336 | 0.774072 | 0.000271 |

## weaker_sensor

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.207273 [0.207007, 0.207540] | 0.021472 | 0.679211 | 0.000141 |
| supervised_brier | 0.207222 [0.206971, 0.207504] | 0.020206 | 0.679254 | 0.000108 |
| reward_accuracy | 0.403724 [0.372615, 0.449305] | 0.441851 | 0.587750 | 0.131832 |
| reward_forecast | 0.226311 [0.221822, 0.234133] | 0.138864 | 0.614006 | 0.004160 |
| supervised_continue | 0.207200 [0.207074, 0.207386] | 0.020122 | 0.679370 | 0.000131 |
| accuracy_continue | 0.299379 [0.297331, 0.301351] | 0.304284 | 0.677100 | 0.074541 |
| accuracy_temperature | 0.272224 [0.262438, 0.286285] | 0.254903 | 0.587750 | 0.016141 |
| continued_temperature | 0.212450 [0.211440, 0.213082] | 0.075126 | 0.677100 | 0.001652 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log-seed-11 | 0.207007 | 0.014894 | 0.679696 | 0.000071 |
| supervised_log-seed-23 | 0.207031 | 0.015680 | 0.679393 | 0.000043 |
| supervised_log-seed-37 | 0.207390 | 0.024598 | 0.679442 | 0.000250 |
| supervised_log-seed-53 | 0.207396 | 0.024717 | 0.678920 | 0.000152 |
| supervised_log-seed-71 | 0.207540 | 0.027471 | 0.678602 | 0.000187 |
| supervised_brier-seed-11 | 0.207453 | 0.025832 | 0.679133 | 0.000183 |
| supervised_brier-seed-23 | 0.206971 | 0.013617 | 0.679640 | 0.000034 |
| supervised_brier-seed-37 | 0.207169 | 0.019590 | 0.679265 | 0.000115 |
| supervised_brier-seed-53 | 0.207504 | 0.026799 | 0.678574 | 0.000149 |
| supervised_brier-seed-71 | 0.207016 | 0.015191 | 0.679658 | 0.000057 |
| reward_accuracy-seed-11 | 0.373687 | 0.408536 | 0.610076 | 0.114196 |
| reward_accuracy-seed-23 | 0.449261 | 0.492418 | 0.550412 | 0.157460 |
| reward_accuracy-seed-37 | 0.449305 | 0.492463 | 0.550412 | 0.157460 |
| reward_accuracy-seed-53 | 0.372615 | 0.407222 | 0.613358 | 0.114342 |
| reward_accuracy-seed-71 | 0.373753 | 0.408616 | 0.614490 | 0.115703 |
| reward_forecast-seed-11 | 0.221822 | 0.122623 | 0.636182 | 0.003147 |
| reward_forecast-seed-23 | 0.234133 | 0.165370 | 0.607837 | 0.005220 |
| reward_forecast-seed-37 | 0.223590 | 0.129634 | 0.582928 | 0.004303 |
| reward_forecast-seed-53 | 0.223537 | 0.129430 | 0.611573 | 0.004110 |
| reward_forecast-seed-71 | 0.228472 | 0.147263 | 0.631510 | 0.004021 |
| supervised_continue-seed-11 | 0.207386 | 0.024507 | 0.679335 | 0.000209 |
| supervised_continue-seed-23 | 0.207118 | 0.018228 | 0.679473 | 0.000099 |
| supervised_continue-seed-37 | 0.207095 | 0.017585 | 0.679522 | 0.000118 |
| supervised_continue-seed-53 | 0.207328 | 0.023291 | 0.678978 | 0.000134 |
| supervised_continue-seed-71 | 0.207074 | 0.016997 | 0.679540 | 0.000096 |
| accuracy_continue-seed-11 | 0.299802 | 0.304986 | 0.676779 | 0.074855 |
| accuracy_continue-seed-23 | 0.299694 | 0.304809 | 0.677274 | 0.074750 |
| accuracy_continue-seed-37 | 0.297331 | 0.300909 | 0.677988 | 0.073180 |
| accuracy_continue-seed-53 | 0.301351 | 0.307515 | 0.676022 | 0.075800 |
| accuracy_continue-seed-71 | 0.298718 | 0.303204 | 0.677438 | 0.074118 |
| accuracy_temperature-seed-11 | 0.263822 | 0.238823 | 0.610076 | 0.015678 |
| accuracy_temperature-seed-23 | 0.285463 | 0.280495 | 0.550412 | 0.015533 |
| accuracy_temperature-seed-37 | 0.286285 | 0.281957 | 0.550412 | 0.016735 |
| accuracy_temperature-seed-53 | 0.263112 | 0.237332 | 0.613358 | 0.016410 |
| accuracy_temperature-seed-71 | 0.262438 | 0.235909 | 0.614490 | 0.016349 |
| continued_temperature-seed-11 | 0.213082 | 0.079354 | 0.676779 | 0.001841 |
| continued_temperature-seed-23 | 0.212865 | 0.077972 | 0.677274 | 0.001790 |
| continued_temperature-seed-37 | 0.211440 | 0.068227 | 0.677988 | 0.001378 |
| continued_temperature-seed-53 | 0.213006 | 0.078873 | 0.676022 | 0.001774 |
| continued_temperature-seed-71 | 0.211856 | 0.071205 | 0.677438 | 0.001478 |
| constant_half | 0.250000 | 0.207881 | 0.499779 | 0.008662 |
| oracle_posterior | 0.206785 | 0.000000 | 0.679807 | 0.000000 |
| oracle_grid | 0.206993 | 0.014411 | 0.678997 | 0.000165 |

## stronger_sensor

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.047664 [0.047116, 0.048214] | 0.055149 | 0.950168 | 0.002547 |
| supervised_brier | 0.048426 [0.047756, 0.049297] | 0.061663 | 0.950168 | 0.003255 |
| reward_accuracy | 0.050057 [0.049796, 0.050793] | 0.073776 | 0.948191 | 0.004492 |
| reward_forecast | 0.091013 [0.080536, 0.110964] | 0.214029 | 0.933708 | 0.028043 |
| supervised_continue | 0.047304 [0.046698, 0.048114] | 0.051740 | 0.950168 | 0.002265 |
| accuracy_continue | 0.049479 [0.049412, 0.049543] | 0.069795 | 0.950168 | 0.004649 |
| accuracy_temperature | 0.075441 [0.071577, 0.081896] | 0.175156 | 0.948191 | 0.017672 |
| continued_temperature | 0.055143 [0.054307, 0.055810] | 0.102612 | 0.950168 | 0.011334 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log-seed-11 | 0.047116 | 0.050082 | 0.950168 | 0.002208 |
| supervised_log-seed-23 | 0.048068 | 0.058822 | 0.950168 | 0.002845 |
| supervised_log-seed-37 | 0.047588 | 0.054595 | 0.950168 | 0.002480 |
| supervised_log-seed-53 | 0.048214 | 0.060052 | 0.950168 | 0.002883 |
| supervised_log-seed-71 | 0.047332 | 0.052193 | 0.950168 | 0.002318 |
| supervised_brier-seed-11 | 0.047756 | 0.056111 | 0.950168 | 0.002839 |
| supervised_brier-seed-23 | 0.048363 | 0.061281 | 0.950168 | 0.003227 |
| supervised_brier-seed-37 | 0.048230 | 0.060189 | 0.950168 | 0.003107 |
| supervised_brier-seed-53 | 0.049297 | 0.068476 | 0.950168 | 0.003809 |
| supervised_brier-seed-71 | 0.048484 | 0.062260 | 0.950168 | 0.003294 |
| reward_accuracy-seed-11 | 0.049926 | 0.072928 | 0.945752 | 0.003704 |
| reward_accuracy-seed-23 | 0.049796 | 0.072027 | 0.950168 | 0.004898 |
| reward_accuracy-seed-37 | 0.049803 | 0.072080 | 0.950168 | 0.004898 |
| reward_accuracy-seed-53 | 0.049966 | 0.073199 | 0.948277 | 0.004398 |
| reward_accuracy-seed-71 | 0.050793 | 0.078648 | 0.946587 | 0.004558 |
| reward_forecast-seed-11 | 0.081871 | 0.193036 | 0.943674 | 0.027156 |
| reward_forecast-seed-23 | 0.092883 | 0.219716 | 0.922284 | 0.029103 |
| reward_forecast-seed-37 | 0.080536 | 0.189548 | 0.947819 | 0.020842 |
| reward_forecast-seed-53 | 0.088811 | 0.210246 | 0.950168 | 0.026615 |
| reward_forecast-seed-71 | 0.110964 | 0.257597 | 0.904596 | 0.036500 |
| supervised_continue-seed-11 | 0.046698 | 0.045718 | 0.950168 | 0.001905 |
| supervised_continue-seed-23 | 0.047223 | 0.051141 | 0.950168 | 0.002292 |
| supervised_continue-seed-37 | 0.047085 | 0.049777 | 0.950168 | 0.002027 |
| supervised_continue-seed-53 | 0.048114 | 0.059210 | 0.950168 | 0.002771 |
| supervised_continue-seed-71 | 0.047402 | 0.052856 | 0.950168 | 0.002332 |
| accuracy_continue-seed-11 | 0.049501 | 0.069950 | 0.950168 | 0.004676 |
| accuracy_continue-seed-23 | 0.049462 | 0.069674 | 0.950168 | 0.004634 |
| accuracy_continue-seed-37 | 0.049478 | 0.069786 | 0.950168 | 0.004651 |
| accuracy_continue-seed-53 | 0.049412 | 0.069315 | 0.950168 | 0.004572 |
| accuracy_continue-seed-71 | 0.049543 | 0.070253 | 0.950168 | 0.004713 |
| accuracy_temperature-seed-11 | 0.071577 | 0.164224 | 0.945752 | 0.016536 |
| accuracy_temperature-seed-23 | 0.079660 | 0.187223 | 0.950168 | 0.019951 |
| accuracy_temperature-seed-37 | 0.081896 | 0.193101 | 0.950168 | 0.019735 |
| accuracy_temperature-seed-53 | 0.071910 | 0.165233 | 0.948277 | 0.016061 |
| accuracy_temperature-seed-71 | 0.072163 | 0.165999 | 0.946587 | 0.016075 |
| continued_temperature-seed-11 | 0.055810 | 0.105842 | 0.950168 | 0.010916 |
| continued_temperature-seed-23 | 0.055052 | 0.102199 | 0.950168 | 0.012312 |
| continued_temperature-seed-37 | 0.055414 | 0.103954 | 0.950168 | 0.012321 |
| continued_temperature-seed-53 | 0.055131 | 0.102583 | 0.950168 | 0.011873 |
| continued_temperature-seed-71 | 0.054307 | 0.098482 | 0.950168 | 0.009247 |
| constant_half | 0.250000 | 0.453202 | 0.498928 | 0.055961 |
| oracle_posterior | 0.044608 | 0.000000 | 0.950168 | 0.000000 |
| oracle_grid | 0.044835 | 0.015082 | 0.950168 | 0.000176 |

## extreme_prior

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.068201 [0.067181, 0.069326] | 0.071220 | 0.918961 | 0.002342 |
| supervised_brier | 0.068198 [0.067216, 0.068882] | 0.071267 | 0.918665 | 0.002455 |
| reward_accuracy | 0.183471 [0.122518, 0.249820] | 0.337925 | 0.807134 | 0.063532 |
| reward_forecast | 0.115883 [0.102135, 0.127391] | 0.228953 | 0.900611 | 0.027021 |
| supervised_continue | 0.067993 [0.067583, 0.068304] | 0.069900 | 0.919266 | 0.002270 |
| accuracy_continue | 0.076899 [0.076530, 0.077241] | 0.117452 | 0.920260 | 0.012215 |
| accuracy_temperature | 0.131123 [0.098611, 0.171280] | 0.254677 | 0.807134 | 0.021579 |
| continued_temperature | 0.068441 [0.067900, 0.068839] | 0.073018 | 0.920260 | 0.004928 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log-seed-11 | 0.067761 | 0.068260 | 0.918595 | 0.002052 |
| supervised_log-seed-23 | 0.067181 | 0.063869 | 0.919835 | 0.001999 |
| supervised_log-seed-37 | 0.069326 | 0.078895 | 0.918809 | 0.002870 |
| supervised_log-seed-53 | 0.068110 | 0.070767 | 0.919450 | 0.002430 |
| supervised_log-seed-71 | 0.068624 | 0.074310 | 0.918118 | 0.002362 |
| supervised_brier-seed-11 | 0.068321 | 0.072246 | 0.917630 | 0.002417 |
| supervised_brier-seed-23 | 0.067216 | 0.064138 | 0.919511 | 0.002035 |
| supervised_brier-seed-37 | 0.068882 | 0.076028 | 0.918802 | 0.002732 |
| supervised_brier-seed-53 | 0.067951 | 0.069637 | 0.919573 | 0.002553 |
| supervised_brier-seed-71 | 0.068621 | 0.074288 | 0.917808 | 0.002536 |
| reward_accuracy-seed-11 | 0.122518 | 0.243755 | 0.853089 | 0.027446 |
| reward_accuracy-seed-23 | 0.249301 | 0.431508 | 0.749635 | 0.099872 |
| reward_accuracy-seed-37 | 0.249820 | 0.432108 | 0.749635 | 0.099872 |
| reward_accuracy-seed-53 | 0.147345 | 0.290247 | 0.841350 | 0.044699 |
| reward_accuracy-seed-71 | 0.148369 | 0.292006 | 0.841958 | 0.045772 |
| reward_forecast-seed-11 | 0.111513 | 0.220026 | 0.920118 | 0.026231 |
| reward_forecast-seed-23 | 0.102135 | 0.197569 | 0.917735 | 0.025363 |
| reward_forecast-seed-37 | 0.117954 | 0.234205 | 0.909180 | 0.020283 |
| reward_forecast-seed-53 | 0.127391 | 0.253553 | 0.835906 | 0.031386 |
| reward_forecast-seed-71 | 0.120421 | 0.239415 | 0.920118 | 0.031844 |
| supervised_continue-seed-11 | 0.068304 | 0.072127 | 0.918631 | 0.002385 |
| supervised_continue-seed-23 | 0.068275 | 0.071924 | 0.918869 | 0.002413 |
| supervised_continue-seed-37 | 0.067583 | 0.066941 | 0.919894 | 0.002062 |
| supervised_continue-seed-53 | 0.067694 | 0.067767 | 0.919732 | 0.002231 |
| supervised_continue-seed-71 | 0.068106 | 0.070739 | 0.919204 | 0.002260 |
| accuracy_continue-seed-11 | 0.076784 | 0.116970 | 0.920263 | 0.012139 |
| accuracy_continue-seed-23 | 0.077228 | 0.118852 | 0.920205 | 0.012469 |
| accuracy_continue-seed-37 | 0.076710 | 0.116654 | 0.920275 | 0.012058 |
| accuracy_continue-seed-53 | 0.077241 | 0.118909 | 0.920258 | 0.012482 |
| accuracy_continue-seed-71 | 0.076530 | 0.115879 | 0.920302 | 0.011927 |
| accuracy_temperature-seed-11 | 0.098611 | 0.188439 | 0.853089 | 0.016724 |
| accuracy_temperature-seed-23 | 0.162111 | 0.314656 | 0.749635 | 0.027612 |
| accuracy_temperature-seed-37 | 0.171280 | 0.328904 | 0.749635 | 0.027612 |
| accuracy_temperature-seed-53 | 0.111742 | 0.220544 | 0.841350 | 0.018008 |
| accuracy_temperature-seed-71 | 0.111872 | 0.220840 | 0.841958 | 0.017938 |
| continued_temperature-seed-11 | 0.068712 | 0.074901 | 0.920263 | 0.006543 |
| continued_temperature-seed-23 | 0.067900 | 0.069268 | 0.920205 | 0.004755 |
| continued_temperature-seed-37 | 0.068839 | 0.075743 | 0.920275 | 0.004445 |
| continued_temperature-seed-53 | 0.068684 | 0.074715 | 0.920258 | 0.005500 |
| continued_temperature-seed-71 | 0.068067 | 0.070465 | 0.920302 | 0.003396 |
| constant_half | 0.250000 | 0.432317 | 0.505495 | 0.050799 |
| oracle_posterior | 0.063102 | 0.000000 | 0.920415 | 0.000000 |
| oracle_grid | 0.063328 | 0.015045 | 0.920352 | 0.000151 |
