# Complete calibration results

Expected metrics integrate event uncertainty on sampled states. Ranges span all five training seeds; they are not confidence intervals.

## in_distribution

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.156571 [0.156506, 0.156679] | 0.014335 | 0.773982 | 0.000082 |
| supervised_brier | 0.156567 [0.156518, 0.156590] | 0.014295 | 0.774004 | 0.000105 |
| reward_accuracy | 0.237924 [0.227583, 0.250825] | 0.285004 | 0.756167 | 0.062256 |
| reward_forecast | 0.169969 [0.166782, 0.175429] | 0.115947 | 0.760110 | 0.007506 |
| supervised_continue | 0.156522 [0.156496, 0.156548] | 0.012632 | 0.774077 | 0.000067 |
| accuracy_continue | 0.210807 [0.210350, 0.211571] | 0.233333 | 0.773988 | 0.045490 |
| accuracy_temperature | 0.177584 [0.171767, 0.184986] | 0.144519 | 0.756167 | 0.006212 |
| continued_temperature | 0.158858 [0.158507, 0.159202] | 0.049905 | 0.773988 | 0.001671 |
| threshold_area | 0.163347 [0.163044, 0.163723] | 0.083559 | 0.764145 | 0.002270 |
| threshold_raw | 0.217509 [0.216022, 0.219632] | 0.247266 | 0.764149 | 0.047618 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| threshold_area-seed-11 | 0.163723 | 0.085798 | 0.762759 | 0.002428 |
| threshold_raw-seed-11 | 0.219632 | 0.251536 | 0.762747 | 0.049032 |
| threshold_area-seed-23 | 0.163310 | 0.083357 | 0.763901 | 0.002174 |
| threshold_raw-seed-23 | 0.217839 | 0.247947 | 0.763909 | 0.047735 |
| threshold_area-seed-37 | 0.163079 | 0.081961 | 0.764266 | 0.002124 |
| threshold_raw-seed-37 | 0.217630 | 0.247524 | 0.764276 | 0.047717 |
| threshold_area-seed-53 | 0.163577 | 0.084940 | 0.764399 | 0.002461 |
| threshold_raw-seed-53 | 0.216022 | 0.244253 | 0.764409 | 0.046392 |
| threshold_area-seed-71 | 0.163044 | 0.081742 | 0.765400 | 0.002163 |
| threshold_raw-seed-71 | 0.216421 | 0.245069 | 0.765401 | 0.047214 |
| supervised_log-seed-11 | 0.156530 | 0.012956 | 0.774096 | 0.000075 |
| supervised_log-seed-23 | 0.156506 | 0.012012 | 0.774038 | 0.000067 |
| supervised_log-seed-37 | 0.156583 | 0.014854 | 0.773953 | 0.000077 |
| supervised_log-seed-53 | 0.156559 | 0.014032 | 0.773996 | 0.000084 |
| supervised_log-seed-71 | 0.156679 | 0.017822 | 0.773825 | 0.000106 |
| supervised_brier-seed-11 | 0.156590 | 0.015116 | 0.774008 | 0.000106 |
| supervised_brier-seed-23 | 0.156518 | 0.012491 | 0.774079 | 0.000085 |
| supervised_brier-seed-37 | 0.156568 | 0.014370 | 0.773943 | 0.000101 |
| supervised_brier-seed-53 | 0.156576 | 0.014631 | 0.774006 | 0.000128 |
| supervised_brier-seed-71 | 0.156583 | 0.014866 | 0.773984 | 0.000107 |
| reward_accuracy-seed-11 | 0.227583 | 0.266873 | 0.760575 | 0.055168 |
| reward_accuracy-seed-23 | 0.250793 | 0.307297 | 0.749029 | 0.070629 |
| reward_accuracy-seed-37 | 0.250825 | 0.307348 | 0.749029 | 0.070629 |
| reward_accuracy-seed-53 | 0.229846 | 0.271080 | 0.761017 | 0.057126 |
| reward_accuracy-seed-71 | 0.230575 | 0.272420 | 0.761187 | 0.057729 |
| reward_forecast-seed-11 | 0.166782 | 0.102081 | 0.768232 | 0.006589 |
| reward_forecast-seed-23 | 0.169732 | 0.115630 | 0.756644 | 0.007112 |
| reward_forecast-seed-37 | 0.167351 | 0.104828 | 0.755168 | 0.004752 |
| reward_forecast-seed-53 | 0.170550 | 0.119112 | 0.758043 | 0.009457 |
| reward_forecast-seed-71 | 0.175429 | 0.138085 | 0.762463 | 0.009622 |
| supervised_continue-seed-11 | 0.156506 | 0.012014 | 0.774126 | 0.000055 |
| supervised_continue-seed-23 | 0.156548 | 0.013636 | 0.773990 | 0.000080 |
| supervised_continue-seed-37 | 0.156496 | 0.011570 | 0.774135 | 0.000059 |
| supervised_continue-seed-53 | 0.156528 | 0.012879 | 0.774060 | 0.000072 |
| supervised_continue-seed-71 | 0.156532 | 0.013062 | 0.774071 | 0.000069 |
| accuracy_continue-seed-11 | 0.211141 | 0.234050 | 0.773959 | 0.045730 |
| accuracy_continue-seed-23 | 0.210569 | 0.232823 | 0.774056 | 0.045338 |
| accuracy_continue-seed-37 | 0.210404 | 0.232469 | 0.774111 | 0.045179 |
| accuracy_continue-seed-53 | 0.211571 | 0.234966 | 0.773728 | 0.046086 |
| accuracy_continue-seed-71 | 0.210350 | 0.232353 | 0.774083 | 0.045117 |
| accuracy_temperature-seed-11 | 0.171767 | 0.124116 | 0.760575 | 0.005039 |
| accuracy_temperature-seed-23 | 0.183231 | 0.163917 | 0.749029 | 0.006738 |
| accuracy_temperature-seed-37 | 0.184986 | 0.169187 | 0.749029 | 0.007467 |
| accuracy_temperature-seed-53 | 0.173996 | 0.132792 | 0.761017 | 0.005910 |
| accuracy_temperature-seed-71 | 0.173941 | 0.132585 | 0.761187 | 0.005909 |
| continued_temperature-seed-11 | 0.159202 | 0.053293 | 0.773959 | 0.001542 |
| continued_temperature-seed-23 | 0.158857 | 0.049948 | 0.774056 | 0.001865 |
| continued_temperature-seed-37 | 0.158754 | 0.048912 | 0.774111 | 0.001845 |
| continued_temperature-seed-53 | 0.158968 | 0.051050 | 0.773728 | 0.001875 |
| continued_temperature-seed-71 | 0.158507 | 0.046320 | 0.774083 | 0.001224 |
| constant_half | 0.250000 | 0.306003 | 0.499109 | 0.021126 |
| oracle_posterior | 0.156362 | 0.000000 | 0.774256 | 0.000000 |
| oracle_grid | 0.156567 | 0.014326 | 0.773808 | 0.000273 |

## weaker_sensor

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.207312 [0.207049, 0.207580] | 0.021468 | 0.679348 | 0.000141 |
| supervised_brier | 0.207261 [0.207009, 0.207537] | 0.020203 | 0.679387 | 0.000109 |
| reward_accuracy | 0.404349 [0.373514, 0.449436] | 0.442560 | 0.587132 | 0.132146 |
| reward_forecast | 0.226346 [0.221897, 0.234117] | 0.138864 | 0.614467 | 0.004154 |
| supervised_continue | 0.207239 [0.207115, 0.207428] | 0.020129 | 0.679508 | 0.000132 |
| accuracy_continue | 0.299352 [0.297342, 0.301240] | 0.304175 | 0.677235 | 0.074505 |
| accuracy_temperature | 0.272577 [0.263035, 0.286302] | 0.255565 | 0.587132 | 0.016210 |
| continued_temperature | 0.212530 [0.211516, 0.213156] | 0.075398 | 0.677235 | 0.001665 |
| threshold_area | 0.235521 [0.231661, 0.239492] | 0.169234 | 0.615943 | 0.008139 |
| threshold_raw | 0.359100 [0.349291, 0.370822] | 0.390076 | 0.615927 | 0.103857 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| threshold_area-seed-11 | 0.239492 | 0.180742 | 0.605123 | 0.009416 |
| threshold_raw-seed-11 | 0.370822 | 0.404966 | 0.605054 | 0.109907 |
| threshold_area-seed-23 | 0.236427 | 0.172053 | 0.609019 | 0.007849 |
| threshold_raw-seed-23 | 0.365471 | 0.398305 | 0.608993 | 0.106772 |
| threshold_area-seed-37 | 0.235241 | 0.168572 | 0.615374 | 0.007753 |
| threshold_raw-seed-37 | 0.359748 | 0.391055 | 0.615412 | 0.104205 |
| threshold_area-seed-53 | 0.234782 | 0.167206 | 0.623828 | 0.008708 |
| threshold_raw-seed-53 | 0.350166 | 0.378605 | 0.623814 | 0.099089 |
| threshold_area-seed-71 | 0.231661 | 0.157597 | 0.626373 | 0.006971 |
| threshold_raw-seed-71 | 0.349291 | 0.377448 | 0.626361 | 0.099314 |
| supervised_log-seed-11 | 0.207049 | 0.014980 | 0.679831 | 0.000076 |
| supervised_log-seed-23 | 0.207068 | 0.015619 | 0.679547 | 0.000042 |
| supervised_log-seed-37 | 0.207431 | 0.024627 | 0.679595 | 0.000245 |
| supervised_log-seed-53 | 0.207430 | 0.024617 | 0.679088 | 0.000151 |
| supervised_log-seed-71 | 0.207580 | 0.027495 | 0.678681 | 0.000189 |
| supervised_brier-seed-11 | 0.207498 | 0.025948 | 0.679184 | 0.000187 |
| supervised_brier-seed-23 | 0.207009 | 0.013601 | 0.679784 | 0.000035 |
| supervised_brier-seed-37 | 0.207208 | 0.019574 | 0.679427 | 0.000115 |
| supervised_brier-seed-53 | 0.207537 | 0.026699 | 0.678739 | 0.000151 |
| supervised_brier-seed-71 | 0.207055 | 0.015191 | 0.679800 | 0.000057 |
| reward_accuracy-seed-11 | 0.374604 | 0.409610 | 0.609004 | 0.114694 |
| reward_accuracy-seed-23 | 0.449392 | 0.492511 | 0.550280 | 0.157517 |
| reward_accuracy-seed-37 | 0.449436 | 0.492556 | 0.550280 | 0.157517 |
| reward_accuracy-seed-53 | 0.373514 | 0.408277 | 0.612496 | 0.114815 |
| reward_accuracy-seed-71 | 0.374799 | 0.409847 | 0.613601 | 0.116189 |
| reward_forecast-seed-11 | 0.221897 | 0.122771 | 0.636877 | 0.003120 |
| reward_forecast-seed-23 | 0.234117 | 0.165205 | 0.608949 | 0.005237 |
| reward_forecast-seed-37 | 0.223708 | 0.129935 | 0.582101 | 0.004326 |
| reward_forecast-seed-53 | 0.223528 | 0.129243 | 0.612671 | 0.004081 |
| reward_forecast-seed-71 | 0.228482 | 0.147163 | 0.631735 | 0.004007 |
| supervised_continue-seed-11 | 0.207428 | 0.024572 | 0.679460 | 0.000210 |
| supervised_continue-seed-23 | 0.207157 | 0.018225 | 0.679592 | 0.000102 |
| supervised_continue-seed-37 | 0.207135 | 0.017616 | 0.679694 | 0.000118 |
| supervised_continue-seed-53 | 0.207362 | 0.023189 | 0.679123 | 0.000133 |
| supervised_continue-seed-71 | 0.207115 | 0.017043 | 0.679670 | 0.000099 |
| accuracy_continue-seed-11 | 0.299951 | 0.305167 | 0.676960 | 0.074911 |
| accuracy_continue-seed-23 | 0.299529 | 0.304474 | 0.677351 | 0.074683 |
| accuracy_continue-seed-37 | 0.297342 | 0.300861 | 0.678111 | 0.073129 |
| accuracy_continue-seed-53 | 0.301240 | 0.307271 | 0.676182 | 0.075755 |
| accuracy_continue-seed-71 | 0.298696 | 0.303104 | 0.677568 | 0.074046 |
| accuracy_temperature-seed-11 | 0.264341 | 0.239825 | 0.609004 | 0.015782 |
| accuracy_temperature-seed-23 | 0.285506 | 0.280502 | 0.550280 | 0.015516 |
| accuracy_temperature-seed-37 | 0.286302 | 0.281917 | 0.550280 | 0.016725 |
| accuracy_temperature-seed-53 | 0.263704 | 0.238495 | 0.612496 | 0.016541 |
| accuracy_temperature-seed-71 | 0.263035 | 0.237087 | 0.613601 | 0.016484 |
| continued_temperature-seed-11 | 0.213156 | 0.079571 | 0.676960 | 0.001859 |
| continued_temperature-seed-23 | 0.212957 | 0.078309 | 0.677351 | 0.001808 |
| continued_temperature-seed-37 | 0.211516 | 0.068496 | 0.678111 | 0.001382 |
| continued_temperature-seed-53 | 0.213088 | 0.079144 | 0.676182 | 0.001793 |
| continued_temperature-seed-71 | 0.211932 | 0.071470 | 0.677568 | 0.001486 |
| constant_half | 0.250000 | 0.207787 | 0.499315 | 0.008643 |
| oracle_posterior | 0.206824 | 0.000000 | 0.679947 | 0.000000 |
| oracle_grid | 0.207033 | 0.014441 | 0.679102 | 0.000162 |

## stronger_sensor

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.047389 [0.046855, 0.047927] | 0.054628 | 0.950431 | 0.002521 |
| supervised_brier | 0.048153 [0.047496, 0.049007] | 0.061217 | 0.950431 | 0.003239 |
| reward_accuracy | 0.049633 [0.049424, 0.050187] | 0.072383 | 0.948807 | 0.004421 |
| reward_forecast | 0.090667 [0.080341, 0.110555] | 0.213748 | 0.934827 | 0.028038 |
| supervised_continue | 0.047034 [0.046442, 0.047825] | 0.051228 | 0.950431 | 0.002236 |
| accuracy_continue | 0.049225 [0.049158, 0.049288] | 0.069532 | 0.950431 | 0.004611 |
| accuracy_temperature | 0.075159 [0.071186, 0.081755] | 0.174948 | 0.948807 | 0.017677 |
| continued_temperature | 0.054835 [0.054002, 0.055505] | 0.102170 | 0.950431 | 0.011331 |
| threshold_area | 0.085485 [0.083808, 0.088930] | 0.202673 | 0.923092 | 0.017076 |
| threshold_raw | 0.065677 [0.062024, 0.073791] | 0.145259 | 0.923090 | 0.009038 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| threshold_area-seed-11 | 0.085020 | 0.201569 | 0.926632 | 0.017137 |
| threshold_raw-seed-11 | 0.064001 | 0.140039 | 0.926650 | 0.009024 |
| threshold_area-seed-23 | 0.085083 | 0.201723 | 0.926779 | 0.017246 |
| threshold_raw-seed-23 | 0.062024 | 0.132794 | 0.926838 | 0.007024 |
| threshold_area-seed-37 | 0.083808 | 0.198540 | 0.925689 | 0.016619 |
| threshold_raw-seed-37 | 0.063295 | 0.137494 | 0.925623 | 0.007923 |
| threshold_area-seed-53 | 0.088930 | 0.211044 | 0.912778 | 0.017810 |
| threshold_raw-seed-53 | 0.073791 | 0.171465 | 0.912754 | 0.012285 |
| threshold_area-seed-71 | 0.084586 | 0.200488 | 0.923583 | 0.016567 |
| threshold_raw-seed-71 | 0.065272 | 0.144505 | 0.923583 | 0.008936 |
| supervised_log-seed-11 | 0.046855 | 0.049642 | 0.950431 | 0.002195 |
| supervised_log-seed-23 | 0.047791 | 0.058314 | 0.950431 | 0.002817 |
| supervised_log-seed-37 | 0.047312 | 0.054049 | 0.950431 | 0.002451 |
| supervised_log-seed-53 | 0.047927 | 0.059466 | 0.950431 | 0.002851 |
| supervised_log-seed-71 | 0.047060 | 0.051669 | 0.950431 | 0.002292 |
| supervised_brier-seed-11 | 0.047496 | 0.055732 | 0.950431 | 0.002837 |
| supervised_brier-seed-23 | 0.048093 | 0.060852 | 0.950431 | 0.003210 |
| supervised_brier-seed-37 | 0.047957 | 0.059724 | 0.950431 | 0.003085 |
| supervised_brier-seed-53 | 0.049007 | 0.067943 | 0.950431 | 0.003780 |
| supervised_brier-seed-71 | 0.048213 | 0.061832 | 0.950431 | 0.003280 |
| reward_accuracy-seed-11 | 0.049424 | 0.070948 | 0.946761 | 0.003629 |
| reward_accuracy-seed-23 | 0.049534 | 0.071717 | 0.950431 | 0.004851 |
| reward_accuracy-seed-37 | 0.049541 | 0.071769 | 0.950431 | 0.004851 |
| reward_accuracy-seed-53 | 0.049480 | 0.071343 | 0.949026 | 0.004324 |
| reward_accuracy-seed-71 | 0.050187 | 0.076139 | 0.947384 | 0.004449 |
| reward_forecast-seed-11 | 0.081604 | 0.192910 | 0.944058 | 0.027138 |
| reward_forecast-seed-23 | 0.092139 | 0.218516 | 0.924467 | 0.029096 |
| reward_forecast-seed-37 | 0.080341 | 0.189608 | 0.948024 | 0.020812 |
| reward_forecast-seed-53 | 0.088693 | 0.210483 | 0.950431 | 0.026638 |
| reward_forecast-seed-71 | 0.110555 | 0.257225 | 0.907153 | 0.036506 |
| supervised_continue-seed-11 | 0.046442 | 0.045299 | 0.950431 | 0.001881 |
| supervised_continue-seed-23 | 0.046959 | 0.050686 | 0.950431 | 0.002263 |
| supervised_continue-seed-37 | 0.046812 | 0.049214 | 0.950431 | 0.001998 |
| supervised_continue-seed-53 | 0.047825 | 0.058608 | 0.950431 | 0.002730 |
| supervised_continue-seed-71 | 0.047129 | 0.052331 | 0.950431 | 0.002306 |
| accuracy_continue-seed-11 | 0.049247 | 0.069691 | 0.950431 | 0.004638 |
| accuracy_continue-seed-23 | 0.049209 | 0.069419 | 0.950431 | 0.004595 |
| accuracy_continue-seed-37 | 0.049222 | 0.069513 | 0.950431 | 0.004614 |
| accuracy_continue-seed-53 | 0.049158 | 0.069050 | 0.950431 | 0.004531 |
| accuracy_continue-seed-71 | 0.049288 | 0.069987 | 0.950431 | 0.004676 |
| accuracy_temperature-seed-11 | 0.071186 | 0.163696 | 0.946761 | 0.016502 |
| accuracy_temperature-seed-23 | 0.079494 | 0.187360 | 0.950431 | 0.019943 |
| accuracy_temperature-seed-37 | 0.081755 | 0.193299 | 0.950431 | 0.019771 |
| accuracy_temperature-seed-53 | 0.071558 | 0.164826 | 0.949026 | 0.016075 |
| accuracy_temperature-seed-71 | 0.071800 | 0.165558 | 0.947384 | 0.016096 |
| continued_temperature-seed-11 | 0.055505 | 0.105428 | 0.950431 | 0.010929 |
| continued_temperature-seed-23 | 0.054757 | 0.101816 | 0.950431 | 0.012278 |
| continued_temperature-seed-37 | 0.055101 | 0.103494 | 0.950431 | 0.012288 |
| continued_temperature-seed-53 | 0.054809 | 0.102074 | 0.950431 | 0.011873 |
| continued_temperature-seed-71 | 0.054002 | 0.098039 | 0.950431 | 0.009287 |
| constant_half | 0.250000 | 0.453442 | 0.502091 | 0.056108 |
| oracle_posterior | 0.044390 | 0.000000 | 0.950431 | 0.000000 |
| oracle_grid | 0.044615 | 0.015001 | 0.950431 | 0.000171 |

## extreme_prior

| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| supervised_log | 0.068502 [0.067478, 0.069632] | 0.071445 | 0.918713 | 0.002338 |
| supervised_brier | 0.068498 [0.067509, 0.069185] | 0.071481 | 0.918444 | 0.002448 |
| reward_accuracy | 0.183668 [0.122293, 0.250887] | 0.337589 | 0.807178 | 0.063562 |
| reward_forecast | 0.116093 [0.102077, 0.127835] | 0.228803 | 0.900100 | 0.026957 |
| supervised_continue | 0.068280 [0.067866, 0.068583] | 0.070035 | 0.919065 | 0.002260 |
| accuracy_continue | 0.077284 [0.076907, 0.077616] | 0.117947 | 0.919984 | 0.012333 |
| accuracy_temperature | 0.131254 [0.098583, 0.171859] | 0.254266 | 0.807178 | 0.021508 |
| continued_temperature | 0.068691 [0.068145, 0.069082] | 0.072891 | 0.919984 | 0.004929 |
| threshold_area | 0.074736 [0.072704, 0.076293] | 0.106425 | 0.919895 | 0.003223 |
| threshold_raw | 0.079486 [0.079012, 0.079837] | 0.126942 | 0.919895 | 0.014466 |

### Every run and oracle reference

| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |
| :--- | ---: | ---: | ---: | ---: |
| threshold_area-seed-11 | 0.076293 | 0.113674 | 0.919895 | 0.003405 |
| threshold_raw-seed-11 | 0.079012 | 0.125065 | 0.919895 | 0.014070 |
| threshold_area-seed-23 | 0.075881 | 0.111848 | 0.919895 | 0.003315 |
| threshold_raw-seed-23 | 0.079411 | 0.126648 | 0.919895 | 0.014604 |
| threshold_area-seed-37 | 0.074884 | 0.107300 | 0.919895 | 0.003236 |
| threshold_raw-seed-37 | 0.079450 | 0.126805 | 0.919895 | 0.014464 |
| threshold_area-seed-53 | 0.072704 | 0.096610 | 0.919895 | 0.003019 |
| threshold_raw-seed-53 | 0.079837 | 0.128320 | 0.919895 | 0.014628 |
| threshold_area-seed-71 | 0.073917 | 0.102694 | 0.919895 | 0.003139 |
| threshold_raw-seed-71 | 0.079722 | 0.127872 | 0.919895 | 0.014565 |
| supervised_log-seed-11 | 0.068037 | 0.068312 | 0.918350 | 0.002056 |
| supervised_log-seed-23 | 0.067478 | 0.064083 | 0.919584 | 0.002011 |
| supervised_log-seed-37 | 0.069632 | 0.079129 | 0.918568 | 0.002857 |
| supervised_log-seed-53 | 0.068416 | 0.071029 | 0.919253 | 0.002423 |
| supervised_log-seed-71 | 0.068947 | 0.074672 | 0.917809 | 0.002340 |
| supervised_brier-seed-11 | 0.068593 | 0.072266 | 0.917370 | 0.002427 |
| supervised_brier-seed-23 | 0.067509 | 0.064329 | 0.919367 | 0.002044 |
| supervised_brier-seed-37 | 0.069185 | 0.076249 | 0.918545 | 0.002710 |
| supervised_brier-seed-53 | 0.068251 | 0.069855 | 0.919425 | 0.002543 |
| supervised_brier-seed-71 | 0.068952 | 0.074708 | 0.917511 | 0.002517 |
| reward_accuracy-seed-11 | 0.122293 | 0.242738 | 0.854291 | 0.027469 |
| reward_accuracy-seed-23 | 0.250359 | 0.432422 | 0.748572 | 0.100289 |
| reward_accuracy-seed-37 | 0.250887 | 0.433032 | 0.748572 | 0.100289 |
| reward_accuracy-seed-53 | 0.146867 | 0.288956 | 0.841957 | 0.044353 |
| reward_accuracy-seed-71 | 0.147933 | 0.290796 | 0.842501 | 0.045407 |
| reward_forecast-seed-11 | 0.111890 | 0.220271 | 0.919895 | 0.026272 |
| reward_forecast-seed-23 | 0.102077 | 0.196739 | 0.918029 | 0.025155 |
| reward_forecast-seed-37 | 0.118379 | 0.234539 | 0.909288 | 0.020218 |
| reward_forecast-seed-53 | 0.127835 | 0.253898 | 0.833393 | 0.031408 |
| reward_forecast-seed-71 | 0.120285 | 0.238566 | 0.919895 | 0.031730 |
| supervised_continue-seed-11 | 0.068583 | 0.072192 | 0.918478 | 0.002369 |
| supervised_continue-seed-23 | 0.068554 | 0.071993 | 0.918693 | 0.002406 |
| supervised_continue-seed-37 | 0.067866 | 0.067045 | 0.919627 | 0.002045 |
| supervised_continue-seed-53 | 0.067998 | 0.068027 | 0.919506 | 0.002233 |
| supervised_continue-seed-71 | 0.068401 | 0.070921 | 0.919020 | 0.002249 |
| accuracy_continue-seed-11 | 0.077211 | 0.117644 | 0.920004 | 0.012289 |
| accuracy_continue-seed-23 | 0.077616 | 0.119352 | 0.919932 | 0.012589 |
| accuracy_continue-seed-37 | 0.077111 | 0.117218 | 0.919984 | 0.012198 |
| accuracy_continue-seed-53 | 0.077574 | 0.119176 | 0.919987 | 0.012553 |
| accuracy_continue-seed-71 | 0.076907 | 0.116346 | 0.920015 | 0.012037 |
| accuracy_temperature-seed-11 | 0.098583 | 0.187649 | 0.854291 | 0.016618 |
| accuracy_temperature-seed-23 | 0.162589 | 0.314989 | 0.748572 | 0.027586 |
| accuracy_temperature-seed-37 | 0.171859 | 0.329376 | 0.748572 | 0.027586 |
| accuracy_temperature-seed-53 | 0.111568 | 0.219538 | 0.841957 | 0.017915 |
| accuracy_temperature-seed-71 | 0.111673 | 0.219777 | 0.842501 | 0.017836 |
| continued_temperature-seed-11 | 0.068955 | 0.074724 | 0.920004 | 0.006564 |
| continued_temperature-seed-23 | 0.068145 | 0.069092 | 0.919932 | 0.004774 |
| continued_temperature-seed-37 | 0.069082 | 0.075569 | 0.919984 | 0.004426 |
| continued_temperature-seed-53 | 0.068949 | 0.074689 | 0.919987 | 0.005518 |
| continued_temperature-seed-71 | 0.068325 | 0.070382 | 0.920015 | 0.003360 |
| constant_half | 0.250000 | 0.432006 | 0.501272 | 0.050601 |
| oracle_posterior | 0.063371 | 0.000000 | 0.920134 | 0.000000 |
| oracle_grid | 0.063599 | 0.015105 | 0.920081 | 0.000152 |
