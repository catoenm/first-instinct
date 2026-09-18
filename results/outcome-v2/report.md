# Outcome-v2 recovered experiment

Report status: **partial**.

| Run | Status | Updates | Steps | Selected update | Best retention eligible | Latest retention eligible |
| --- | --- | ---: | ---: | --- | --- | --- |
| outcome-s77 | complete | 60 | 120 | 60 | True | True |
| outcome-s83 | complete | 60 | 120 | 60 | True | True |
| reward-s77 | complete | 60 | 120 | 40 | True | True |
| reward-s83 | complete | 60 | 120 | 60 | True | True |
| hybrid-s77 | bounded_stop | 58 | 116 | 40 | — | — |
| hybrid-s83 | complete | 60 | 120 | 60 | True | True |

## Realized test rewards

Equal environment weighting; 100 cents = 1 reward unit.

| Run / checkpoint | Environment | Roots | Controller | Native | Fixed | Exact reference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| outcome-s77 / best | retry | 128 | 0.3169 | 0.2441 | 0.2944 | 0.5362 |
| outcome-s77 / best | workflow | 128 | 0.1727 | 0.1670 | 0.1585 | 0.3277 |
| outcome-s77 / latest | retry | 128 | 0.3169 | 0.2441 | 0.2944 | 0.5362 |
| outcome-s77 / latest | workflow | 128 | 0.1727 | 0.1670 | 0.1585 | 0.3277 |
| outcome-s83 / best | retry | 128 | 0.3501 | 0.2310 | 0.2944 | 0.5362 |
| outcome-s83 / best | workflow | 128 | 0.1871 | 0.1652 | 0.1585 | 0.3277 |
| outcome-s83 / latest | retry | 128 | 0.3501 | 0.2310 | 0.2944 | 0.5362 |
| outcome-s83 / latest | workflow | 128 | 0.1871 | 0.1652 | 0.1585 | 0.3277 |
| reward-s77 / best | retry | 128 | 0.2730 | 0.2356 | 0.2944 | 0.5362 |
| reward-s77 / best | workflow | 128 | -0.4401 | 0.1883 | 0.1585 | 0.3277 |
| reward-s77 / latest | retry | 128 | 0.1942 | 0.1462 | 0.2944 | 0.5362 |
| reward-s77 / latest | workflow | 128 | -0.4373 | 0.1956 | 0.1585 | 0.3277 |
| reward-s83 / best | retry | 128 | 0.1701 | 0.1921 | 0.2944 | 0.5362 |
| reward-s83 / best | workflow | 128 | -0.4387 | 0.2000 | 0.1585 | 0.3277 |
| reward-s83 / latest | retry | 128 | 0.1701 | 0.1921 | 0.2944 | 0.5362 |
| reward-s83 / latest | workflow | 128 | -0.4387 | 0.2000 | 0.1585 | 0.3277 |
| hybrid-s77 / best | missing | — | — | — | — | — |
| hybrid-s77 / latest | missing | — | — | — | — | — |
| hybrid-s83 / best | retry | 128 | 0.2706 | 0.3383 | 0.2944 | 0.5362 |
| hybrid-s83 / best | workflow | 128 | 0.1493 | 0.2271 | 0.1585 | 0.3277 |
| hybrid-s83 / latest | retry | 128 | 0.2706 | 0.3383 | 0.2944 | 0.5362 |
| hybrid-s83 / latest | workflow | 128 | 0.1493 | 0.2271 | 0.1585 | 0.3277 |

## Independent fixed-continuation audit

| Run / checkpoint | Environment | Outcome Brier | Outcome log loss | Outcome exact MSE | Cost Brier | Cost log loss | Cost exact MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| outcome-s77 / best | retry | 0.3711 | 0.6315 | 0.0222 | 0.4043 | 0.7932 | 0.0598 |
| outcome-s77 / best | workflow | 0.2264 | 0.3622 | 0.0245 | 0.4603 | 0.9604 | 0.0437 |
| outcome-s77 / latest | retry | 0.3711 | 0.6315 | 0.0222 | 0.4043 | 0.7932 | 0.0598 |
| outcome-s77 / latest | workflow | 0.2264 | 0.3622 | 0.0245 | 0.4603 | 0.9604 | 0.0437 |
| outcome-s83 / best | retry | 0.3683 | 0.6331 | 0.0217 | 0.4034 | 0.7918 | 0.0595 |
| outcome-s83 / best | workflow | 0.2326 | 0.3707 | 0.0262 | 0.4607 | 0.9646 | 0.0440 |
| outcome-s83 / latest | retry | 0.3683 | 0.6331 | 0.0217 | 0.4034 | 0.7918 | 0.0595 |
| outcome-s83 / latest | workflow | 0.2326 | 0.3707 | 0.0262 | 0.4607 | 0.9646 | 0.0440 |
| reward-s77 / best | retry | 0.6397 | 1.0627 | 0.1007 | 0.6320 | 1.9769 | 0.1148 |
| reward-s77 / best | workflow | 0.5199 | 1.0953 | 0.1235 | 0.5256 | 1.1562 | 0.0548 |
| reward-s77 / latest | retry | 0.6840 | 1.1827 | 0.1142 | 0.6464 | 2.1398 | 0.1185 |
| reward-s77 / latest | workflow | 0.5336 | 1.2030 | 0.1282 | 0.5318 | 1.1780 | 0.0565 |
| reward-s83 / best | retry | 0.6597 | 1.1408 | 0.1068 | 0.6628 | 2.3106 | 0.1230 |
| reward-s83 / best | workflow | 0.5283 | 1.1689 | 0.1269 | 0.5615 | 1.3117 | 0.0588 |
| reward-s83 / latest | retry | 0.6597 | 1.1408 | 0.1068 | 0.6628 | 2.3106 | 0.1230 |
| reward-s83 / latest | workflow | 0.5283 | 1.1689 | 0.1269 | 0.5615 | 1.3117 | 0.0588 |
| hybrid-s83 / best | retry | 0.3753 | 0.6460 | 0.0234 | 0.4129 | 0.8113 | 0.0619 |
| hybrid-s83 / best | workflow | 0.2587 | 0.4201 | 0.0344 | 0.4678 | 0.9813 | 0.0463 |
| hybrid-s83 / latest | retry | 0.3753 | 0.6460 | 0.0234 | 0.4129 | 0.8113 | 0.0619 |
| hybrid-s83 / latest | workflow | 0.2587 | 0.4201 | 0.0344 | 0.4678 | 0.9813 | 0.0463 |

## Paired controller differences

Positive differences favor hybrid. Intervals describe root draws within these environment families.

| Comparison | Training seed | Difference | 95% interval | Status |
| --- | --- | ---: | --- | --- |
| hybrid-minus-reward/best | 77 | — | — | unavailable |
| hybrid-minus-reward/best | 83 | 0.3443 | [0.2294, 0.4614] | complete |
| hybrid-minus-reward/best | 77+83, fixed | — | — | unavailable |
| hybrid-minus-reward/latest | 77 | — | — | unavailable |
| hybrid-minus-reward/latest | 83 | 0.3443 | [0.2294, 0.4614] | complete |
| hybrid-minus-reward/latest | 77+83, fixed | — | — | unavailable |
| hybrid-minus-outcome/best | 77 | — | — | unavailable |
| hybrid-minus-outcome/best | 83 | -0.0586 | [-0.1294, 0.0052] | complete |
| hybrid-minus-outcome/best | 77+83, fixed | — | — | unavailable |
| hybrid-minus-outcome/latest | 77 | — | — | unavailable |
| hybrid-minus-outcome/latest | 83 | -0.0586 | [-0.1294, 0.0052] | complete |
| hybrid-minus-outcome/latest | 77+83, fixed | — | — | unavailable |

- hybrid-minus-reward/best / 77: Missing or invalid run/checkpoint artifacts.
- hybrid-minus-reward/latest / 77: Missing or invalid run/checkpoint artifacts.
- hybrid-minus-outcome/best / 77: Missing or invalid run/checkpoint artifacts.
- hybrid-minus-outcome/latest / 77: Missing or invalid run/checkpoint artifacts.

## Interpretation limits

- Rewards are realized execution cents / 100, with equal environment weighting.
- Bootstrap intervals resample paired root draws within each environment; related roots do not constitute independent mechanisms.
- The two training seeds are held fixed. Their agreement is not statistical proof of training-seed robustness or generalization.
- Hybrid receives additional outcome/cost labels and simulator work. Unequal committed training doses are reported, not controlled away.
- Independent audit errors describe the declared fixed continuation, not end-to-end probabilities of the replanning controller.
- The exact controller is one-step rollout improvement with replanning, not a fully optimal planner.
- No baseline test result is invented. A complete best test with selected_update=0 is explicitly identified as the starting adapter.

Detailed retention changes, training work, missing-file/error receipts, inference counts, and input hashes are in `report.json`.
