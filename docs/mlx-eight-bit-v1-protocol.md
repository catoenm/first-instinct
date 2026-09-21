# Higher-precision Mac follow-up

The original adapter's unquantized MLX bridge agreed on all 34 frozen smoke
choices (mean maximum probability difference 0.00481, worst 0.02883). Four-bit
base weights preserved 32 choices but caused mean maximum probability drift
0.1071 and worst drift 0.6658, so that version failed and must not be deployed.
Preserve both results and the first setup failure, which made zero model calls.

Test **eight-bit** base linear/embedding weights, group size 64, retaining the
same float32 low-rank tensors and 36 native output rows. Keep the exact original
step-2742 checkpoint, token IDs, option order and the same 34 questions selected
before the previous scores. No additional questions, calibration fitting, training
or GPU rental occurs. This is an adaptive engineering check, not an independent
statistical validation or a change to training promotion gates.

Keep the same compression feasibility gates: at least 94% choice agreement,
mean per-question maximum probability difference at most 0.03, worst difference
at most 0.20, and inference peak memory at most 10 GiB. Measure memory after
conversion and explicitly distinguish it from conversion memory. Measure on the
current M5 Max with 128 GB; the future Mac mini remains unbenchmarked.

Passing permits a full development regression and package reload qualification,
not a demo/public replacement. A quantized copy of the existing released model
is not a newly trained checkpoint or evidence that reinforcement learning worked.
