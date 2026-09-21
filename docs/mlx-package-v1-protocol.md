# Portable eight-bit package and full regression

The eight-bit smoke check passed on the released original step-2742 adapter:
34/34 matching choices, mean maximum probability difference 0.00645, worst
0.03514, and peak inference allocation 10,452,453,324 bytes. This supports a
larger engineering check; it is not a newly trained or promoted model.

Export a local package containing the quantized text base, exact 36 float32
native output rows, the original unmodified low-rank adapter file, tokenizer
assets, pinned configuration and license. Exclude raw training/development
examples, predictions, optimizer state, credentials and vision weights. Every
file must have a digest and explicit lineage to the pinned foundation and
original adapter. Do not merge or retrain adapters. No public upload occurs here.

Reload this package in a fresh process without using the original foundation
weight files. Compare all 34 frozen probes against the eight-bit in-memory
reference; require identical choices and maximum probability difference at most
0.00001. Record process peak resident memory, inference allocation and package
bytes separately from conversion memory. This is still the M5 Max, not a Mac mini.

If reload passes, score the exact **6,072-question** frozen release development
cohort, preserving complete token IDs and options. This includes 3,465 general
questions, 2,217 tool choices, 366 outcome forecasts and 24 application decisions.
No reserved transfer question is used. Save every prediction, then independently
recompute metrics against the existing original CUDA reference.

For this serving conversion to pass, general and tool macro accuracy may each
decline by at most 0.005 and their macro log loss increase by at most 0.02. No
declared product slice may lose more than 0.01 accuracy. The 24 application
decisions must not lose accuracy. In each of the three outcome groups, expected
Brier may increase by at most 0.005 and log loss by at most 0.01. Additionally,
overall choice agreement must be at least 0.99, mean per-question maximum
probability difference at most 0.01, and worst difference at most 0.20. Peak
inference allocation must remain at most 10 GiB with one request at a time.
These are prospective serving-equivalence bounds, not relaxed training upgrade
gates: no trained continuation is promoted through this check.

Use an isolated local process, at most two hours, with incremental output and no
model/optimizer updates or rented hardware. Retain failures and partial results.
The current demo stays unchanged. Even a passing local package still requires
bounded request handling and a serving smoke check before replacing that demo;
public distribution and performance on the future Mac mini remain separate.
