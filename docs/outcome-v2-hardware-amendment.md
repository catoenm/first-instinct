# Outcome decisions v2: hardware amendment

Recorded on 2026-09-18, after the original source freeze and **before any new
model inference or optimizer step**. The original protocol and freeze remain
unchanged for the audit trail.

Runpod rejected the planned two-H200 allocation with “There are no instances
currently available.” A subsequent account check confirmed no rental had been
created. Its live availability query listed two-H100 configurations, including
H100 NVL (94 GB per device) and H100 SXM (80 GB per device).

Permit **two H100 devices instead of two H200 devices** for this first launch.
Both seed workers use the same allocated device type and runtime. Keep the
$100 additional ceiling, maximum total quoted rate of $10/hour, eight-hour
independent provider stop, two-hour per-process bound, and all recovery reserves.
The total quoted availability rates were $6.38/hour for two H100 NVL devices and
$6.98/hour for two H100 SXM devices; the rental receipt records the actual rate.
No rental is started if its actual total rate exceeds the ceiling.

The starting adapter, data, splits, loss weights, learning rates, update limits,
selection gates and test procedure do not change. The original protocol permits
reducing only the microbatch if memory requires it, with full objective averaging
and a recorded change. No such reduction has been made at amendment time.
Different throughput can cause the fixed time bound to interrupt work; report
actual completed training and incomplete evaluations rather than extrapolating.

Use [freeze-h100.json](../results/outcome-v2/freeze-h100.json) for the amended run.
It pins this amendment, the original freeze, and the same source and input files.
This is an infrastructure substitution after a failed allocation, not a response
to model performance or an additional training attempt.
