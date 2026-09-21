# Local Mac serving qualification prototype

While the frozen supervised pilot runs, test a separate Apple-silicon inference
path for the **already released original step-2742 adapter**. This changes no
training run, checkpoint selection, demo server, public endpoint or model release.
It allocates no rented hardware. A failed supervised continuation is never the
serving reference.

Use the existing pinned Qwen3.5-9B foundation files and native label-token weights.
Load its text transformer through MLX-LM 0.31.3 with MLX 0.32.2 in an isolated
environment. Transfer all 496 trained low-rank tensors (248 projections) without
merging or retraining them. Preserve the PyTorch adapter operation's float32
low-rank calculation and final cast to the base result dtype. Verify every name,
shape, scale and adapter file hash. The exact 36 native output rows remain
float32; no newly trained classifier or generated-text classification is added.

First compare unquantized MLX against the saved original CUDA development
predictions. Freeze eight questions from each of the four existing development
suites by input-independent ID hashing, plus the shortest and longest full
development prompts. Preserve exact input token IDs and option order. No reserved
transfer questions are opened. Record every prediction, discrepancy, length,
latency and memory measurement, including cold compilation separately.

For this **smoke check only**, advancing to quantization requires at least 97%
choice agreement, average per-question maximum probability difference at most
0.02, and worst difference at most 0.10. These engineering thresholds are not a
release-retention test. If the bridge fails, diagnose it before conversion.

Then quantize base linear and embedding weights to four bits with group size 64,
keeping all trained adapters and the selected native output rows unquantized.
The same fixed probes must show at least 94% choice agreement with CUDA, mean
maximum probability difference at most 0.03, and worst difference at most 0.20.
Record resident model memory and peak memory for the longest prompt. A target
below 10 GiB for one 4,096-token request is only a feasibility screen for a
16-GB Mac; this machine is an M5 Max with 128 GB, not the user's future Mac mini.
Do not imply matching latency or guaranteed fit on a different machine.

Any public/demo replacement still needs the full frozen development regression
cohort, probability-quality and product-slice checks, package reload equivalence,
bounded concurrency and request limits. Passing this prototype alone does not
authorize promotion or establish that quantization preserved model quality.
Keep artifact lineage explicit and all development examples/predictions private.

The implementation follows the primary
[MLX-LM Qwen3.5 support](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)
and the foundation's [pinned license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE).
