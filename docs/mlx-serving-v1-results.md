# A smaller Mac package: precision matters for probabilities

The original released 9B model has a local eight-bit text-only package, but it
**failed serving qualification and has not been deployed or published**. Its
fresh-process reload reproduced all 34 smoke probes exactly. The complete
6,072-question development regression subsequently found unacceptable drift
in one consequence-forecast group.

The package keeps all 496 original float32 internal adapter tensors and the exact
36 native label-output rows in float32. It quantizes the base projections and
embeddings, omits unused vision weights and removes unused vocabulary output
rows. It does not merge adapters, retrain, generate an explanation, or add a new
classifier over generated text. Caller-supplied answer descriptions still enter
the transformer with the question and state.

| Smoke conversion | Matching choices | Mean maximum probability difference | Worst difference | Peak active inference memory |
|---|---:|---:|---:|---:|
| Unquantized MLX | 34/34 | 0.00481 | 0.02883 | 18.02 GB |
| Four-bit base | 32/34 | 0.10709 | 0.66583 | 6.48 GB |
| Eight-bit base | 34/34 | 0.00645 | 0.03514 | 10.45 GB |

Differences compare with the original CUDA predictions on the same complete
token IDs and answer menus. Mean maximum difference averages the largest
per-option change within each question. These are 34 preselected development
probes, not reserved transfer. The failed four-bit conversion shows why matching
many top choices is insufficient for a model that exposes its probabilities.

The eight-bit package occupies **8.63 GB** including the unchanged adapter and
tokenizer. After loading, the smoke process held 8.61 GB of active model memory;
peak inference allocation was 10.45 GB (9.73 GiB). Median latency after the first
probe was 0.303 seconds; the longest, 4,009-token prompt took 1.904 seconds.
These measurements used an **M5 Max with 128 GB**. They do not establish latency,
total process fit or concurrency on a 16-GB Mac mini. Resident process memory,
allocator cache and active model allocation are distinct measurements.

The pinned runtime is MLX 0.32.2, MLX-LM 0.31.3 and Transformers 5.17.0 under
Python 3.12. The implementation follows the official
[MLX Qwen3.5 model](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py),
with a separate wrapper preserving the original adapter arithmetic. The package
contains a hash manifest, pinned model revision and foundation license. The
loader uses only package files and does not download the original foundation.

The first setup attempt failed before any model call because a configuration
guard checked the wrong location for the weight-tying flag. That failure and its
source snapshot remain preserved; it did not produce a partial model result.

The [complete conversion regression](mlx-package-v1-protocol.md) finished with
6,041/6,072 matching choices (99.49%), mean maximum probability difference
0.00511 and worst difference 0.15940. General and tool accuracy, product slices,
application decisions and the aggregate probability-drift checks passed. However,
the 308-question report-forecast group increased expected Brier score from
0.64046 to 0.65086 and log loss from 1.08680 to 1.10374. Those increases exceeded
the predefined limits of 0.005 and 0.01. An independent audit reproduced the
failure. The [local API qualification](mlx-api-v1-protocol.md) therefore did not
run, and the existing released checkpoint was not replaced.

A subsequent 392-question normalization diagnostic tested reference arithmetic
and unquantized versus eight-bit execution. Correcting normalization arithmetic
slightly reduced eight-bit forecast drift, but did not meet the forecast or
probability-drift gates. It performed 1,176 labeled inference presentations and
three additional memory probes, with zero training updates or new underlying
tasks. Both failed experiments remain preserved.

After the user reported a Mac out-of-memory event, local model work and automatic
follow-through were paused and the remaining original demo server was stopped.
The reported active allocations and process measurements did not establish a
safe bound on combined host memory, allocator caches and concurrent model use.
The exact cause of the incident has not been established. No further local model
run should start without explicit memory and cache limits and a host-pressure
stop condition. None of this conversion work is new training, proof of
arbitrary-question calibration, or Jev replication.
