# Supplementary typed-question audit

One supervised Qwen/Qwen3.5-9B checkpoint, step 2,742. No model comparison or checkpoint selection.

| Deterministic family | Roots | Questions | Mean root accuracy | Clipped log loss | Brier score |
|---|---:|---:|---:|---:|---:|
| Routing rules | 12 | 120 | 100.0% | 0.012 | 0.001 |
| Partial-knowledge entailment | 12 | 120 | 100.0% | 0.002 | 0.000 |
| Ordered urgency | 12 | 96 | 100.0% | 0.003 | 0.000 |

Finite-experiment forecasts: 12 roots / 120 questions. Modal accuracy 100.0%; probability root mean squared error 17.6%. The latter measures error against the known conditional distribution, not error against sampled outcomes.

Exact expected clipped log score 0.779; exact expected Brier score 0.406. Even a perfect forecast has positive expected loss for a random event.

| Equivalent variant | Mean total variation | Answer flips |
|---|---:|---:|
| wording | 0.7% | 0.0% |
| opaque ids | 0.0% | 0.0% |
| irrelevant metadata | 0.3% | 0.0% |
| reordered | 0.4% | 0.0% |

Both required modal answers correct on evidence-change pairs: 100.0% (root-averaged across variants and all four families).

Opaque option-ID changes produce identical model prompts by construction. Their consistency tests serialization and response mapping, not learned semantic robustness.

All 456 questions are correlated views of 48 authored configurations from four templates. These are supplementary diagnostics, not broad calibration evidence or an independent benchmark. No inference about improvement over the foundation or parity with Jev follows from this run.

The response stream records question order by index. Model metadata and separately recorded disk hashes are checked against the freeze; they do not attest to tensors resident in server memory.

Log scores clip probabilities at 1e-12. Every metric above was recomputed from the complete saved response stream.
