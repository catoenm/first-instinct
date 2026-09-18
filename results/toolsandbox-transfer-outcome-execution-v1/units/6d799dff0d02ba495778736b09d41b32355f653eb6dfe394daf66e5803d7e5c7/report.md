# ToolSandbox transfer diagnostic: Outcome-v2 identity 6d799dff0d02

Held-out supplementary authored transfer diagnostic. All collection splits are evaluation-only. No learning, selection or tuning on this corpus.

**Primary endpoint:** expected decision value/regret at the initial state, averaged equally over 48 roots.

| Fixed-continuation action selector | Expected value | Expected regret |
|---|---:|---:|
| forecast controller | 0.208 | 20.406 |
| stop | 0.208 | 20.406 |
| complete query | -36.065 | 56.680 |
| cheap continuation | -3.669 | 24.284 |
| uniform action | -13.175 | 33.790 |
| exact menu optimum | 20.615 | 0.000 |

The exact menu optimum maximizes expected value under the same fixed continuation; it does not see the realized hidden world.

| Forecast marginal, root state | Initial-state model questions | Excess expected Brier | Exact expected clipped log loss |
|---|---:|---:|---:|
| outcome | 144 | 0.803702 | 3.868702 |
| cost | 96 | 0.492309 | 3.672786 |

Excess expected Brier equals summed squared probability error. Mean error per option is also retained; it is not directly comparable across menus with different numbers of options.

Full machine-readable results include prior-weighted phone-conditioned endpoints, an explicitly secondary equal-context macro, operation/split strata and per-root records. Phone results exclude already-paid lookup costs and are not a combined adaptive episode evaluation.

All 720 prediction questions are retained; 144 known singleton cost answers bypass prediction and are excluded from learned-forecast denominators. Expected log scores clip at 1e-12; oracle expected losses retain irreducible uncertainty.

Equal roots. Root-state endpoint is primary. Phone contexts use declared observation probabilities within each root. Known singleton costs excluded from model forecast metrics.

48 root configurations from one authored four-world mechanism and three operations; not 144 independent contexts or 720 independent tasks. No general calibration or Jev claim.
