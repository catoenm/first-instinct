# Forecasts can support better choices than direct selection

**Final follow-up:** the run stopped at update 60 and selected update 40 using
its original rule. The [completed-run audit](appworld-supervised-v2-results.md)
confirms the update-40 figures below and reports the reserved-transfer data gate.
The interim aggregate receipt remains preserved as originally recorded.

This is an **exploratory development diagnostic while supervised v2 is running**,
using saved predictions through update 40. It was defined after the original and
update-10 aggregate development results were visible. It changes neither training
nor checkpoint selection and performs no additional model inference or world
execution. The final selected-checkpoint result must be reported separately.

On the same 57 supplied-plan decision questions, the original model's direct
choices have negative mean return despite 68.2% acceptable-choice accuracy.
Some selected plans cost more than the maximum possible completion reward.
Those plans lose money in the declared utility even if every operation succeeds.
A stopping option has nonnegative return under this contract.

| Controller | Original | Update 20 | Update 40 |
| --- | ---: | ---: | ---: |
| Model's direct choice | -0.1432 | -0.1027 | -0.0003 |
| Direct choice after excluding provably cost-dominated plans | 0.5614 | 0.6019 | 0.5891 |
| Predicted continuation success minus stated plan cost | 0.1640 | 0.3072 | 0.6423 |
| Cheapest completion rule | 0.4594 | 0.4594 | 0.4594 |
| Longest completion rule | 0.5610 | 0.5610 | 0.5610 |
| Best offered plan using executed outcomes | 0.6425 | 0.6425 | 0.6425 |

Returns are first averaged within each of two phone task programs, then across
programs. Higher is better. These questions cover five underlying worlds; costs,
menus and histories yield several questions about each world. The last row uses
ground truth and is an upper bound, not an executable controller. The fixed
completion rules inspect command shape and public costs but not the goal or
observed history; they also reject visibly invalid session handles. They are
strong baselines for these reference-assisted menus.

The forecast controller asks a separate binary question about each exact supplied
continuation, then computes `predicted success - fee × number of calls`. It also
uses the model's forecast for stopping; it does not assume that stopping means
failure. It receives no executed outcomes or hidden world state. All 57 menus
have exact matching forecasts over the same visible history and same underlying
world population. Ground truth is used only to validate those joins and score
the resulting decisions. Option ordering is matched by meaning rather than by
assuming the first answer means success.

At update 40 this controller almost reaches the best offered return on these
development questions, even though direct selection remains weak. The gain is
not merely arithmetic applied to the original model: the same forecast controller
scores 0.1640 before training. However, this is a mixed supervised recipe with
both decision and forecast examples plus replay, so it does not isolate the
effect of forecast supervision from the other learning components.

The probability estimates still need work. Continued-success expected Brier
score falls from 0.4725 to 0.2463, but unsuccessful nonempty plans retain mean
predicted success of 48.8% at update 40. Ten stopping forecasts whose executed
outcomes all fail retain mean predicted success of 26.3%. These are descriptive
row averages over this small slice, not population calibration estimates.
Good expected-return ranking does not establish calibrated probabilities.

This supports testing consequence forecasts plus explicit cost arithmetic as a
controller design. It does not establish general tool use, dynamic proposal
quality, reinforcement learning, novel architecture or Jev parity. The menus are
reference-assisted and the phone programs are exposed development data. The
controller definitions are frozen before reserved-application transfer execution
in [the transfer protocol](appworld-transfer-local-v1-protocol.md).

The [aggregate receipt](../results/appworld-controller-diagnostic-v1/summary.json)
contains all observed checkpoints, cost breakdowns and prediction hashes. Private
question and prediction files are not redistributed. Reproduce locally with:

```sh
python -m tool_lab.appworld_controller_diagnostic \
  --data output/appworld-supervised-v2-data \
  --predictions output/appworld-supervised-v2-development-snapshot \
  --steps 0 10 20 30 40 \
  --output output/controller-report.json
```
