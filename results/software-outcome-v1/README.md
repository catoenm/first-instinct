# Software outcome model: measured pilot

Selected Qwen3.5-4B adapter: update 150, 4,800 training views, 3,271 candidate programs, 4,565,626 input tokens.
The complete bounded run reached 189 updates; the released weights are the best recorded validation checkpoint, not the last weights.

Each final split evaluates 128 candidates with seven evidence views. Checkpoint selection and temperature fitting use disjoint validation source groups.
`report.json` includes Brier scores, log losses, calibration bins, source-group bootstrap comparisons and limitations.
`workflow-summary.json` plugs these forecasts into existing small acquisition policies without further training.
Compressed predictions allow inspection without downloading model weights. The full release also contains prepared requests and source snapshots.

See [the model report](../../docs/software-outcome-model.md), [the data factory](../../docs/software-inspection.md),
and [the release](https://github.com/catoenm/first-instinct/releases/tag/software-outcome-v1).

The language model was trained from outcome labels. The small acquisition policies were trained separately using Proximal Policy Optimization.
