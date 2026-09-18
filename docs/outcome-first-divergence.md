# First shared-state decision divergence

This supplementary, read-only analysis explains a narrow question: at the first state where the forecast controller chooses a different action from the exact one-step controller, is its local ranking error associated with predicted terminal utility, future cost, or both? It uses existing complete raw `best-test-trajectories.jsonl` or `latest-test-trajectories.jsonl`. It performs no inference, environment execution, training, checkpoint selection, or remote access. It changes no frozen experiment inputs.

The module is `general_lab.outcome_divergence`. Its APIs are `analyze_rows(rows, provenance)`, `analyze_file(path, provenance)`, and `markdown(report)`. File analysis records the raw input checksum and checksums of this module and the original receipt validator. The nonempty provenance object is supplied by the caller: identify arm, seed, checkpoint role, checkpoint checksum, and the recovered run/freeze as available. This module does not independently certify that provenance. Analyze selected and latest checkpoints separately; a later cohort report may map all six runs to these roles without selecting among them on this diagnostic.

```sh
python -m general_lab.outcome_divergence \
  --traces /path/to/recovered/best-test-trajectories.jsonl \
  --provenance /path/to/checkpoint-provenance.json \
  --output /path/to/new-diagnostic-folder
```

The output folder must not exist. Missing or malformed receipts, incomplete trajectories, duplicate roots, invalid probability distributions, inconsistent candidate choices, or unexplained state drift raise an error. Caller-provided files are read without modification. Native and fixed-continuation receipts are checked for complete accounting by the existing report validator; the detailed alignment uses only the forecast controller and exact controller.

## Matching states

Each raw root contains a shared scenario and random tape. Both policies must have consecutive complete trajectories. At every step along their common action prefix, the analysis requires canonical equality of the public observation and full actor input, including ordered action descriptions. Candidate action order must match that public menu; outcome labels and cost labels/amounts must agree between policies. Every forecast must name the original fixed-continuation scope. Equal actions must also produce equal observations, rewards, costs, and termination. A drift is an error, not permission to match by root ID alone.

Comparison stops at the first different action, even when both actions have tied exact values. Later visited states are not joined. Each root yields one diagnostic or a same-path record. Independent forecast-audit rows are deliberately unused: their exploration prefixes need not match the policy's visited state.

## Components and local substitutions

For each candidate action, let the stored net expectation be `Q`, and compute expected future cost `C` from its cost distribution and cent-valued menu. Recover terminal expectation as `T = Q + C`. **This terminal component relies on the stored net-value receipt. It is not an independent reconstruction from outcome probabilities or a new verifier.**

At the common state let `a` be the forecast controller's action and `b` the exact controller's action. A hat denotes a model prediction; a star denotes the stored exact reference. The reported exact-menu gap is `Q*(b) - Q*(a)`. The predicted-minus-exact chosen-versus-reference margin decomposes as:

```text
terminal contribution = [T_hat(a) - T*(a)] - [T_hat(b) - T*(b)]
negative-cost contribution = -([C_hat(a) - C*(a)] - [C_hat(b) - C*(b)])
predicted margin = exact margin + terminal contribution + negative-cost contribution
```

The implementation verifies that identity. Signed contributions can cancel; the larger number alone does not identify a sufficient correction.

The terminal substitution ranks **every** candidate by `T*(x) - C_hat(x)`. The cost substitution ranks every candidate by `T_hat(x) - C*(x)`. The both-exact check ranks by stored `Q*(x)`. Each chooses the first maximum in the public menu and reports `max_x Q*(x) - Q*(chosen)`. This includes third actions and recognizes an alternative exact-optimal action as a successful local correction. The exact-value tie tolerance is an absolute `1e-8` cents; original stored choices must still follow the evaluator's strict first-maximum rule.

These substitutions do not execute an action or rerun a policy. The exact menu is one-step rollout improvement over the **fixed continuation**, not a hidden-world or globally optimal planner. The analysis makes no claim that repairing a local margin causes a corresponding change in the full adaptive controller's realized return.

## Weighting and useful follow-up

All roots remain in the denominator, including same-path roots with zero recorded local gap and first differences with tied exact values. Root means are computed within each environment; macro means equally weight environments. Conditional summaries explicitly name value-loss roots. There are no confidence intervals or claims that roots from the same mechanisms are independent experiments.

The small report gives action pairs, first-divergence depth, signed margin components, and local gaps after the two substitutions. Optional context includes only explicit public numeric clock/deadline fields and known workflow observation fields with validated types. Opaque text and rule prose are not mined for invented state categories. Full prompts, hidden tapes, and private scenarios are not copied into the report; their root identity is hashed.

If cost substitutions consistently remove local errors, design fresh training-only contrasts of action costs and continuation lengths. If terminal substitutions help, design fresh training-only evidence and outcome contrasts at similar public decision stages. If neither suffices alone, include both contrasts. The diagnostic is evidence for targeted coverage, not a reason to copy held-out roots or labels into training or to alter checkpoint selection.

Offline tests use hand-authored receipt dictionaries, including third-action corrections, exact-value ties, shared-prefix drift, incomplete inputs, and unequal environment sizes. They do not call the model or simulator.
