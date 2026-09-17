"""Summarize every seed; estimate uncertainty by resampling whole source states."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def bootstrap_models(predictions_by_model, samples=2000, seed=20260917):
    names = list(predictions_by_model)
    reference = predictions_by_model[names[0]]
    sources = sorted({p["source_id"] for p in reference})
    tasks = sorted({p["task"] for p in reference})
    source_index = {name: i for i, name in enumerate(sources)}
    task_index = {name: i for i, name in enumerate(tasks)}
    counts = np.zeros((len(sources), len(tasks)))
    correct = np.zeros((len(names), len(sources), len(tasks)))
    pair_count = np.zeros(len(sources))
    pair_correct = np.zeros((len(names), len(sources)))
    families = {}
    reference_ids = {p["id"] for p in reference}
    for m, name in enumerate(names):
        predictions = predictions_by_model[name]
        if {p["id"] for p in predictions} != reference_ids:
            raise ValueError("Models must be compared on the same question identifiers")
        pairs = defaultdict(list)
        for p in predictions:
            s, t = source_index[p["source_id"]], task_index[p["task"]]
            families[p["source_id"]] = p["family"]
            correct[m, s, t] += p["choice"] == p["target"]
            if m == 0:
                counts[s, t] += 1
            if "contrast_id" in p:
                pairs[p["source_id"]].append(p)
        for source, pair in pairs.items():
            if len(pair) != 2 or {p["target"] for p in pair} != {"yes", "no"}:
                raise ValueError("Expected a complete opposite-answer pair")
            pair_correct[m, source_index[source]] = all(p["choice"] == p["target"] for p in pair)
            pair_count[source_index[source]] = 1
    rng = np.random.default_rng(seed)
    boot_correct = np.zeros((samples, len(names), len(tasks)))
    boot_counts = np.zeros((samples, len(tasks)))
    boot_pair_correct = np.zeros((samples, len(names)))
    boot_pair_count = np.zeros(samples)
    for family in sorted(set(families.values())):
        indices = [source_index[s] for s in sources if families[s] == family]
        multiplicities = rng.multinomial(len(indices), np.ones(len(indices)) / len(indices), size=samples)
        boot_correct += np.einsum("bs,mst->bmt", multiplicities, correct[:, indices, :])
        boot_counts += multiplicities @ counts[indices]
        boot_pair_correct += multiplicities @ pair_correct[:, indices].T
        boot_pair_count += multiplicities @ pair_count[indices]
    macro = (boot_correct / boot_counts[:, None, :]).mean(axis=-1)
    paired = boot_pair_correct / boot_pair_count[:, None]
    points = (correct.sum(axis=1) / counts.sum(axis=0)[None, :]).mean(axis=-1)
    pair_points = pair_correct.sum(axis=1) / pair_count.sum()
    models = {name: {"macro_accuracy": float(points[i]), "macro_accuracy_interval": interval(macro[:, i]),
                     "paired_accuracy": float(pair_points[i]), "paired_accuracy_interval": interval(paired[:, i])}
              for i, name in enumerate(names)}
    full = [i for i, name in enumerate(names) if name.startswith("full-")]
    frozen = [i for i, name in enumerate(names) if name.startswith("frozen-")]
    means = {}
    for label, indices in (("full", full), ("frozen", frozen)):
        if indices:
            means[label] = {"macro_accuracy": float(points[indices].mean()),
                            "macro_accuracy_seed_range": [float(points[indices].min()), float(points[indices].max())],
                            "macro_accuracy_interval": interval(macro[:, indices].mean(axis=1)),
                            "paired_accuracy": float(pair_points[indices].mean()),
                            "paired_accuracy_interval": interval(paired[:, indices].mean(axis=1))}
    if full and frozen:
        means["full_minus_frozen"] = {
            "macro_accuracy_difference": float(points[full].mean() - points[frozen].mean()),
            "macro_accuracy_difference_interval": interval(macro[:, full].mean(axis=1) - macro[:, frozen].mean(axis=1)),
            "paired_accuracy_difference": float(pair_points[full].mean() - pair_points[frozen].mean()),
            "paired_accuracy_difference_interval": interval(paired[:, full].mean(axis=1) - paired[:, frozen].mean(axis=1)),
        }
    return {"models": models, "seed_means": means, "source_states": len(sources),
            "resampling": {"samples": samples, "seed": seed, "unit": "whole source state, stratified by dataset family",
                           "interval": "95% percentile; paired resamples shared across models",
                           "scope": "Variation across the sampled source states; not uncertainty over new task families or all possible training seeds."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.run / "comparison.json").read_text())
    predictions = {name: json.loads((args.run / name / "test_canonical.json").read_text())["predictions"]
                   for name in comparison["runs"]}
    summary = bootstrap_models(predictions)
    summary["primary_run"] = comparison["primary_run"]
    summary["by_task"] = {name: {task: metrics["accuracy"] for task, metrics in result["canonical"]["by_task"].items()}
                          for name, result in comparison["runs"].items()}
    summary["variant_macro_accuracy"] = {name: {variant: result[variant]["macro_accuracy"] for variant in result}
                                         for name, result in comparison["runs"].items()}
    summary["variant_paired_accuracy"] = {name: {variant: result[variant]["contrast_pairs"]["accuracy"] for variant in result}
                                          for name, result in comparison["runs"].items()}
    summary["uniform_expected_accuracy_by_task"] = {
        task: float(np.mean([1 / len(p["probabilities"]) for p in next(iter(predictions.values())) if p["task"] == task]))
        for task in next(iter(summary["by_task"].values()))}
    destination = args.run / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"primary_run": summary["primary_run"], "seed_means": summary["seed_means"],
                      "by_task": summary["by_task"]}, indent=2))


if __name__ == "__main__":
    main()
