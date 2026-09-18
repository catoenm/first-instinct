"""Reconstruct the frozen comparison's training-world coverage without a model.

Uses only the train generator and exact planner. No held-out predictions or
outcomes select the sample. This audits roots, not policy-visited states.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from . import environments


def audit(seeds=(47, 53), updates=100, episodes_per_update=64):
    if updates < 1 or episodes_per_update < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use positive counts and distinct training seeds")
    output = {
        "scope": "Training-world coverage audit; no held-out examples or model inference",
        "code_sha256": hashlib.sha256(Path(environments.__file__).read_bytes()).hexdigest(),
        "generation": {"updates": updates, "episodes_per_update": episodes_per_update,
                       "seed_formula": "training_seed * 10**8 + update * episodes_per_update + index; update starts at 1"},
        "seeds": {},
    }
    for seed in seeds:
        counts, domains = Counter(), Counter()
        gains, margins, priors = [], [], []
        for update in range(1, updates + 1):
            for index in range(episodes_per_update):
                scenario = environments.make_episode(
                    seed * 10**8 + update * episodes_per_update + index, "train").scenario
                values = environments.action_values(scenario)
                ranked = sorted(values, key=values.get, reverse=True)
                counts[ranked[0]] += 1
                domains[scenario.domain] += 1
                gains.append(max(values.values()) - max(
                    environments.terminal_values(scenario, scenario.prior).values()))
                margins.append(values[ranked[0]] - values[ranked[1]])
                priors.append(scenario.prior)
        output["seeds"][str(seed)] = {
            "worlds": len(gains), "optimal_root_action_counts": dict(counts),
            "domains": dict(domains),
            "positive_information_value_worlds": sum(value > 1e-10 for value in gains),
            "mean_information_value": statistics.mean(gains),
            "small_top_action_margin_under_0_02": sum(value < .02 for value in margins),
            "information_value_over_0_1": sum(value > .1 for value in gains),
            "prior_range": [min(priors), max(priors)],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
