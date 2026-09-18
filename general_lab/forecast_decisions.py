"""Post-hoc, offline decision consequences of saved event forecasts.

At each of the four saved evidence masks, a fixed controller stops immediately
and chooses affirm, deny, or abstain using the forecast and declared payoffs.
Its terminal reward is integrated exactly against the true posterior. This is
not the learned sequential policy: no further checks can be purchased, past
inspection costs are sunk, and the four sampled evidence states have equal
weight rather than policy visitation weights. No native actor is conditioned or
renormalized. This diagnostic never performs inference, training, or selection.

The four masks share a sampled world/outcome. Paired bootstrap intervals resample
whole worlds; they do not cover training-seed variation. Inputs must come from
completed runs whose recorded environment hash matches this frozen generator.
Every declared evaluation split and baseline/best/latest trace is required.

Example: python -m general_lab.forecast_decisions --runs runs/rl-s47-hybrid \
    --output output/forecast-decisions
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import re

from scale_lab.common import MODELS, file_hash, write_json
from . import environments
from .environments import make_episode, posterior, terminal_values
from .report import cluster_bootstrap, finite, read_json, read_lines


EVALUATION_SEED = 71043
CHECKPOINTS = ("baseline", "best", "latest")
SPLITS = {"validation", "test", "shift", "new_domain"}
ACTIONS = ("affirm", "deny", "abstain")
FORECAST_NAME = re.compile(r"(baseline|best|latest)-(.+)-forecasts\.jsonl$")
ENVIRONMENT_PATH = Path(environments.__file__).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def terminal_decision(scenario, probability):
    """Maximize forecast utility; exact ties prefer affirm, then deny, then abstain."""
    probability = finite(probability, "Forecast probability", 0, 1)
    values = terminal_values(scenario, probability)
    return max(ACTIONS, key=values.__getitem__)


def score_decision(scenario, probability, true_posterior):
    """Evaluate a forced-stop decision, excluding all past and future check costs."""
    true_posterior = finite(true_posterior, "True posterior", 0, 1)
    action = terminal_decision(scenario, probability)
    oracle_action = terminal_decision(scenario, true_posterior)
    values = terminal_values(scenario, true_posterior)
    reward, oracle_reward = values[action], values[oracle_action]
    regret = oracle_reward - reward
    require(math.isfinite(reward) and math.isfinite(regret) and regret >= 0,
            "Invalid expected terminal reward or regret")
    return {"action": action, "terminal_oracle_action": oracle_action,
            "expected_terminal_reward": reward, "terminal_oracle_expected_reward": oracle_reward,
            "terminal_regret": regret, "oracle_action_mismatch": action != oracle_action,
            "strictly_suboptimal": regret > 1e-12}


def validate_receipt(receipt):
    require(isinstance(receipt, dict) and receipt.get("status") == "complete",
            "Forecast diagnostic requires a completed reinforcement run")
    require(receipt.get("model") in MODELS.values(), "Run foundation/revision is not supported")
    config = receipt.get("config")
    require(isinstance(config, dict) and positive_int(config.get("eval_episodes")),
            "Missing or invalid evaluation world count")
    splits = config.get("eval_splits")
    require(isinstance(splits, list) and splits and all(isinstance(s, str) and s in SPLITS for s in splits)
            and len(splits) == len(set(splits)) and "validation" in splits,
            "Missing or invalid declared evaluation splits")
    selection = receipt.get("checkpoint_selection")
    require(isinstance(selection, dict) and positive_int(selection.get("evaluation_seed"))
            and selection["evaluation_seed"] == EVALUATION_SEED,
            "Recorded evaluation seed does not match the frozen reconstruction")
    hashes = receipt.get("code_sha256")
    require(isinstance(hashes, dict) and hashes.get("general_lab/environments.py") == file_hash(ENVIRONMENT_PATH),
            "Recorded environment code hash differs from the frozen reconstruction")
    updates, selected = receipt.get("updates"), receipt.get("selected_update")
    require(positive_int(updates) and positive_int(receipt.get("optimizer_steps"))
            and isinstance(selected, int) and not isinstance(selected, bool) and 0 <= selected <= updates,
            "Missing or invalid completed-update/checkpoint-selection receipt")
    return config["eval_episodes"], splits


def expected_states(split, count):
    """Reconstruct the exact sampled worlds used by general_lab.rl.evaluate."""
    result = {}
    for index in range(count):
        episode = make_episode(EVALUATION_SEED * 100003 + index, split)
        for mask in range(4):
            key = (episode.scenario.identity, mask)
            require(key not in result, "Duplicate reconstructed world/mask")
            result[key] = {"scenario": episode.scenario, "domain": episode.scenario.domain,
                           "outcome": int(episode.truth),
                           "posterior": posterior(episode.scenario, episode.observations(mask))}
    return result


def validated_forecasts(path, expected):
    """Verify complete identity, event, evidence mask, and posterior provenance."""
    result = {}
    for row in read_lines(path):
        require(isinstance(row, dict), "Forecast rows must be objects")
        identity, mask = row.get("id"), row.get("mask")
        require(isinstance(identity, str) and isinstance(mask, int) and not isinstance(mask, bool)
                and mask in range(4), "Invalid forecast world identity or evidence mask")
        key = (identity, mask)
        require(key in expected and key not in result, "Unknown or duplicate forecast world/mask")
        world = expected[key]
        require(row.get("domain") == world["domain"], "Forecast domain differs from reconstructed world")
        outcome = row.get("outcome")
        require(isinstance(outcome, int) and not isinstance(outcome, bool)
                and outcome == world["outcome"], "Forecast outcome differs from reconstructed world")
        q = finite(row.get("posterior"), "Saved exact posterior", 0, 1)
        require(math.isclose(q, world["posterior"], rel_tol=1e-12, abs_tol=1e-12),
                "Saved posterior differs from reconstructed evidence")
        result[key] = finite(row.get("forecast"), "Saved forecast", 0, 1)
    require(result.keys() == expected.keys(), "Forecast trace is missing required world/mask states")
    return result


def summarize_decisions(rows):
    actions = Counter(row["action"] for row in rows)
    oracle_actions = Counter(row["terminal_oracle_action"] for row in rows)
    mismatches = sum(row["oracle_action_mismatch"] for row in rows)
    return {"states": len(rows), "worlds": len({row["id"] for row in rows}),
            "mean_expected_terminal_reward": sum(row["expected_terminal_reward"] for row in rows) / len(rows),
            "mean_terminal_oracle_expected_reward": sum(row["terminal_oracle_expected_reward"] for row in rows) / len(rows),
            "mean_terminal_regret": sum(row["terminal_regret"] for row in rows) / len(rows),
            "oracle_action_mismatch_count": mismatches, "oracle_action_mismatch_fraction": mismatches / len(rows),
            "strictly_suboptimal_count": sum(row["strictly_suboptimal"] for row in rows),
            "action_counts": {action: actions[action] for action in ACTIONS},
            "terminal_oracle_action_counts": {action: oracle_actions[action] for action in ACTIONS},
            "action_confusion_counts": {oracle: {action: sum(row["terminal_oracle_action"] == oracle
                                                         and row["action"] == action for row in rows)
                                                   for action in ACTIONS} for oracle in ACTIONS}}


def analyze_run(folder, *, seed=41, resamples=1000):
    """Analyze all declared splits. Any missing or inconsistent input aborts the run."""
    require(positive_int(resamples), "Bootstrap resamples must be positive")
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "Invalid bootstrap seed")
    folder = Path(folder).resolve()
    receipt_path = folder / "run.json"
    receipt = read_json(receipt_path)
    count, splits = validate_receipt(receipt)
    required = {f"{checkpoint}-{split}-forecasts.jsonl" for checkpoint in CHECKPOINTS for split in splits}
    found = {path.name for path in folder.iterdir() if FORECAST_NAME.fullmatch(path.name)}
    require(required == found and all((folder / name).is_file() for name in required),
            "Forecast files do not exactly cover all declared splits and checkpoints")
    hashes = {"run.json": file_hash(receipt_path), **{name: file_hash(folder / name) for name in sorted(required)}}
    output = {"name": folder.name, "source": str(folder), "status": receipt["status"], "model": receipt["model"],
              "training_seed": receipt["config"].get("seed"), "completed_updates": receipt["updates"],
              "selected_update": receipt["selected_update"], "best_contains_rl_updates": receipt["selected_update"] > 0,
              "checkpoint_note": ("Selected update zero: best is the supervised baseline, not an RL-trained selection."
                                  if receipt["selected_update"] == 0 else "Best uses the run's original selected update; this diagnostic does not reselect it."),
              "environment_sha256": receipt["code_sha256"]["general_lab/environments.py"],
              "input_sha256": hashes, "splits": {}}
    for split in splits:
        expected = expected_states(split, count)
        forecasts = {checkpoint: validated_forecasts(folder / f"{checkpoint}-{split}-forecasts.jsonl", expected)
                     for checkpoint in CHECKPOINTS}
        if receipt["selected_update"] == 0:
            require(forecasts["best"] == forecasts["baseline"], "Selected update zero forecasts differ from baseline")
        elif receipt["selected_update"] == receipt["updates"]:
            require(forecasts["best"] == forecasts["latest"], "Selected final-update forecasts differ from latest")
        decisions = {checkpoint: [{"id": identity, "mask": mask,
                                  **score_decision(expected[(identity, mask)]["scenario"], values[(identity, mask)],
                                                   expected[(identity, mask)]["posterior"])}
                                 for identity, mask in sorted(expected)] for checkpoint, values in forecasts.items()}
        comparisons = {}
        for checkpoint in ("best", "latest"):
            pairs = list(zip(decisions["baseline"], decisions[checkpoint]))
            comparisons[f"baseline_to_{checkpoint}"] = {
                "mean_terminal_regret_delta": cluster_bootstrap(
                    [(a["id"], b["terminal_regret"] - a["terminal_regret"]) for a, b in pairs],
                    seed=seed, resamples=resamples),
                "action_changed_count": sum(a["action"] != b["action"] for a, b in pairs),
                "oracle_mismatch_count_delta": sum(b["oracle_action_mismatch"] - a["oracle_action_mismatch"] for a, b in pairs)}
        output["splits"][split] = {"worlds": count, "states": count * 4,
                                   "checkpoints": {name: summarize_decisions(rows) for name, rows in decisions.items()},
                                   "paired_world_comparisons": comparisons}
    require(all(file_hash(folder / name) == checksum for name, checksum in hashes.items()),
            "Forecast diagnostic inputs changed while being read")
    require(file_hash(ENVIRONMENT_PATH) == output["environment_sha256"],
            "Environment source changed during reconstruction")
    return output


def markdown(report):
    lines = ["# Post-hoc forecast-to-action diagnostic", "",
             "This is a fixed **stop now** controller, not the learned sequential policy. It uses saved event forecasts to choose affirm, deny, or abstain under the declared payoffs. It never buys another check; past check costs are sunk. This analysis does not select or retrain a model.", "",
             "Each sampled world contributes four equally weighted evidence masks. Rewards integrate the latent event exactly conditional on that evidence; they do not integrate every possible sensor report or follow policy visitation frequencies. The terminal-only oracle knows the exact posterior, not the realized event.", "",
             "| Run / split / checkpoint | Mean terminal reward | Mean terminal regret | Oracle-action mismatches / states | Affirm / deny / abstain |",
             "|---|---:|---:|---:|---:|"]
    for run in report["runs"]:
        name = run["name"].replace("|", "/")
        for split, figures in run["splits"].items():
            for checkpoint, values in figures["checkpoints"].items():
                label = "best (supervised update zero)" if checkpoint == "best" and not run["best_contains_rl_updates"] else checkpoint
                counts = " / ".join(str(values["action_counts"][action]) for action in ACTIONS)
                lines.append(f"| {name} / {split} / {label} | {values['mean_expected_terminal_reward']:.6f} | {values['mean_terminal_regret']:.6f} | {values['oracle_action_mismatch_count']} / {values['states']} | {counts} |")
    lines += ["", "| Run / split / comparison | Mean regret change [95% world-bootstrap interval] | Actions changed |",
              "|---|---:|---:|"]
    for run in report["runs"]:
        for split, figures in run["splits"].items():
            for name, comparison in figures["paired_world_comparisons"].items():
                delta = comparison["mean_terminal_regret_delta"]
                lines.append(f"| {run['name'].replace('|', '/')} / {split} / {name} | {delta['difference']:+.6f} [{delta['lower_95']:+.6f}, {delta['upper_95']:+.6f}] | {comparison['action_changed_count']} |")
    lines += ["", "Changes are checkpoint minus baseline; negative regret change is better. Whole-world resampling keeps all four correlated states together. Intervals cover sampled-world variation only, not training-seed uncertainty or multiple-comparison corrections.", "",
              "Ties deterministically prefer affirm, then deny, then abstain. Oracle-action mismatch counts use that same rule; strictly-suboptimal counts in report.json require regret greater than 1e-12. All named splits, input hashes, and reconstruction checks are recorded; no incomplete subset is silently accepted.", ""]
    return "\n".join(lines)


def build_report(folders, output, *, seed=41, resamples=1000):
    folders = [Path(folder).resolve() for folder in folders]
    require(folders and len(set(folders)) == len(folders), "Provide distinct completed run folders")
    output = Path(output)
    require(not output.exists(), "Refusing to overwrite an existing diagnostic output")
    runs = [analyze_run(folder, seed=seed, resamples=resamples) for folder in folders]
    report = {"schema": "first-instinct-posthoc-forecast-decisions-v1", "post_hoc": True,
              "model_selection_criterion": False, "native_actor_compared": False,
              "controller": "Stop immediately; maximize terminal_values(scenario, saved_event_forecast) over affirm, deny, abstain.",
              "evaluation_seed": EVALUATION_SEED, "world_seed_formula": "evaluation_seed * 100003 + zero_based_world_index",
              "reward_scope": "Exact expected terminal reward conditional on each sampled evidence state; no future inspection, no past inspection charges, equal weight for four masks per world. Not sequential-policy reward.",
              "tie_order": list(ACTIONS), "bootstrap": {"seed": seed, "resamples": resamples,
                  "scope": "Paired world-sampling variation, with all four masks kept together; not training-seed variation or multiplicity-adjusted inference."},
              "analysis_code_sha256": {"general_lab/forecast_decisions.py": file_hash(Path(__file__)),
                                       "general_lab/report.py": file_hash(Path(__file__).with_name("report.py")),
                                       "general_lab/environments.py": file_hash(ENVIRONMENT_PATH)}, "runs": runs}
    rendered = markdown(report)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(rendered)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41, help="Bootstrap seed only; reconstruction seed is fixed")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    report = build_report(args.runs, args.output, seed=args.seed, resamples=args.resamples)
    print(f"Wrote post-hoc fixed-stop diagnostic for {len(report['runs'])} completed runs to {args.output}")


if __name__ == "__main__":
    main()
