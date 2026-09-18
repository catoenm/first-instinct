"""Offline paired evaluation reports with world-cluster uncertainty estimates.

Reads existing predictions and receipts only. No inference, training, network
access, or automatic declaration of a winning model is performed.
"""

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from scale_lab.common import file_hash, write_json

METRICS = ("accuracy", "acceptable_set_log_loss", "single_label_multiclass_brier")
RL_METRICS = ("policy_expected_reward", "event_brier", "event_log_loss", "posterior_mean_squared_error")


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def read_lines(path):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
                except ValueError as error:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from error
    if not rows:
        raise ValueError(f"Empty predictions: {path}")
    return rows


def finite(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{label} is outside its valid range")
    return float(value)


def identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def prediction_map(rows):
    result = {}
    for row in rows:
        identity = identifier(row.get("id"), "Prediction id")
        if identity in result:
            raise ValueError(f"Duplicate prediction id: {identity}")
        for key in ("group_id", "task"):
            identifier(row.get(key), key)
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict) or len(probabilities) < 2:
            raise ValueError(f"Expected at least two option probabilities: {identity}")
        for option, value in probabilities.items():
            identifier(option, "Option identifier"); finite(value, "Option probability", 0, 1)
        if not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-5):
            raise ValueError(f"Option probabilities do not sum to one: {identity}")
        targets = row.get("target_ids")
        if not isinstance(targets, list) or not targets or any(not isinstance(x, str) for x in targets) or len(set(targets)) != len(targets):
            raise ValueError(f"Targets must be a nonempty set of option ids: {identity}")
        if not set(targets) <= probabilities.keys() or row.get("choice") not in probabilities:
            raise ValueError(f"Target or choice absent from options: {identity}")
        if probabilities[row["choice"]] < max(probabilities.values()) - 1e-6:
            raise ValueError(f"Recorded choice is not a most-probable option: {identity}")
        result[identity] = row
    if not result:
        raise ValueError("No predictions")
    return result


def pair_predictions(base, trained):
    left, right = prediction_map(base), prediction_map(trained)
    if left.keys() != right.keys():
        raise ValueError("Prediction id sets differ; incomplete evaluations cannot be paired")
    pairs = []
    for identity in sorted(left):
        a, b = left[identity], right[identity]
        if any(a[k] != b[k] for k in ("group_id", "task")) or set(a["target_ids"]) != set(b["target_ids"]) or a["probabilities"].keys() != b["probabilities"].keys():
            raise ValueError(f"Prediction group/task/target/option contract mismatch: {identity}")
        pairs.append((a, b))
    return pairs


def row_metrics(row):
    p, targets = row["probabilities"], row["target_ids"]
    return {"accuracy": float(row["choice"] in targets),
            "acceptable_set_log_loss": -math.log(max(1e-12, min(1., sum(p[x] for x in targets)))),
            "single_label_multiclass_brier": sum((value - float(option == targets[0])) ** 2 for option, value in p.items()) if len(targets) == 1 else None}


def average_metrics(rows):
    values = [row_metrics(row) for row in rows]
    return {key: sum(v[key] for v in values if v[key] is not None) / sum(v[key] is not None for v in values)
            if any(v[key] is not None for v in values) else None for key in METRICS}


def cluster_bootstrap(values, *, seed=41, resamples=1000):
    """Paired scalar differences, resampling entire groups with replacement.

    ``values`` contains (group_id, paired_difference). Sampling eligible groups
    directly ensures a Brier interval never fabricates values for excluded rows.
    """
    if resamples <= 0:
        raise ValueError("Resamples must be positive")
    grouped = defaultdict(list)
    for group, value in values:
        grouped[group].append(finite(value, "Paired difference"))
    if not grouped:
        return {"difference": None, "lower_95": None, "upper_95": None, "rows": 0, "groups": 0, "resamples": resamples}
    groups = sorted(grouped)
    sums = np.array([sum(grouped[g]) for g in groups], dtype=np.float64)
    counts = np.array([len(grouped[g]) for g in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = []
    # Bound temporary arrays for larger evaluations.
    chunk = max(1, min(128, 2_000_000 // len(groups)))
    for start in range(0, resamples, chunk):
        selected = rng.integers(len(groups), size=(min(chunk, resamples - start), len(groups)))
        estimates.extend((sums[selected].sum(1) / counts[selected].sum(1)).tolist())
    low, high = np.quantile(estimates, [.025, .975])
    return {"difference": float(sums.sum() / counts.sum()), "lower_95": float(low), "upper_95": float(high),
            "rows": int(counts.sum()), "groups": len(groups), "resamples": resamples,
            "single_group_interval_is_uninformative": len(groups) == 1}


def task_family(task):
    if re.match(r"task\d+(?:_|$)", task):
        return "public"
    if task.startswith("verified_"):
        return "verified"
    if task.startswith("environment_"):
        return "environment"
    return "other"


def summarize_pairs(pairs, seed=41, resamples=1000):
    tasks = defaultdict(list)
    groups = defaultdict(list)
    for pair in pairs:
        tasks[pair[0]["task"]].append(pair)
        groups[pair[0]["group_id"]].append(pair)
    by_task = {}
    for task, members in sorted(tasks.items()):
        left = average_metrics([a for a, _ in members]); right = average_metrics([b for _, b in members])
        by_task[task] = {"rows": len(members), "groups": len({a["group_id"] for a, _ in members}),
                         "single_label_rows": sum(len(a["target_ids"]) == 1 for a, _ in members),
                         "base": left, "trained": right,
                         "difference": {k: right[k] - left[k] if left[k] is not None else None for k in METRICS}}
    macro = {}
    for side in ("base", "trained", "difference"):
        macro[side] = {key: sum(t[side][key] for t in by_task.values() if t[side][key] is not None)
                      / sum(t[side][key] is not None for t in by_task.values())
                      if any(t[side][key] is not None for t in by_task.values()) else None for key in METRICS}
    left = average_metrics([a for a, _ in pairs]); right = average_metrics([b for _, b in pairs])
    differences = {key: [] for key in METRICS}
    for a, b in pairs:
        ma, mb = row_metrics(a), row_metrics(b)
        for key in METRICS:
            if ma[key] is not None:
                differences[key].append((a["group_id"], mb[key] - ma[key]))
    group_sizes = defaultdict(int)
    complete = {"base": 0, "trained": 0}
    for members in groups.values():
        group_sizes[len(members)] += 1
        complete["base"] += all(a["choice"] in a["target_ids"] for a, _ in members)
        complete["trained"] += all(b["choice"] in b["target_ids"] for _, b in members)
    group_complete = {key: value / len(groups) for key, value in complete.items()}
    group_complete.update(difference=group_complete["trained"] - group_complete["base"], groups=len(groups),
                          group_size_distribution={str(size): count for size, count in sorted(group_sizes.items())},
                          note="A group is correct only when every evaluated row in that group is correct within this report scope. Groups have equal weight; two-row groups measure both members correct.")
    return {"rows": len(pairs), "groups": len({a["group_id"] for a, _ in pairs}), "tasks": len(tasks),
            "single_label_rows": sum(len(a["target_ids"]) == 1 for a, _ in pairs),
            "multi_answer_rows": sum(len(a["target_ids"]) > 1 for a, _ in pairs),
            "group_complete_accuracy": group_complete,
            "row_weighted": {"base": left, "trained": right,
                             "paired_group_bootstrap": {key: cluster_bootstrap(values, seed=seed, resamples=resamples) for key, values in differences.items()}},
            "macro_task": {**macro, "brier_eligible_tasks": sum(t["single_label_rows"] > 0 for t in by_task.values()),
                           "note": "Each task has equal weight. These are descriptive means; bootstrap intervals above estimate row-weighted changes."},
            "by_task": by_task}


def compare_predictions(base, trained, seed=41, resamples=1000):
    pairs = pair_predictions(base, trained)
    families = defaultdict(list)
    for pair in pairs:
        families[task_family(pair[0]["task"])].append(pair)
    group_tasks = defaultdict(set); group_rows = defaultdict(int)
    for a, _ in pairs:
        group_tasks[a["group_id"]].add(a["task"])
        group_rows[a["group_id"]] += 1
    return {"overall": summarize_pairs(pairs, seed, resamples),
            "subgroups": {name: summarize_pairs(rows, seed, resamples) for name, rows in sorted(families.items())},
            "shared_world_audit": {"groups": len(group_tasks), "groups_spanning_multiple_tasks": sum(len(x) > 1 for x in group_tasks.values()),
                                   "maximum_tasks_per_group": max(map(len, group_tasks.values())),
                                   "maximum_rows_per_group": max(group_rows.values())},
            "contract": {"paired_on": ["id", "group_id", "task", "target_ids_set", "option_ids_set"],
                         "brier": "Multiclass sum of squared probabilities against a one-hot target, range [0,2]. Only rows with exactly one accepted answer are eligible; this is not binary event Brier.",
                         "log_loss_floor": 1e-12,
                         "probability_note": "Probabilities are conditional on the supplied option set. A multiple-acceptable-answer set does not define a probability distribution over mutually exclusive outcomes."}}


def _forecast_map(rows):
    result = {}; masks = defaultdict(set); truths = {}
    for row in rows:
        identity = identifier(row.get("id"), "Forecast world id")
        mask = row.get("mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or mask not in range(4):
            raise ValueError("Forecast masks must be integers in 0..3")
        key = (identity, mask)
        if key in result:
            raise ValueError("Duplicate forecast world/mask")
        outcome = row.get("outcome")
        if outcome not in (0, 1):
            raise ValueError("Forecast outcome must be zero or one")
        finite(row.get("forecast"), "Event forecast", 0, 1)
        finite(row.get("posterior"), "Exact posterior", 0, 1)
        if identity in truths and truths[identity] != outcome:
            raise ValueError("A world's outcome changed between evidence masks")
        truths[identity] = outcome; masks[identity].add(mask); result[key] = row
    if not result or any(value != {0, 1, 2, 3} for value in masks.values()):
        raise ValueError("Each forecast world must contain all four common evidence masks")
    return result


def _policy_map(rows):
    result = {}
    for row in rows:
        identity = identifier(row.get("id"), "Policy world id")
        if identity in result:
            raise ValueError("Duplicate policy world")
        for key in ("policy_expected_reward", "oracle_planner_expected_reward", "no_inspection_oracle_expected_reward"):
            finite(row.get(key), key)
        if row.get("truth") not in (False, True, 0, 1) or not isinstance(row.get("reports"), (list, tuple)) or len(row["reports"]) != 2 or any(x not in (False, True, 0, 1) for x in row["reports"]):
            raise ValueError("Policy trace requires its common latent event and two reports")
        result[identity] = row
    if not result:
        raise ValueError("Empty policy audit")
    return result


def compare_rl_artifacts(base_forecasts, trained_forecasts, base_policy, trained_policy, seed=41, resamples=1000):
    fa, fb = _forecast_map(base_forecasts), _forecast_map(trained_forecasts)
    pa, pb = _policy_map(base_policy), _policy_map(trained_policy)
    if fa.keys() != fb.keys() or pa.keys() != pb.keys() or set(pa) != {key[0] for key in fa}:
        raise ValueError("Reinforcement-learning evaluations do not share common world/mask identities")
    values = {key: [] for key in RL_METRICS}
    for key in sorted(fa):
        a, b = fa[key], fb[key]
        if any(a.get(k) != b.get(k) for k in ("outcome", "posterior", "domain")):
            raise ValueError("Common forecast worlds differ in outcome/posterior/domain")
        for row in (a, b):
            if int(pa[row["id"]]["truth"]) != row["outcome"]:
                raise ValueError("Forecast outcome and policy event disagree")
            if pa[row["id"]].get("domain") != row.get("domain"):
                raise ValueError("Forecast and policy domains disagree")
        def losses(row):
            p, y, q = row["forecast"], row["outcome"], row["posterior"]
            return {"event_brier": (p - y) ** 2,
                    "event_log_loss": -math.log(max(1e-12, p if y else 1 - p)),
                    "posterior_mean_squared_error": (p - q) ** 2}
        left, right = losses(a), losses(b)
        for metric in left:
            values[metric].append((key[0], right[metric] - left[metric]))
    for identity in sorted(pa):
        a, b = pa[identity], pb[identity]
        invariants = ("truth", "reports", "domain", "oracle_planner_expected_reward", "no_inspection_oracle_expected_reward")
        if any(a.get(k) != b.get(k) for k in invariants):
            raise ValueError("Common policy worlds differ in hidden state or oracle/payoff reference")
        values["policy_expected_reward"].append((identity, b["policy_expected_reward"] - a["policy_expected_reward"]))
    return {"worlds": len(pa), "forecast_rows": len(fa), "evidence_masks_per_world": 4,
            "identity_verified": True,
            "paired_world_bootstrap": {key: cluster_bootstrap(rows, seed=seed, resamples=resamples) for key, rows in values.items()},
            "note": "All four forecast masks are resampled together with their world. Intervals do not treat them as independent outcomes."}


def _recompute_rl(forecasts, policy):
    f = list(_forecast_map(forecasts).values()); p = list(_policy_map(policy).values())
    return {"policy_expected_reward": sum(row["policy_expected_reward"] for row in p) / len(p),
            "event_brier": sum((row["forecast"] - row["outcome"]) ** 2 for row in f) / len(f),
            "event_log_loss": -sum(math.log(max(1e-12, row["forecast"] if row["outcome"] else 1 - row["forecast"])) for row in f) / len(f),
            "posterior_mean_squared_error": sum((row["forecast"] - row["posterior"]) ** 2 for row in f) / len(f)}


def _reported_rl(measured):
    paths = {"policy_expected_reward": ("policy", "policy_expected_reward"),
             "event_brier": ("forecast", "brier"), "event_log_loss": ("forecast", "log_loss"),
             "posterior_mean_squared_error": ("forecast", "mean_squared_error_to_exact_posterior")}
    return {key: finite(measured[section][field], key) for key, (section, field) in paths.items()}


def read_rl_run(folder, seed=41, resamples=1000):
    folder = Path(folder).resolve(); run = read_json(folder / "run.json")
    hashes = {"run.json": file_hash(folder / "run.json")}; checkpoints = {}; artifacts = {}; missing = []
    for checkpoint in ("baseline", "best", "latest"):
        metrics_path = folder / f"{checkpoint}-metrics.json"
        if not metrics_path.exists():
            missing.append(metrics_path.name); continue
        hashes[metrics_path.name] = file_hash(metrics_path)
        checkpoints[checkpoint] = {}
        for split, measured in sorted(read_json(metrics_path).items()):
            reported = _reported_rl(measured)
            fpath = folder / f"{checkpoint}-{split}-forecasts.jsonl"
            ppath = folder / f"{checkpoint}-{split}-policy.jsonl"
            entry = {"metrics": reported, "reported_root_episodes": measured.get("root_episodes"), "verified_from_raw_artifacts": False}
            if fpath.exists() and ppath.exists():
                forecasts, policy = read_lines(fpath), read_lines(ppath)
                recomputed = _recompute_rl(forecasts, policy)
                # Self-pairing validates full common-world integrity too.
                integrity = compare_rl_artifacts(forecasts, forecasts, policy, policy, seed, 1)
                if measured.get("root_episodes") != integrity["worlds"]:
                    raise ValueError(f"Reported world count disagrees with raw data: {folder.name}/{checkpoint}/{split}")
                for key in reported:
                    if not math.isclose(reported[key], recomputed[key], rel_tol=1e-7, abs_tol=1e-8):
                        raise ValueError(f"Metric disagrees with raw artifacts: {folder.name}/{checkpoint}/{split}/{key}")
                entry.update(verified_from_raw_artifacts=True, worlds=integrity["worlds"], forecast_rows=integrity["forecast_rows"])
                artifacts[(checkpoint, split)] = (forecasts, policy)
                hashes[fpath.name] = file_hash(fpath); hashes[ppath.name] = file_hash(ppath)
            else:
                missing.extend(p.name for p in (fpath, ppath) if not p.exists())
            checkpoints[checkpoint][split] = entry
    comparisons = {}
    for checkpoint in ("best", "latest"):
        for split in checkpoints.get(checkpoint, {}):
            if ("baseline", split) in artifacts and (checkpoint, split) in artifacts:
                a, b = artifacts[("baseline", split)], artifacts[(checkpoint, split)]
                comparisons[f"baseline_to_{checkpoint}:{split}"] = compare_rl_artifacts(a[0], b[0], a[1], b[1], seed, resamples)
    delta, gradient = run.get("language_parameter_audit"), run.get("pure_policy_language_gradient")
    changed = finite(delta.get("changed_elements"), "Changed language elements", 0) > 0 if delta else None
    nonzero = finite(gradient.get("nonzero_elements"), "Policy-gradient elements", 0) > 0 if gradient else None
    output = {"path": str(folder), "name": folder.name, "status": run.get("status"), "config": run.get("config", {}),
              "model": run.get("model"), "updates": run.get("updates"), "optimizer_steps": run.get("optimizer_steps"),
              "selected_update": run.get("selected_update"), "checkpoint_selection": run.get("checkpoint_selection"),
              "language_update_evidence": {"latest_adapter_changed": changed, "pure_policy_language_gradient_nonzero": nonzero,
                                           "parameter_audit": delta, "policy_gradient_audit": gradient,
                                           "scope": "These are training-receipt assertions, not independent checkpoint tensor verification. The parameter delta refers to latest; selected step zero remains the supervised baseline."},
              "checkpoints": checkpoints, "paired_baseline_comparisons": comparisons,
              "missing_artifacts": sorted(set(missing)), "input_sha256": hashes,
              "starting_adapter_sha256": run.get("starting_adapter_sha256"),
              "environment_code_sha256": run.get("code_sha256", {}).get("general_lab/environments.py")}
    return output, artifacts


def compare_rl_runs(folders, seed=41, resamples=1000):
    loaded = [read_rl_run(folder, seed, resamples) for folder in folders]
    paired = []
    for (a, aa), (b, bb) in itertools.combinations(loaded, 2):
        if a["config"].get("seed") is None or a["config"].get("seed") != b["config"].get("seed"):
            continue
        if not a["starting_adapter_sha256"] or a["starting_adapter_sha256"] != b["starting_adapter_sha256"] or a["model"] != b["model"]:
            continue
        if not a["environment_code_sha256"] or a["environment_code_sha256"] != b["environment_code_sha256"]:
            continue
        splits = sorted({split for checkpoint, split in aa if checkpoint == "best"} & {split for checkpoint, split in bb if checkpoint == "best"})
        for split in splits:
            left, right = aa[("best", split)], bb[("best", split)]
            paired.append({"left": a["name"], "right": b["name"], "difference_direction": "right minus left",
                           "training_seed": a["config"]["seed"], "checkpoint": "best", "split": split,
                           **compare_rl_artifacts(left[0], right[0], left[1], right[1], seed, resamples)})
    return {"runs": [run for run, _ in loaded], "paired_selected_runs": paired,
            "pairing_rule": "Only runs with the same training seed, starting-adapter checksums, model specification and environment code hash are paired. World identities and oracle references must also agree. Different training seeds are not pooled."}


def _number(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _interval(value):
    if value["difference"] is None:
        return "—"
    return f"{value['difference']:+.4f} [{value['lower_95']:+.4f}, {value['upper_95']:+.4f}]"


def markdown(report):
    general = report["general"]; overall = general["overall"]; row = overall["row_weighted"]
    lines = ["# Paired decision-model evaluation", "", "All changes below are trained minus base. Positive accuracy and negative losses indicate their respective directions; the report does not declare an overall winner.", "",
             f"{overall['rows']:,} paired rows, {overall['groups']:,} world/document groups, {overall['tasks']} tasks. {overall['multi_answer_rows']:,} rows have multiple acceptable answers.", "",
             "| Row-weighted metric | Base | Trained | Change [95% group-bootstrap interval] |", "|---|---:|---:|---:|"]
    for key in METRICS:
        lines.append(f"| {key} | {_number(row['base'][key])} | {_number(row['trained'][key])} | {_interval(row['paired_group_bootstrap'][key])} |")
    lines += ["", "| Equal-task macro metric | Base | Trained | Change |", "|---|---:|---:|---:|"]
    for key in METRICS:
        macro = overall["macro_task"]
        lines.append(f"| {key} | {_number(macro['base'][key])} | {_number(macro['trained'][key])} | {_number(macro['difference'][key], True)} |")
    complete = overall["group_complete_accuracy"]
    lines += ["", "| Complete-group accuracy: every row correct | Base | Trained | Change | Groups |", "|---|---:|---:|---:|---:|",
              f"| All evaluated members correct | {_number(complete['base'])} | {_number(complete['trained'])} | {_number(complete['difference'], True)} | {complete['groups']} |",
              "", "For two-row groups, this measures both members correct. The full group-size distribution is recorded in report.json."]
    lines += ["", "| Subgroup | Rows / groups / tasks | Accuracy, base → trained | Set log loss, base → trained |", "|---|---:|---:|---:|"]
    for name, group in general["subgroups"].items():
        a, b = group["row_weighted"]["base"], group["row_weighted"]["trained"]
        lines.append(f"| {name} | {group['rows']} / {group['groups']} / {group['tasks']} | {a['accuracy']:.4f} → {b['accuracy']:.4f} | {a['acceptable_set_log_loss']:.4f} → {b['acceptable_set_log_loss']:.4f} |")
    lines += ["", "Single-label multiclass Brier is the sum over options and ranges from 0 to 2. Multiple-acceptable-answer rows are excluded. These option distributions are not automatically calibrated success forecasts.", "",
              f"Intervals use {report['bootstrap']['resamples']:,} paired resamples of complete groups (seed {report['bootstrap']['seed']}). Rows sharing a world stay together. Macro means are descriptive. Intervals cover world-sampling variation, not training-seed variation, task selection, or pretraining contamination."]
    if report.get("reinforcement_learning", {}).get("runs"):
        lines += ["", "## Reinforcement-learning runs", "", "Best checkpoints were selected by the criterion recorded in each run; latest results and all paired intervals are in report.json. Parameter-change evidence refers to latest weights.", "",
                  "| Run | Selected update | Latest changed / policy gradient nonzero | Status |", "|---|---:|---|---|"]
        for run in report["reinforcement_learning"]["runs"]:
            proof = run["language_update_evidence"]
            lines.append(f"| {run['name'].replace('|', '/')} | {run['selected_update']} | {proof['latest_adapter_changed']} / {proof['pure_policy_language_gradient_nonzero']} | {run['status']} |")
        lines += ["", "| Run / split | Expected reward, baseline → best | Event Brier, baseline → best | Posterior mean squared error, baseline → best | Raw identities verified |", "|---|---:|---:|---:|---|"]
        for run in report["reinforcement_learning"]["runs"]:
            for split, best in run["checkpoints"].get("best", {}).items():
                baseline = run["checkpoints"].get("baseline", {}).get(split)
                if baseline is None:
                    continue
                a, b = baseline["metrics"], best["metrics"]
                arrows = [f"{a[key]:.4f} → {b[key]:.4f}" for key in ("policy_expected_reward", "event_brier", "posterior_mean_squared_error")]
                verified = f"baseline_to_best:{split}" in run["paired_baseline_comparisons"]
                lines.append(f"| {run['name'].replace('|', '/')} / {split} | {' | '.join(arrows)} | {verified} |")
        lines += ["", "Binary event Brier ranges from 0 to 1. The four forecast evidence masks share one event and are clustered by world. Missing receipts or evaluations remain missing; receipt claims are not independent tensor verification."]
    return "\n".join(lines) + "\n"


def build_report(base, trained, output, rl_runs=(), seed=41, resamples=1000):
    base, trained, output = Path(base), Path(trained), Path(output)
    receipts = {}
    for name, path in (("base", base), ("trained", trained)):
        metadata = path.parent / "metrics.json"
        if metadata.exists():
            recorded = read_json(metadata)
            if isinstance(recorded, dict) and "data_sha256" in recorded and "split" in recorded:
                receipts[name] = {"sha256": file_hash(metadata), "split": recorded["split"],
                                  "data_sha256": recorded["data_sha256"], "model": recorded.get("model")}
    if len(receipts) == 2 and any(receipts["base"][key] != receipts["trained"][key] for key in ("data_sha256", "split")):
        raise ValueError("Evaluation receipts identify different data or splits")
    report = {"schema": "first-instinct-paired-report-v1", "bootstrap": {"seed": seed, "resamples": resamples,
               "method": "paired percentile cluster bootstrap; complete group/world resampling"},
              "inputs": {"base": str(base.resolve()), "trained": str(trained.resolve()),
                         "base_sha256": file_hash(base), "trained_sha256": file_hash(trained),
                         "evaluation_receipts": receipts, "same_data_receipt_verified": len(receipts) == 2},
              "general": compare_predictions(read_lines(base), read_lines(trained), seed, resamples)}
    if rl_runs:
        report["reinforcement_learning"] = compare_rl_runs(rl_runs, seed, resamples)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rl-runs", nargs="*", type=Path, default=[])
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--resamples", type=int, default=1000)
    args = p.parse_args()
    if args.resamples <= 0:
        p.error("resamples must be positive")
    report = build_report(args.base, args.trained, args.output, args.rl_runs, args.seed, args.resamples)
    print(json.dumps({"output": str(args.output), "rows": report["general"]["overall"]["rows"],
                      "groups": report["general"]["overall"]["groups"], "tasks": report["general"]["overall"]["tasks"]}))


if __name__ == "__main__":
    main()
