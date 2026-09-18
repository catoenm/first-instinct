"""Synthetic contracts for the offline paired evaluation reporter."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from general_lab.report import (build_report, cluster_bootstrap, compare_predictions,
                                compare_rl_artifacts, compare_rl_runs, prediction_map,
                                read_rl_run, row_metrics)


def prediction(identity, p, target="a", group=None, task="task101_example"):
    probabilities = {"a": p, "b": 1 - p}
    return {"id": identity, "group_id": group or identity, "task": task,
            "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities,
            "target_ids": [target]}


def rl_rows(confidence=.5, reward=0.):
    forecasts, policies = [], []
    for i, outcome in enumerate((0, 1)):
        identity = f"world-{i}"
        for mask in range(4):
            forecasts.append({"id": identity, "mask": mask, "domain": "toy", "outcome": outcome,
                              "posterior": .75 if outcome else .25,
                              "forecast": confidence if outcome else 1 - confidence})
        policies.append({"id": identity, "truth": bool(outcome), "reports": [True, False], "domain": "toy",
                         "policy_expected_reward": reward, "oracle_planner_expected_reward": 1.,
                         "no_inspection_oracle_expected_reward": .4})
    return forecasts, policies


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_rl_run(folder, seed=47, confidence=.8, weight=.5):
    folder.mkdir()
    receipt = {"status": "complete", "config": {"seed": seed, "forecast_weight": weight}, "model": {"id": "toy"},
               "updates": 2, "optimizer_steps": 2, "selected_update": 1,
               "checkpoint_selection": {"metric": "validation.policy.policy_expected_reward", "criterion": "reward_only"},
               "language_parameter_audit": {"changed_elements": 10}, "pure_policy_language_gradient": {"nonzero_elements": 4},
               "starting_adapter_sha256": {"adapter.safetensors": "toy-initialization"},
               "code_sha256": {"general_lab/environments.py": "toy-environment"}}
    (folder / "run.json").write_text(json.dumps(receipt))
    for checkpoint in ("baseline", "best", "latest"):
        c = .5 if checkpoint == "baseline" else confidence
        reward = 0. if checkpoint == "baseline" else .25
        forecasts, policies = rl_rows(c, reward)
        metrics = {"test": {"root_episodes": 2, "policy": {"policy_expected_reward": reward},
                   "forecast": {"brier": (1 - c) ** 2, "log_loss": -math.log(c),
                                "mean_squared_error_to_exact_posterior": (c - .75) ** 2}}}
        (folder / f"{checkpoint}-metrics.json").write_text(json.dumps(metrics))
        write_rows(folder / f"{checkpoint}-test-forecasts.jsonl", forecasts)
        write_rows(folder / f"{checkpoint}-test-policy.jsonl", policies)


class GeneralComparisonTests(unittest.TestCase):
    def test_known_row_and_equal_task_macro_changes(self):
        base = [prediction(f"easy-{i}", .1) for i in range(3)] + [prediction("hard", .8, task="verified_logic_choice")]
        trained = [prediction(f"easy-{i}", .8) for i in range(3)] + [prediction("hard", .1, task="verified_logic_choice")]
        result = compare_predictions(base, trained, resamples=1000)
        overall = result["overall"]
        self.assertEqual(overall["row_weighted"]["base"]["accuracy"], .25)
        self.assertEqual(overall["row_weighted"]["trained"]["accuracy"], .75)
        self.assertEqual(overall["row_weighted"]["paired_group_bootstrap"]["accuracy"]["difference"], .5)
        self.assertEqual(overall["macro_task"]["base"]["accuracy"], .5)
        self.assertEqual(overall["macro_task"]["trained"]["accuracy"], .5)
        self.assertEqual(overall["macro_task"]["difference"]["accuracy"], 0.)
        self.assertEqual(set(result["subgroups"]), {"public", "verified"})

    def test_multiple_answers_are_not_a_brier_distribution(self):
        a = prediction("multi", .4)
        a.update(probabilities={"a": .4, "b": .3, "c": .3}, target_ids=["a", "b"], choice="a")
        b = copy.deepcopy(a); b.update(probabilities={"a": .3, "b": .6, "c": .1}, choice="b")
        result = compare_predictions([a], [b])["overall"]
        self.assertEqual(result["multi_answer_rows"], 1)
        self.assertEqual(result["row_weighted"]["base"]["accuracy"], 1.)
        self.assertAlmostEqual(result["row_weighted"]["base"]["acceptable_set_log_loss"], -math.log(.7))
        self.assertAlmostEqual(result["row_weighted"]["trained"]["acceptable_set_log_loss"], -math.log(.9))
        self.assertIsNone(result["row_weighted"]["base"]["single_label_multiclass_brier"])
        self.assertEqual(result["row_weighted"]["paired_group_bootstrap"]["single_label_multiclass_brier"]["groups"], 0)

    def test_multiclass_brier_definition_and_log_floor(self):
        score = row_metrics(prediction("one", .8))
        self.assertAlmostEqual(score["single_label_multiclass_brier"], .08)
        floor = row_metrics(prediction("zero", 0.))
        self.assertAlmostEqual(floor["acceptable_set_log_loss"], -math.log(1e-12))

    def test_strict_pair_contracts(self):
        a = prediction("one", .6)
        changes = [{"id": "two"}, {"group_id": "other"}, {"task": "other"}, {"target_ids": ["b"]},
                   {"probabilities": {"a": .6, "c": .4}}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_predictions([a], [{**a, **change}])
        with self.assertRaises(ValueError):
            compare_predictions([a, a], [a])

    def test_invalid_probability_receipts_rejected(self):
        for values in ({"a": math.nan, "b": .5}, {"a": .7, "b": .7}, {"a": -.1, "b": 1.1}, {"a": True, "b": 0.}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                prediction_map([{**prediction("one", .6), "probabilities": values}])
        with self.assertRaises(ValueError):
            prediction_map([{**prediction("one", .6), "choice": "b"}])

    def test_cluster_bootstrap_does_not_count_views_as_independent(self):
        single = cluster_bootstrap([("a", 1.), ("b", -1.)], seed=12)
        repeated = cluster_bootstrap([("a", 1.)] * 100 + [("b", -1.)] * 100, seed=12)
        self.assertEqual(repeated["rows"], 200)
        self.assertEqual(repeated["groups"], 2)
        self.assertEqual(single["lower_95"], repeated["lower_95"])
        self.assertEqual(single["upper_95"], repeated["upper_95"])
        self.assertEqual(repeated["lower_95"], -1.)
        self.assertEqual(repeated["upper_95"], 1.)

    def test_shared_worlds_and_input_order_invariance(self):
        a = [prediction("one", .6, group="world"), prediction("two", .6, group="world", task="environment_forecast")]
        b = [prediction("one", .7, group="world"), prediction("two", .7, group="world", task="environment_forecast")]
        first = compare_predictions(a, b)
        self.assertEqual(first, compare_predictions(list(reversed(a)), list(reversed(b))))
        self.assertEqual(first["shared_world_audit"]["groups_spanning_multiple_tasks"], 1)
        self.assertEqual(first["shared_world_audit"]["maximum_rows_per_group"], 2)

    def test_report_writes_only_observed_results_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "base.jsonl", [prediction("one", .6)])
            write_rows(root / "trained.jsonl", [prediction("one", .8)])
            report = build_report(root / "base.jsonl", root / "trained.jsonl", root / "report")
            self.assertNotIn("reinforcement_learning", report)
            self.assertTrue((root / "report" / "report.json").exists())
            self.assertIn("does not declare an overall winner", (root / "report" / "report.md").read_text())
            self.assertEqual(len(report["inputs"]["base_sha256"]), 64)

    def test_conflicting_data_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("base", "trained"):
                (root / name).mkdir()
                write_rows(root / name / "predictions.jsonl", [prediction("one", .6)])
                (root / name / "metrics.json").write_text(json.dumps({"split": "test", "data_sha256": name}))
            with self.assertRaises(ValueError):
                build_report(root / "base" / "predictions.jsonl", root / "trained" / "predictions.jsonl", root / "report")


class RLReportTests(unittest.TestCase):
    def test_known_paired_event_and_reward_changes(self):
        a, pa = rl_rows(.5, 0.)
        b, pb = rl_rows(.8, .25)
        report = compare_rl_artifacts(a, b, pa, pb)
        self.assertEqual(report["worlds"], 2)
        self.assertEqual(report["forecast_rows"], 8)
        delta = report["paired_world_bootstrap"]
        self.assertAlmostEqual(delta["event_brier"]["difference"], -.21)
        self.assertAlmostEqual(delta["posterior_mean_squared_error"]["difference"], -.06)
        self.assertAlmostEqual(delta["policy_expected_reward"]["difference"], .25)
        self.assertEqual(delta["event_brier"]["groups"], 2)

    def test_changed_worlds_or_missing_masks_are_rejected(self):
        a, pa = rl_rows(.5); b, pb = rl_rows(.8)
        for field, changed in (("outcome", 1), ("posterior", .99), ("id", "different"), ("domain", "elsewhere")):
            wrong = copy.deepcopy(b); wrong[0][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_rl_artifacts(a, wrong, pa, pb)
        with self.assertRaises(ValueError):
            compare_rl_artifacts(a, b[:-1], pa, pb)
        wrong_policy = copy.deepcopy(pb); wrong_policy[0]["oracle_planner_expected_reward"] = .1
        with self.assertRaises(ValueError):
            compare_rl_artifacts(a, b, pa, wrong_policy)

    def test_verified_run_metrics_and_language_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "hybrid"
            write_rl_run(folder)
            report, _ = read_rl_run(folder)
            self.assertTrue(report["checkpoints"]["best"]["test"]["verified_from_raw_artifacts"])
            self.assertTrue(report["language_update_evidence"]["latest_adapter_changed"])
            self.assertTrue(report["language_update_evidence"]["pure_policy_language_gradient_nonzero"])
            self.assertFalse(report["missing_artifacts"])
            self.assertIn("baseline_to_best:test", report["paired_baseline_comparisons"])
            path = folder / "best-metrics.json"
            wrong = json.loads(path.read_text()); wrong["test"]["forecast"]["brier"] = .001
            path.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):
                read_rl_run(folder)

    def test_missing_artifacts_and_proof_stay_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "run.json").write_text(json.dumps({"status": "bounded_stop", "config": {}}))
            report, artifacts = read_rl_run(folder)
            self.assertEqual(report["checkpoints"], {})
            self.assertEqual(artifacts, {})
            self.assertIsNone(report["language_update_evidence"]["latest_adapter_changed"])
            self.assertEqual(set(report["missing_artifacts"]), {"baseline-metrics.json", "best-metrics.json", "latest-metrics.json"})

    def test_only_matching_training_seeds_and_initializations_are_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rl_run(root / "reward", seed=47, confidence=.7, weight=0.)
            write_rl_run(root / "hybrid", seed=47, confidence=.8, weight=.5)
            write_rl_run(root / "another_seed", seed=53, confidence=.9, weight=.5)
            report = compare_rl_runs([root / "reward", root / "hybrid", root / "another_seed"])
            self.assertEqual(len(report["paired_selected_runs"]), 1)
            pair = report["paired_selected_runs"][0]
            self.assertEqual(pair["left"], "reward")
            self.assertEqual(pair["right"], "hybrid")
            self.assertEqual(pair["training_seed"], 47)
            self.assertAlmostEqual(pair["paired_world_bootstrap"]["event_brier"]["difference"], -.05)


if __name__ == "__main__":
    unittest.main()
