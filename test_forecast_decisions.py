"""Offline fixed-stop diagnostic contracts; tiny regenerated worlds, no model calls."""

import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from general_lab.environments import make_episode
from general_lab.forecast_decisions import (CHECKPOINTS, ENVIRONMENT_PATH, EVALUATION_SEED,
                                           analyze_run, build_report, expected_states, main,
                                           score_decision, terminal_decision)
from scale_lab.common import MODELS, file_hash


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class TerminalDecisionTests(unittest.TestCase):
    def test_asymmetric_costs_change_the_threshold_and_exact_regret(self):
        # Affirm utility is 10p-9; deny utility is 1-2p. The threshold
        # is 5/6, not one half. Abstaining at -2 is never optimal here.
        scenario = replace(make_episode(8).scenario, correct_reward=1.,
                           false_positive_cost=9., false_negative_cost=1., abstain_cost=2.)
        self.assertEqual(terminal_decision(scenario, .8), "deny")
        self.assertEqual(terminal_decision(scenario, .9), "affirm")
        result = score_decision(scenario, .8, .9)
        self.assertAlmostEqual(result["expected_terminal_reward"], -.8)
        self.assertAlmostEqual(result["terminal_regret"], .8)
        self.assertTrue(result["oracle_action_mismatch"])
        # Forced-stop continuation does not charge any check costs.
        cheap = replace(scenario, sensors=tuple(replace(sensor, cost=0.) for sensor in scenario.sensors))
        self.assertEqual(result, score_decision(cheap, .8, .9))

    def test_abstention_and_exact_ties_are_deterministic(self):
        scenario = replace(make_episode(9).scenario, correct_reward=1.,
                           false_positive_cost=9., false_negative_cost=9., abstain_cost=.1)
        self.assertEqual(terminal_decision(scenario, .5), "abstain")
        tied = replace(scenario, false_positive_cost=1., false_negative_cost=1., abstain_cost=0.)
        self.assertEqual(terminal_decision(tied, .5), "affirm")
        deny_tie = replace(tied, correct_reward=0.)
        self.assertEqual(terminal_decision(deny_tie, 0.), "deny")
        for p in (0., .1, .5, .9, 1.):
            self.assertEqual(score_decision(scenario, p, p)["terminal_regret"], 0.)


class ForecastDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, *, selected=0, perfect=False, splits=("validation", "test")):
        folder = self.root / "run"
        folder.mkdir()
        receipt = {"status": "complete", "model": copy.deepcopy(MODELS["qwen35-9b"]),
                   "config": {"eval_episodes": 2, "eval_splits": list(splits), "seed": 47},
                   "updates": 2, "optimizer_steps": 4, "selected_update": selected,
                   "checkpoint_selection": {"evaluation_seed": EVALUATION_SEED},
                   "code_sha256": {"general_lab/environments.py": file_hash(ENVIRONMENT_PATH)}}
        write_json(folder / "run.json", receipt)
        for split in splits:
            expected = expected_states(split, 2)
            for checkpoint in CHECKPOINTS:
                rows = []
                for (identity, mask), state in expected.items():
                    q = state["posterior"]
                    probability = q if perfect or checkpoint == "latest" or (checkpoint == "best" and selected == 2) else .5
                    if checkpoint == "best" and selected == 1 and not perfect:
                        probability = .25 + .5 * q
                    rows.append({"id": identity, "mask": mask, "domain": state["domain"],
                                 "outcome": state["outcome"], "posterior": q, "forecast": probability})
                write_rows(folder / f"{checkpoint}-{split}-forecasts.jsonl", rows)
        return folder, receipt

    def test_perfect_forecasts_have_zero_regret_on_all_declared_splits(self):
        folder, _ = self.fixture(perfect=True, splits=("validation", "test", "shift", "new_domain"))
        result = analyze_run(folder, resamples=30)
        self.assertEqual(set(result["splits"]), {"validation", "test", "shift", "new_domain"})
        for split in result["splits"].values():
            self.assertEqual((split["worlds"], split["states"]), (2, 8))
            for figures in split["checkpoints"].values():
                self.assertEqual(figures["mean_terminal_regret"], 0.)
                self.assertEqual(figures["oracle_action_mismatch_count"], 0)
                self.assertEqual(figures["strictly_suboptimal_count"], 0)
                self.assertEqual(sum(figures["action_counts"].values()), 8)

    def test_selected_zero_is_baseline_and_bootstrap_clusters_whole_worlds(self):
        folder, _ = self.fixture()
        result = analyze_run(folder, resamples=30)
        self.assertFalse(result["best_contains_rl_updates"])
        self.assertIn("not an RL-trained selection", result["checkpoint_note"])
        for split in result["splits"].values():
            best = split["paired_world_comparisons"]["baseline_to_best"]
            delta = best["mean_terminal_regret_delta"]
            self.assertEqual((delta["rows"], delta["groups"]), (8, 2))
            self.assertEqual((delta["difference"], delta["lower_95"], delta["upper_95"]), (0., 0., 0.))
            self.assertEqual(best["action_changed_count"], 0)
            latest = split["paired_world_comparisons"]["baseline_to_latest"]["mean_terminal_regret_delta"]
            expected = split["checkpoints"]["latest"]["mean_terminal_regret"] - split["checkpoints"]["baseline"]["mean_terminal_regret"]
            self.assertAlmostEqual(latest["difference"], expected)
        path = folder / "best-test-forecasts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["forecast"] = .2
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "Selected update zero"):
            analyze_run(folder, resamples=30)

    def test_final_update_selection_must_match_latest(self):
        folder, _ = self.fixture(selected=2)
        self.assertTrue(analyze_run(folder, resamples=30)["best_contains_rl_updates"])
        path = folder / "best-test-forecasts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["forecast"] = 1 - rows[0]["forecast"]
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "Selected final-update"):
            analyze_run(folder, resamples=30)

    def test_world_identity_domain_outcome_mask_and_posterior_must_match(self):
        folder, _ = self.fixture(selected=1)
        path = folder / "latest-test-forecasts.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        changes = {"id": "unrelated-world", "domain": "different-domain", "outcome": 1 - original[0]["outcome"],
                   "mask": 4, "posterior": 1 - original[0]["posterior"], "forecast": 1.1}
        for field, value in changes.items():
            with self.subTest(field=field):
                rows = copy.deepcopy(original)
                rows[0][field] = value
                write_rows(path, rows)
                with self.assertRaises(ValueError):
                    analyze_run(folder, resamples=10)
        for label, rows in (("missing-mask", original[:-1]), ("duplicate", original + original[:1])):
            with self.subTest(label=label):
                write_rows(path, rows)
                with self.assertRaises(ValueError):
                    analyze_run(folder, resamples=10)

    def test_incomplete_or_mismatched_provenance_fails_without_output(self):
        folder, receipt = self.fixture()
        changes = [{"status": "training"}, {"status": "bounded_stop"}, {"updates": 0},
                   {"selected_update": 3}, {"optimizer_steps": 0},
                   {"checkpoint_selection": {"evaluation_seed": 7}},
                   {"code_sha256": {"general_lab/environments.py": "f" * 64}},
                   {"config": {**receipt["config"], "eval_episodes": 3}},
                   {"config": {**receipt["config"], "eval_splits": ["validation", "unknown"]}}]
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                write_json(folder / "run.json", {**receipt, **change})
                output = self.root / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    build_report([folder], output, resamples=10)
                self.assertFalse(output.exists())

    def test_missing_or_undeclared_trace_aborts_instead_of_skipping(self):
        folder, _ = self.fixture()
        path = folder / "latest-test-forecasts.jsonl"
        data = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "all declared splits"):
            build_report([folder], self.root / "missing", resamples=10)
        self.assertFalse((self.root / "missing").exists())
        path.write_text(data)
        (folder / "latest-undeclared-forecasts.jsonl").write_text(data)
        with self.assertRaisesRegex(ValueError, "all declared splits"):
            analyze_run(folder, resamples=10)

    def test_cli_writes_posthoc_report_with_hashes_and_no_reselection(self):
        folder, _ = self.fixture()
        output = self.root / "diagnostic"
        args = ["forecast_decisions", "--runs", str(folder), "--output", str(output), "--resamples", "30"]
        with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO):
            main()
        report = json.loads((output / "report.json").read_text())
        self.assertTrue(report["post_hoc"])
        self.assertFalse(report["model_selection_criterion"])
        self.assertFalse(report["native_actor_compared"])
        self.assertEqual(report["runs"][0]["input_sha256"]["run.json"], file_hash(folder / "run.json"))
        self.assertEqual(report["analysis_code_sha256"]["general_lab/environments.py"], file_hash(ENVIRONMENT_PATH))
        text = (output / "report.md").read_text()
        self.assertIn("not the learned sequential policy", text)
        self.assertIn("best (supervised update zero)", text)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            build_report([folder], output, resamples=10)


if __name__ == "__main__":
    unittest.main()
