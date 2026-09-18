"""Semantic and provenance tests for the separately versioned retry adapter."""

from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from general_lab import retry_v2 as retry


def scenario(**changes):
    return replace(retry.Scenario(3, False, 0, (1, 1, 0), (0, 1, 0), 50, 5, 10), **changes)


def start(config):
    with retry.EpisodeAdapter(config, retry.Tape(None, None, False)) as episode:
        return episode.observe()


class RetryV2Tests(unittest.TestCase):
    def test_engine_pin_and_public_input_boundary(self):
        retry.assert_engine()
        self.assertEqual(retry.source_receipt()["engine_sha256"], retry.ENGINE_SHA256)
        config = scenario()
        inputs = []
        for tape in (retry.Tape(None, None, False), retry.Tape(0, 1, True), retry.Tape(2, 2, False)):
            with retry.EpisodeAdapter(config, tape) as episode:
                inputs.append(retry.forecast_inputs(config, episode.observe(), "lookup"))
                with self.assertRaises(ValueError):
                    episode.truth_receipt()
                self.assertEqual(episode.legal_actions(), retry.ACTIONS)
                self.assertEqual(set(episode.input()), {"state", "question", "options"})
        self.assertEqual(inputs, [inputs[0]] * 3)
        for forbidden in ("initial_at", "retry_delay", "verifier", "completion_rows", "draw_seed"):
            self.assertNotIn(forbidden, json.dumps(inputs))
        with patch.object(retry, "conditional_support", side_effect=AssertionError("Hidden enumeration forbidden")):
            self.assertEqual(retry.forecast_inputs(config, start(config), "lookup"), inputs[0])
        with self.assertRaises(TypeError):
            retry.forecast_input(config, object(), "retry")

    def test_costs_scaled_probabilities_and_hidden_draws_share_group(self):
        config = scenario()
        variants = [replace(config, lookup_cost=999), replace(config, retry_cost=222),
                    replace(config, wait_cost=100, abstain_cost=333),
                    replace(config, initial_weights=(2, 2, 0), retry_weights=(0, 8, 0))]
        for variant in variants:
            self.assertEqual(retry.mechanism_id(config), retry.mechanism_id(variant))
            self.assertEqual(retry.split_owner(config), retry.split_owner(variant))
        self.assertNotEqual(retry.mechanism_id(config), retry.mechanism_id(replace(config, ack_percent=75)))

    def test_whole_mechanism_combinations_are_held_out(self):
        mechanisms, combinations = {}, {}
        for split in retry.SPLITS:
            configs = [retry.make_scenario(seed, index, split) for seed in (1, 2, 3) for index in range(30)]
            mechanisms[split] = {retry.mechanism_id(config) for config in configs}
            combinations[split] = {retry.mechanism_combination(config) for config in configs}
            self.assertTrue(all(retry.split_owner(config) == split for config in configs))
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(mechanisms[a] & mechanisms[b])
            self.assertFalse(combinations[a] & combinations[b])
        self.assertEqual(combinations["train"], set(retry.TRAIN_COMBINATIONS))
        self.assertEqual(combinations["validation"], set(retry.HELD_OUT["validation"]))
        self.assertEqual(combinations["test"], set(retry.HELD_OUT["test"]))
        for split in retry.SPLITS:
            self.assertEqual({row[1] for row in combinations[split]}, {False, True})
        # Every pair is seen during training; the test concerns their conjunction.
        all_combinations = set.union(*combinations.values())
        for indices in ((0, 1), (0, 2), (1, 2)):
            project = lambda values: {tuple(row[index] for index in indices) for row in values}
            self.assertEqual(project(combinations["train"]), project(all_combinations))

    def test_configuration_and_transport_streams_are_reproducible(self):
        for split in retry.SPLITS:
            config = retry.make_scenario(70, 9, split)
            self.assertEqual(config, retry.make_scenario(70, 9, split))
            self.assertEqual(retry.sample_tape(config, random.Random(9)), retry.sample_tape(config, random.Random(9)))
        for split in ("other", None):
            with self.assertRaises(ValueError):
                retry.make_scenario(70, 9, split)
        for index in (-1, 1.5):
            with self.assertRaises(ValueError):
                retry.make_scenario(70, index)

    def test_lookup_continuation_retries_empty_and_stops_on_positive(self):
        config = scenario(ack_percent=100)
        observation = start(config)
        missing = retry.execute_forecast(config, observation, "lookup", retry.Tape(None, 1, True))
        existing = retry.execute_forecast(config, observation, "lookup", retry.Tape(0, 1, True))
        self.assertEqual([step["action"] for step in missing["steps"]], ["lookup", "retry", "abstain"])
        self.assertEqual([step["action"] for step in existing["steps"]], ["lookup", "abstain"])
        self.assertEqual(missing["future_cost_cents"], 18)
        self.assertEqual(existing["future_cost_cents"], 8)
        self.assertEqual(missing["outcome"], existing["outcome"])
        self.assertEqual(missing["outcome"], "one")
        forecast = retry.exact_forecast(config, observation, "lookup")
        self.assertEqual(forecast["outcome_probabilities"], {"zero": 0, "one": 1, "two": 0})
        self.assertEqual(forecast["cost_probabilities"]["8"], Fraction(1, 2))
        self.assertEqual(forecast["cost_probabilities"]["18"], Fraction(1, 2))
        self.assertEqual(sum(forecast["cost_probabilities"].values()), 1)
        self.assertEqual(retry.expected_utility(**dict(zip(
            ("outcome_probabilities", "cost_probabilities"), forecast.values()))), 87)
        passive = retry.exact_forecast(config, observation, "lookup", "stop_now")
        self.assertEqual(retry.expected_utility(**passive), -5)
        self.assertGreater(retry.expected_utility(**forecast), retry.expected_utility(**passive))

    def test_stale_empty_snapshot_can_trigger_duplicate_side_effect(self):
        config = scenario(receipt_lag=1, initial_weights=(0, 0, 1), ack_percent=100)
        observation = start(config)
        result = retry.execute_forecast(config, observation, "lookup", retry.Tape(2, 1, True))
        self.assertEqual(result["outcome"], "two")
        self.assertEqual(result["steps"][0]["after"]["history"][-1]["completed_jobs"], 0)
        keyed = replace(config, keyed=True)
        result_keyed = retry.execute_forecast(keyed, start(keyed), "lookup", retry.Tape(2, 1, True))
        self.assertEqual(result_keyed["outcome"], "one")

    def test_acknowledgement_stops_continuation_without_revealing_count(self):
        config = scenario(ack_percent=100)
        observations = []
        for initial in (None, 0):
            with retry.EpisodeAdapter(config, retry.Tape(initial, 1, True)) as episode:
                observation = episode.step("retry")["observation"]
                observations.append(observation)
                self.assertEqual(retry.continuation_action(config, observation), "abstain")
                dist = retry.exact_distribution(config, observation, "abstain")
                self.assertEqual(dist, {"zero": 0, "one": Fraction(1, 2), "two": Fraction(1, 2)})
        self.assertEqual(observations[0], observations[1])

    def test_stop_now_is_distinct_and_has_explicit_zero_followup_cost(self):
        config = scenario(initial_weights=(1, 0, 0))
        observation = start(config)
        stopped = retry.execute_forecast(config, observation, "lookup", retry.Tape(None, 1, False), "stop_now")
        continued = retry.execute_forecast(config, observation, "lookup", retry.Tape(None, 1, False))
        self.assertEqual(stopped["outcome"], "zero")
        self.assertEqual(continued["outcome"], "one")
        self.assertEqual(stopped["future_cost_cents"], 5)
        self.assertEqual(retry.future_cost_options(config, observation, "lookup", "stop_now"), (5,))
        self.assertIn("no further requests or paid actions", retry.forecast_input(config, observation, "lookup", "stop_now")["state"])

    def test_zero_and_two_outcomes_have_different_expected_utility(self):
        # Both have zero probability of exactly-one success, but are not equally bad.
        cost = {"5": 1}
        self.assertEqual(retry.expected_utility({"zero": 1, "one": 0, "two": 0}, cost), -105)
        self.assertEqual(retry.expected_utility({"zero": 0, "one": 0, "two": 1}, cost), -205)
        # Equal success probabilities: distinguishing duplicate harm reverses
        # the preference a success-only controller gets by minimizing cost.
        costly_but_no_duplicates = retry.expected_utility({"zero": .5, "one": .5, "two": 0}, {"30": 1})
        cheap_duplicate_risk = retry.expected_utility({"zero": 0, "one": .5, "two": .5}, {"0": 1})
        self.assertEqual((costly_but_no_duplicates, cheap_duplicate_risk), (-30, -50))
        self.assertGreater(costly_but_no_duplicates, cheap_duplicate_risk)
        for probabilities in ({"zero": 0, "one": 2, "two": 0}, {"zero": 0, "one": float("nan"), "two": 0}):
            with self.assertRaises(ValueError):
                retry.expected_utility(probabilities, cost)
        with self.assertRaises(ValueError):
            retry.expected_utility({"zero": 1, "one": 0, "two": 0}, {"1.5": 1})

    def test_full_distributions_are_separate_per_action(self):
        config = scenario(initial_weights=(0, 1, 0), ack_percent=100)
        observation = start(config)
        forecasts = {action: retry.exact_distribution(config, observation, action) for action in retry.ACTIONS}
        self.assertTrue(all(sum(distribution.values()) == 1 for distribution in forecasts.values()))
        self.assertGreater(sum(distribution["one"] for distribution in forecasts.values()), 1)
        self.assertEqual(forecasts["retry"]["two"], 1)
        self.assertEqual(forecasts["abstain"]["one"], 1)

    def test_future_costs_exclude_sunk_costs_and_match_all_executed_sequences(self):
        for split in retry.SPLITS:
            for index in range(8):
                config = retry.make_scenario(11, index, split)
                for tape, _ in retry.tape_support(config):
                    with retry.EpisodeAdapter(config, tape) as episode:
                        episode.step("wait")
                        observation = episode.observe()
                        for action in episode.legal_actions():
                            for continuation in retry.CONTINUATIONS:
                                result = retry.execute_forecast(config, observation, action, tape, continuation)
                                self.assertIn(result["future_cost_cents"], retry.future_cost_options(config, observation, action, continuation))
                                self.assertEqual(result["future_cost_cents"], sum(step["cost_cents"] for step in result["steps"]))
                                self.assertEqual(result["truth_receipt"]["spent_cents"], observation.spent_cents + result["future_cost_cents"])

    def test_incremental_episode_rewards_and_terminal_handling(self):
        config = scenario()
        episode = retry.EpisodeAdapter(config, retry.Tape(None, 1, False))
        results = [episode.step(action) for action in ("lookup", "retry", "wait")]
        self.assertTrue(results[-1]["terminal"])
        self.assertEqual(sum(result["reward_cents"] for result in results), episode.truth_receipt()["return_cents"])
        self.assertEqual(episode.legal_actions(), ())
        self.assertIsNone(retry.continuation_action(config, episode.observe()))
        with self.assertRaises(ValueError):
            episode.step("wait")
        with self.assertRaises(ValueError):
            retry.forecast_inputs(config, episode.observe(), "retry")
        episode.close()
        episode.close()

    def test_empirical_labels_share_rollout_but_vary_with_independent_seeds(self):
        config = scenario(ack_percent=100)
        observation = start(config)
        pairs = [retry.forecast_bundle(config, observation, "retry", seed) for seed in range(35)]
        self.assertEqual({pair["outcome_target"] for pair in pairs}, {"one", "two"})
        for pair in pairs:
            receipt = pair["replay"]
            self.assertEqual(receipt["outcome_input_sha256"], retry.digest(pair["outcome_input"]))
            self.assertEqual(receipt["cost_input_sha256"], retry.digest(pair["cost_input"]))
            self.assertEqual(pair["outcome_target"], receipt["result"]["outcome"])
            self.assertEqual(pair["cost_target"], str(receipt["result"]["future_cost_cents"]))
            tape = retry.Tape(**receipt["result"]["truth_receipt"]["tape"])
            self.assertEqual(retry.execute_forecast(config, observation, "retry", tape), receipt["result"])
            public_pair = json.dumps({key: value for key, value in pair.items() if key != "replay"})
            for forbidden in ("verifier_probability", "outcome_probabilities", "cost_probabilities", "future_return_cents", "truth_receipt"):
                self.assertNotIn(forbidden, public_pair)
        self.assertEqual(pairs[4], retry.forecast_bundle(config, observation, "retry", 4))

    def test_forged_public_history_or_illegal_continuation_fails(self):
        config = scenario()
        observation = start(config)
        with self.assertRaises(ValueError):
            retry.forecast_inputs(config, observation, "invented")
        with self.assertRaises(ValueError):
            retry.forecast_inputs(config, observation, "retry", "learned_future_policy")
        forged = replace(observation, spent_cents=999)
        with self.assertRaises(ValueError):
            retry.forecast_bundle(config, forged, "retry", 1)

    def test_generated_rows_and_receipts_replay_and_no_splits_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            summary = retry.generate(first, worlds_per_split=5, seed=17)
            self.assertEqual(summary, retry.generate(second, worlds_per_split=5, seed=17))
            all_groups = {}
            for name, expected_hash in summary["files"].items():
                self.assertEqual(hashlib.sha256((first / name).read_bytes()).hexdigest(), expected_hash)
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            for split in retry.SPLITS:
                read = lambda name: [json.loads(line) for line in (first / f"{split}-{name}.jsonl").read_text().splitlines()]
                examples = {row["id"]: row for row in read("examples")}
                for row in read("replay"):
                    example = examples[row["id"]]
                    config = retry.Scenario(**row["scenario"])
                    observation = retry.observation_from_dict(row["observation"])
                    expected = retry.forecast_bundle(config, observation, row["offered_action"], row["draw_seed"])
                    self.assertEqual(expected["outcome_target"], example["outcome_target"])
                    self.assertEqual(expected["cost_target"], example["cost_target"])
                    self.assertEqual(retry.canonical(expected["replay"]), retry.canonical({key: value for key, value in row.items() if key not in ("id", "root_id")}))
                    self.assertEqual(example["split"], split)
                    all_groups.setdefault(split, set()).add(example["group_id"])
                    self.assertNotIn("replay", example)
                for row in read("trajectories"):
                    config = retry.Scenario(**row["scenario"])
                    tape = retry.Tape(**row["truth_receipt"]["tape"])
                    with retry.EpisodeAdapter(config, tape) as episode:
                        for step in row["steps"]:
                            self.assertEqual(episode.input(), step["input"])
                            self.assertEqual(step["action_probability"], 1 / len(episode.legal_actions()))
                            episode.step(step["action"])
                        self.assertEqual(episode.truth_receipt(), row["truth_receipt"])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse(all_groups[a] & all_groups[b])
            with self.assertRaises(ValueError):
                retry.generate(first, worlds_per_split=1)


if __name__ == "__main__":
    unittest.main()
