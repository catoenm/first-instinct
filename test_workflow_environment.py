"""Semantic and leakage checks for executable workshop decision workflows."""

from dataclasses import asdict, replace
from fractions import Fraction
import json
import random
import unittest
from unittest.mock import patch

from general_lab import workflow_environment as env


def scenario(**changes):
    return replace(env.Scenario(False, "part", True, 5, (1, 1, 1), 50, 3, 2), **changes)


class WorkflowEnvironmentTests(unittest.TestCase):
    def test_ready_part_can_be_assembled_and_submitted_without_inspection(self):
        with env.Episode(scenario(), env.Tape("ready", False)) as episode:
            first = episode.step("assemble")
            self.assertFalse(first["terminal"])
            second = episode.step("submit")
            self.assertTrue(second["terminal"])
            receipt = episode.truth_receipt()
            self.assertEqual(receipt["outcome"], "completed")
            self.assertEqual(receipt["spent_cents"], 9)
            self.assertEqual(first["reward_cents"] + second["reward_cents"], 91)
            self.assertEqual(receipt["return_cents"], 91)

    def test_failed_assembly_consumes_the_irreplaceable_casing(self):
        with env.Episode(scenario(), env.Tape("needs_preparation", True)) as episode:
            result = episode.step("assemble")
            self.assertEqual(result["observation"].history[-1].response, "assembly_rejected_casing_destroyed")
            self.assertTrue(result["terminal"])
            self.assertEqual(result["reward_cents"], -106)
            self.assertEqual(episode.truth_receipt()["outcome"], "damaged")
            self.assertFalse(episode.truth_receipt()["casing_available"])
            self.assertEqual(episode.legal_actions(), [])
            with self.assertRaises(ValueError):
                episode.step("acquire")

    def test_reusable_casing_allows_recovery_after_prerequisite_failure(self):
        with env.Episode(scenario(failure_consumes_casing=False), env.Tape("needs_preparation", False)) as episode:
            failed = episode.step("assemble")
            self.assertFalse(failed["terminal"])
            self.assertIn("assemble", episode.legal_actions())
            episode.step("prepare")
            episode.step("assemble")
            episode.step("submit")
            self.assertEqual(episode.truth_receipt()["outcome"], "completed")
            self.assertEqual(episode.truth_receipt()["spent_cents"], 18)

    def test_preparation_never_repairs_a_broken_part(self):
        with env.Episode(scenario(), env.Tape("broken", True)) as episode:
            episode.step("prepare")
            self.assertIsNone(episode.observe().known_condition)
            episode.step("assemble")
            self.assertEqual(episode.truth_receipt()["outcome"], "damaged")

    def test_inspection_is_a_real_dependency_and_a_failed_attempt_costs_time(self):
        with env.Episode(scenario(inspection_required=True), env.Tape("needs_preparation", True)) as episode:
            result = episode.step("prepare")
            self.assertEqual(result["observation"].history[-1].response, "preparation_rejected_inspection_required")
            self.assertFalse(episode.observe().prepared)
            self.assertEqual((episode.observe().tick, episode.observe().spent_cents), (1, 3))
            self.assertIn("prepare", episode.legal_actions())
            for action in ("inspect", "prepare", "assemble", "submit"):
                episode.step(action)
            self.assertEqual(episode.truth_receipt()["outcome"], "completed")

    def test_replacing_part_invalidates_part_preparation_but_preserves_station_preparation(self):
        states = {}
        for scope in ("station", "part"):
            with self.subTest(scope=scope), env.Episode(scenario(preparation_scope=scope), env.Tape("broken", True)) as episode:
                episode.step("prepare")
                episode.step("acquire")
                states[scope] = episode.observe()
                self.assertEqual(episode.observe().known_condition, "needs_preparation")
                self.assertFalse(episode.observe().inspected)
                result = episode.step("assemble")
                if scope == "station":
                    self.assertFalse(result["terminal"])
                    episode.step("submit")
                    self.assertEqual(episode.truth_receipt()["outcome"], "completed")
                else:
                    self.assertEqual(episode.truth_receipt()["outcome"], "damaged")
        self.assertTrue(states["station"].prepared)
        self.assertFalse(states["part"].prepared)

    def test_acquisition_failure_is_visible_and_does_not_change_the_current_part(self):
        with env.Episode(scenario(), env.Tape("ready", False)) as episode:
            episode.step("inspect")
            episode.step("acquire")
            obs = episode.observe()
            self.assertEqual(obs.known_condition, "ready")
            self.assertTrue(obs.inspected)
            self.assertEqual(obs.history[-1].response, "spare_unavailable")
            self.assertNotIn("acquire", episode.legal_actions())
            episode.step("assemble")
            episode.step("submit")
            self.assertEqual(episode.truth_receipt()["outcome"], "completed")

    def test_deadline_requires_submission_not_just_a_finished_assembly(self):
        with env.Episode(scenario(deadline=4), env.Tape("ready", False)) as episode:
            for action in ("inspect", "acquire", "prepare", "assemble"):
                episode.step(action)
            truth = episode.truth_receipt()
            self.assertTrue(truth["assembled"])
            self.assertFalse(truth["submitted"])
            self.assertEqual(truth["outcome"], "unfinished")

    def test_abandonment_and_premature_submission_are_unfinished(self):
        for action in ("abstain", "submit"):
            with self.subTest(action=action), env.Episode(scenario(), env.Tape("ready", True)) as episode:
                episode.step(action)
                self.assertEqual(episode.truth_receipt()["outcome"], "unfinished")

    def test_hidden_facts_do_not_change_initial_inputs_or_legal_menus(self):
        config = scenario()
        inputs, forecasts, menus = [], [], []
        for tape, _ in env.tape_support(config):
            with env.Episode(config, tape) as episode:
                inputs.append(episode.input())
                forecasts.append(env.forecast_inputs(config, episode.observe(), "acquire"))
                menus.append(episode.legal_actions())
                self.assertEqual(episode.options(), episode.input()["options"])
                with self.assertRaises(ValueError):
                    episode.truth_receipt()
                with self.assertRaises(TypeError):
                    env.public_input(config, episode)
        self.assertTrue(all(x == inputs[0] for x in inputs))
        self.assertTrue(all(x == forecasts[0] for x in forecasts))
        self.assertTrue(all(x == list(env.ACTIONS) for x in menus))
        for forbidden in ('"tape"', '"initial_condition"', '"spare_available"', '"verifier_probability"', '"group_id"'):
            self.assertNotIn(forbidden, inputs[0]["state"])

    def test_public_forecast_inputs_do_not_enumerate_or_sample_hidden_tapes(self):
        config = scenario()
        with env.Episode(config, env.Tape("ready", False)) as episode:
            with patch.object(env, "tape_support", side_effect=AssertionError("hidden enumeration")), \
                    patch.object(env, "draw_tape", side_effect=AssertionError("hidden sampling")):
                inputs = env.forecast_inputs(config, episode.observe(), "prepare")
            self.assertEqual(inputs["cost_values"], {x["id"]: int(x["id"]) for x in inputs["cost_input"]["options"]})

    def test_failed_acquisition_does_not_reveal_uninspected_part_condition(self):
        observations = []
        for condition in env.CONDITIONS:
            with env.Episode(scenario(), env.Tape(condition, False)) as episode:
                episode.step("acquire")
                observations.append(episode.observe())
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(observations[1], observations[2])

    def test_posterior_conditions_only_on_visible_execution_history(self):
        config = scenario(initial_weights=(2, 3, 5), spare_percent=25)
        with env.Episode(config, env.Tape("broken", False)) as episode:
            episode.step("inspect")
            support = env.conditional_support(config, episode.observe())
            self.assertEqual(dict(support), {env.Tape("broken", False): Fraction(3, 4), env.Tape("broken", True): Fraction(1, 4)})
            episode.step("acquire")
            self.assertEqual(env.conditional_support(config, episode.observe()), [(env.Tape("broken", False), Fraction(1))])
            impossible = replace(episode.observe(), known_condition="ready")
            with self.assertRaises(ValueError):
                env.conditional_support(config, impossible)

    def test_counterfactual_outcomes_and_costs_match_hand_computed_distribution(self):
        config = scenario()
        with env.Episode(config, env.Tape("ready", True)) as episode:
            exact = env.exact_forecast(config, episode.observe(), "assemble")
            self.assertEqual(exact["outcome_probabilities"], {"completed": Fraction(1, 3), "damaged": Fraction(2, 3), "unfinished": 0})
            nonzero_costs = {k: v for k, v in exact["cost_probabilities"].items() if v}
            self.assertEqual(nonzero_costs, {"6": Fraction(2, 3), "9": Fraction(1, 3)})
            self.assertEqual(exact["expected_future_return_cents"], Fraction(-121, 3))
            # Several actions may each succeed; per-action forecasts are not a
            # distribution over actions. Each action's own outcomes sum to one.
            for row in env.exact_counterfactuals(config, episode.observe()).values():
                self.assertEqual(sum(row["outcome_probabilities"].values()), 1)
                self.assertEqual(sum(row["cost_probabilities"].values()), 1)

    def test_future_costs_exclude_sunk_costs_and_share_the_outcome_execution(self):
        config = scenario(initial_weights=(1, 0, 0), spare_percent=0)
        with env.Episode(config, env.Tape("ready", False)) as episode:
            episode.step("inspect")
            bundle = env.forecast_bundle(config, episode.observe(), "assemble", 71)
            self.assertEqual(bundle["outcome_target"], "completed")
            self.assertEqual(bundle["cost_target"], "9")
            self.assertEqual(bundle["replay"]["truth_receipt"]["spent_cents"], 12)
            self.assertEqual(bundle["replay"]["future_cost_cents"], 9)
            self.assertIn("exclude the already spent 3 cents", bundle["cost_input"]["question"])
            self.assertIn(env.CONTINUATION, bundle["outcome_input"]["question"])
            self.assertIn(env.CONTINUATION, bundle["cost_input"]["question"])

    def test_fresh_counterfactual_draws_are_reproducible_and_can_disagree(self):
        config = scenario()
        with env.Episode(config, env.Tape("ready", False)) as episode:
            before = episode.observe()
            bundles = [env.forecast_bundle(config, before, "assemble", seed) for seed in range(40)]
            self.assertEqual(bundles[3], env.forecast_bundle(config, before, "assemble", 3))
            self.assertEqual({b["outcome_target"] for b in bundles}, {"completed", "damaged"})
            self.assertTrue(all(b["outcome_input"] == bundles[0]["outcome_input"] for b in bundles))
            for b in bundles:
                receipt = b["replay"]["truth_receipt"]
                self.assertEqual(b["outcome_target"], receipt["outcome"])
                self.assertEqual(b["cost_target"], str(receipt["spent_cents"]))
            self.assertEqual(episode.observe(), before)

    def test_cost_menu_is_public_bounded_and_contains_every_executed_cost(self):
        for split in ("train", "validation", "test"):
            for index in range(12):
                config = env.make_scenario(33, index, split)
                with env.Episode(config, env.sample_tape(config, random.Random(index))) as episode:
                    for _ in range(index % 3):
                        if episode.observe().terminal:
                            break
                        episode.step(random.Random(index).choice(episode.legal_actions()))
                    if episode.observe().terminal:
                        continue
                    for action in episode.legal_actions():
                        inputs = env.forecast_inputs(config, episode.observe(), action)
                        self.assertLessEqual(len(inputs["cost_input"]["options"]), 26)
                        for tape, _ in env.conditional_support(config, episode.observe()):
                            _, cost, _ = env.execute_counterfactual(config, episode.observe(), action, tape)
                            self.assertIn(str(cost), inputs["cost_values"])

    def test_known_terminal_and_last_tick_costs_have_singleton_menus(self):
        config = scenario()
        with env.Episode(config, env.Tape("needs_preparation", False)) as episode:
            self.assertEqual(env.future_cost_options(config, episode.observe(), "abstain"), (0,))
            self.assertEqual(env.future_cost_options(config, episode.observe(), "submit"), (3,))
            for action in ("inspect", "acquire", "prepare", "assemble"):
                episode.step(action)
            self.assertEqual(episode.observe().tick, 4)
            for action in episode.legal_actions():
                self.assertEqual(env.future_cost_options(config, episode.observe(), action), (env.action_costs(config)[action],))

    def test_generated_mechanism_coverage_is_balanced_before_model_use(self):
        for split, width in (("train", 4), ("validation", 2), ("test", 2)):
            mechanisms = [env.make_scenario(13, index, split).mechanism for index in range(2 * width)]
            self.assertEqual(len(set(mechanisms)), width)
            self.assertTrue(all(mechanisms.count(m) == 2 for m in set(mechanisms)))

    def test_split_ownership_groups_all_parameter_and_cost_variants(self):
        groups = {"train": set(), "validation": set(), "test": set()}
        for split in groups:
            for i in range(100):
                config = env.make_scenario(51, i, split)
                self.assertEqual(config, env.make_scenario(51, i, split))
                changed = replace(config, unit_cost_cents=20, acquire_units=5, deadline=4, initial_weights=(1, 0, 0), spare_percent=0)
                self.assertEqual(config.group_id, changed.group_id)
                self.assertEqual(config.split, split)
                self.assertEqual(env.split_owner(config.group_id), split)
                groups[split].add(config.group_id)
        self.assertEqual([len(groups[s]) for s in groups], [4, 2, 2])
        self.assertTrue(groups["train"].isdisjoint(groups["validation"] | groups["test"]))
        self.assertTrue(groups["validation"].isdisjoint(groups["test"]))
        train = [m for m, owner in env.MECHANISM_SPLITS.items() if owner == "train"]
        for heldout, owner in env.MECHANISM_SPLITS.items():
            if owner != "train":
                for field, value in enumerate(heldout):
                    self.assertTrue(any(m[field] == value for m in train))

        # Parity split gives training all four combinations of each pair of
        # factors, and no heldout split structurally excludes casing damage.
        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertEqual(len({(m[first], m[second]) for m in train}), 4)
        for split in groups:
            mechanisms = [m for m, owner in env.MECHANISM_SPLITS.items() if owner == split]
            self.assertEqual({m[2] for m in mechanisms}, {False, True})
            outcomes = set()
            for m in mechanisms:
                config = scenario(inspection_required=m[0], preparation_scope=m[1], failure_consumes_casing=m[2])
                for tape, _ in env.tape_support(config):
                    for action in env.ACTIONS:
                        with env.Episode(config, tape) as episode:
                            outcomes.add(env.execute_counterfactual(config, episode.observe(), action, tape)[0])
            self.assertEqual(outcomes, set(env.OUTCOMES))

    def test_serializable_observation_replay_and_reset(self):
        config = scenario()
        tape = env.Tape("needs_preparation", True)
        with env.Episode(config, tape) as episode:
            initial = episode.observe()
            for action in ("inspect", "prepare", "assemble", "submit"):
                episode.step(action)
            obs = env.observation_from_dict(json.loads(json.dumps(asdict(episode.observe()))))
            with env.replay_to(config, tape, obs) as replay:
                self.assertEqual(replay.truth_receipt(), episode.truth_receipt())
            self.assertEqual(episode.reset(), initial)
            episode.reset(env.Tape("ready", False))
            episode.step("inspect")
            self.assertEqual(episode.observe().known_condition, "ready")

    def test_fixed_continuation_is_visible_only_and_always_legal(self):
        for mechanism in env.MECHANISM_SPLITS:
            config = scenario(inspection_required=mechanism[0], preparation_scope=mechanism[1], failure_consumes_casing=mechanism[2])
            for tape, _ in env.tape_support(config):
                with env.Episode(config, tape) as episode:
                    rewards = []
                    while not episode.observe().terminal:
                        action = env.continuation_action(config, episode.observe())
                        self.assertIn(action, episode.legal_actions())
                        rewards.append(episode.step(action)["reward_cents"])
                    self.assertLessEqual(episode.observe().tick, 5)
                    self.assertEqual(sum(rewards), episode.truth_receipt()["return_cents"])


if __name__ == "__main__":
    unittest.main()
