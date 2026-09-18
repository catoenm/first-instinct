"""Executable retry-pilot invariants. Standard library only; no model loading."""

from dataclasses import asdict, replace
from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile
import unittest

from general_lab.retry_environment import (
    ACTIONS, Episode, Observation, Scenario, Tape, VisibleEvent, action_mask,
    conditional_support, digest, draw_tape, exact_forecasts, execute_counterfactual,
    generate, make_scenario, public_input, receipt_token, seed_for, split_owner, tape_support,
)


def scenario(**changes):
    return replace(Scenario(3, False, 0, (1, 1, 0), (0, 1, 0), 50, 5, 10), **changes)


def observation(value):
    return Observation(**{**value, "history": tuple(VisibleEvent(**event) for event in value["history"])})


class RetryEnvironmentTests(unittest.TestCase):
    def test_lost_acknowledgement_then_unkeyed_retry_processes_twice(self):
        with Episode(scenario(), Tape(0, 1, False)) as episode:
            first = episode.step("retry")
            self.assertEqual(first["observation"].history[-1].response, "acknowledgement_timeout")
            episode.step("abstain")
            truth = episode.truth_receipt()
            self.assertEqual(truth["completed_jobs"], 2)
            self.assertFalse(truth["success"])
            self.assertEqual([row[1] for row in truth["completion_rows"]], [None, None])
            self.assertEqual(truth["return_cents"], -200 - 10 - 3)

    def test_unique_idempotency_key_prevents_second_side_effect(self):
        config = scenario(keyed=True)
        with Episode(config, Tape(0, 1, True)) as episode:
            event = episode.step("retry")["observation"].history[-1]
            episode.step("abstain")
            truth = episode.truth_receipt()
            self.assertEqual(truth["completed_jobs"], 1)
            self.assertFalse(truth["deliveries"]["retry"]["created"])
            self.assertEqual(truth["deliveries"]["initial"]["receipt"], truth["deliveries"]["retry"]["receipt"])
            self.assertEqual(event.receipt_token, receipt_token(config, "initial"))
            self.assertEqual(receipt_token(config, "initial"), receipt_token(config, "retry"))
            self.assertTrue(event.receipt_token.startswith("receipt-"))
            self.assertEqual(len(event.receipt_token), 72)
            self.assertNotEqual(event.receipt_token, receipt_token(replace(config, lookup_cost=6), "initial"))

    def test_delayed_original_can_duplicate_an_earlier_retry(self):
        for keyed, expected in [(False, 2), (True, 1)]:
            with self.subTest(keyed=keyed), Episode(scenario(keyed=keyed), Tape(2, 1, True)) as episode:
                result = episode.step("retry")
                self.assertEqual(result["observation"].history[-1].response, "acknowledged")
                episode.step("abstain")
                self.assertEqual(episode.truth_receipt()["completed_jobs"], expected)

    def test_timestamped_stale_empty_receipt_does_not_cancel_pending_work(self):
        with Episode(scenario(receipt_lag=1), Tape(2, None, False)) as episode:
            event = episode.step("lookup")["observation"].history[-1]
            self.assertEqual((event.as_of, event.completed_jobs), (0, 0))
            episode.step("wait")
            visible = json.loads(public_input(episode.scenario, episode.observe())["state"])
            self.assertEqual(visible["history"][1]["completed_jobs"], 0)
            self.assertEqual(visible["history"][1]["as_of"], 0)
            episode.step("abstain")
            self.assertEqual(episode.truth_receipt()["completed_jobs"], 1)

    def test_lookup_exposes_only_commits_at_or_before_its_timestamp(self):
        for lag, count in [(0, 1), (1, 0)]:
            with self.subTest(lag=lag), Episode(scenario(receipt_lag=lag), Tape(None, 2, False)) as episode:
                episode.step("retry")
                event = episode.step("lookup")["observation"].history[-1]
                self.assertEqual((event.as_of, event.completed_jobs), (2 - lag, count))

    def test_repeated_lookup_retry_and_terminal_stepping_are_illegal(self):
        for action in ("lookup", "retry"):
            with self.subTest(action=action), Episode(scenario(), Tape(None, None, False)) as episode:
                episode.step(action)
                before = episode.observe()
                self.assertFalse(action_mask(episode.scenario, before)[action])
                with self.assertRaises(ValueError):
                    episode.step(action)
                self.assertEqual(episode.observe(), before)
        with Episode(scenario(), Tape(2, None, False)) as episode:
            episode.step("abstain")
            self.assertEqual(episode.truth_receipt()["completed_jobs"], 1)
            self.assertFalse(any(action_mask(episode.scenario, episode.observe()).values()))
            for action in (*ACTIONS, "invented"):
                with self.assertRaises(ValueError):
                    episode.step(action)

    def test_cost_accounting_and_horizon_terminal_reward(self):
        with Episode(scenario(deadline=2), Tape(None, None, False)) as episode:
            first, second = episode.step("wait"), episode.step("lookup")
            self.assertFalse(first["terminal"])
            self.assertTrue(second["terminal"])
            self.assertEqual(first["reward_cents"] + second["reward_cents"], episode.truth_receipt()["return_cents"])
            self.assertEqual(episode.truth_receipt()["return_cents"], -106)

    def test_forecasts_are_action_specific_and_not_a_categorical_distribution(self):
        with Episode(scenario(initial_weights=(0, 1, 0)), Tape(0, 1, True)) as episode:
            values = exact_forecasts(episode.scenario, episode.observe())
            self.assertEqual(values, {"retry": 0, "lookup": 1, "wait": 1, "abstain": 1})
            self.assertGreater(sum(values.values()), 1)
        with Episode(scenario(keyed=True), Tape(None, 1, False)) as episode:
            values = exact_forecasts(episode.scenario, episode.observe())
            self.assertEqual(values["retry"], 1)
            self.assertEqual(values["wait"], Fraction(1, 2))

    def test_same_retry_has_different_target_at_different_deadlines(self):
        for deadline, expected in [(2, 0), (3, 1)]:
            config = scenario(deadline=deadline, initial_weights=(1, 0, 0), retry_weights=(0, 0, 1))
            with self.subTest(deadline=deadline), Episode(config, Tape(None, 2, False)) as episode:
                episode.step("wait")
                state = episode.observe()
                self.assertEqual(exact_forecasts(config, state)["retry"], expected)
                query = public_input(config, state, "retry")["question"]
                self.assertIn(f"tick {deadline}", query)
                self.assertIn("no further requests or lookups", query)
                success, receipt = execute_counterfactual(config, state, "retry", Tape(None, 2, False))
                self.assertEqual(success, bool(expected))
                if not expected:
                    self.assertEqual(receipt["pending_after_deadline"], [[3, "retry"]])

    def test_exact_forecasts_condition_on_acquired_receipt(self):
        config = scenario()
        with Episode(config, Tape(None, 1, False)) as episode:
            episode.step("lookup")
            state = episode.observe()
            self.assertTrue(all(tape.initial_at is None for tape, _ in conditional_support(config, state)))
            self.assertEqual(exact_forecasts(config, state), {"retry": 1, "wait": 0, "abstain": 0})

    def test_unobserved_completion_and_future_transport_do_not_leak(self):
        config = scenario()
        inputs = []
        for tape in (Tape(None, None, False), Tape(0, 1, True), Tape(2, 2, False)):
            with Episode(config, tape) as episode:
                item = public_input(config, episode.observe(), "retry")
                inputs.append(item)
                for forbidden in ("tape", "initial_at", "completion_rows", "verifier_probability", "completed_jobs", "return_cents"):
                    self.assertNotIn(forbidden, item["state"])
                with self.assertRaises(TypeError):
                    public_input(config, episode)
                with self.assertRaises(ValueError):
                    episode.truth_receipt()
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual(inputs[1], inputs[2])

    def test_logical_receipt_token_does_not_leak_count_even_to_verifier(self):
        config = scenario(initial_weights=(1, 1, 0), ack_percent=100)
        states = []
        for initial in (None, 0):
            with Episode(config, Tape(initial, 1, True)) as episode:
                episode.step("retry")
                state = episode.observe()
                states.append(state)
                self.assertEqual(state.history[-1].receipt_token, receipt_token(config, "retry"))
                visible = public_input(config, state)["state"]
                self.assertNotIn('"receipt_id"', visible)
                self.assertNotIn('"receipt":2', visible)
                self.assertEqual(set(exact_forecasts(config, state).values()), {Fraction(1, 2)})
        self.assertEqual(states[0], states[1])
        self.assertEqual(public_input(config, states[0]), public_input(config, states[1]))

    def test_seed_tape_and_split_ownership_are_deterministic(self):
        config = make_scenario(41, 20)
        self.assertEqual(config, make_scenario(41, 20))
        support = list(tape_support(config))
        self.assertEqual(sum(weight for _, weight in support), 1)
        self.assertEqual(draw_tape(support, random.Random(51)), draw_tape(support, random.Random(51)))
        self.assertEqual(config.group_id, replace(config).group_id)
        self.assertEqual(split_owner(config.group_id), split_owner(replace(config).group_id))

    def test_independent_outcome_draws_can_disagree_at_the_same_public_state(self):
        config = scenario(initial_weights=(1, 1, 0), retry_weights=(1, 0, 0))
        with Episode(config, Tape(None, None, False)) as episode:
            state = episode.observe()
        support = conditional_support(config, state)
        outcomes = [execute_counterfactual(config, state, "wait", draw_tape(support, random.Random(seed)))[0]
                    for seed in range(30)]
        self.assertEqual(set(outcomes), {False, True})
        self.assertEqual(exact_forecasts(config, state)["wait"], Fraction(1, 2))

    def test_generated_files_replay_match_and_groups_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = (Path(temporary) / name for name in ("one", "two"))
            a, b = generate(first, worlds=35, audit_worlds=5, seed=9), generate(second, worlds=35, audit_worlds=5, seed=9)
            self.assertEqual(a, b)
            for name in (*a["files"], "summary.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            with self.assertRaises(ValueError):
                generate(first, worlds=1)
            read = lambda name: [json.loads(line) for line in (first / name).read_text().splitlines()]
            by_split = {key: set() for key in ("train", "validation", "test")}
            traces = read("exploration.jsonl")
            for trace in traces:
                by_split[trace["split"]].add(trace["group_id"])
                self.assertEqual(split_owner(trace["group_id"]), trace["split"])
                self.assertIn(trace["recorded_action_count"], (2, 3, 4))
                config = Scenario(**trace["scenario"])
                with Episode(config, Tape(**trace["truth_receipt"]["tape"])) as episode:
                    for record in trace["steps"]:
                        self.assertEqual(public_input(config, episode.observe()), record["input"])
                        self.assertTrue(record["valid_action_mask"][record["action"]])
                        self.assertEqual(record["action_probability"], 1 / sum(record["valid_action_mask"].values()))
                        episode.step(record["action"])
                    self.assertEqual(episode.truth_receipt(), trace["truth_receipt"])
            self.assertFalse(by_split["train"] & by_split["validation"] | by_split["train"] & by_split["test"] | by_split["test"] & by_split["validation"])
            examples = {row["id"]: row for row in read("forecast_examples.jsonl")}
            for receipt in read("forecast_replay.jsonl"):
                row = examples[receipt["id"]]
                self.assertNotIn("verifier_probability", row)
                self.assertEqual(digest(row["input"]), receipt["input_sha256"])
                success, truth = execute_counterfactual(Scenario(**receipt["scenario"]), observation(receipt["observation"]),
                                                      receipt["offered_action"], Tape(**receipt["truth_receipt"]["tape"]))
                self.assertEqual(truth, receipt["truth_receipt"])
                self.assertEqual(row["target"]["option_id"], "yes" if success else "no")
            audit_groups = set()
            for row in read("counterfactual_audit.jsonl"):
                audit_groups.add(row["group_id"])
                self.assertEqual(split_owner(row["group_id"]), "test")
                p = exact_forecasts(Scenario(**row["scenario"]), observation(row["observation"]))[row["offered_action"]]
                self.assertEqual([p.numerator, p.denominator], row["verifier_fraction"])
                self.assertNotIn("verifier_probability", row["input"]["state"])
            self.assertFalse(audit_groups & set.union(*by_split.values()))
            self.assertEqual(len(audit_groups), 5)

    def test_invalid_config_and_impossible_history_fail(self):
        with self.assertRaises(ValueError):
            scenario(initial_weights=(0, 0, 0))
        with self.assertRaises(ValueError):
            Tape(1, 0, True)
        config = scenario(initial_weights=(1, 0, 0))
        with Episode(config, Tape(None, 1, False)) as episode:
            state = episode.observe()
        impossible = replace(state, history=state.history + (VisibleEvent("lookup", 1, "receipt_snapshot", as_of=1, completed_jobs=9),),
                             tick=1, lookup_used=True, spent_cents=config.lookup_cost)
        with self.assertRaises(ValueError):
            conditional_support(config, impossible)


if __name__ == "__main__":
    unittest.main()
