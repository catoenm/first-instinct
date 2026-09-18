"""Independent examples and invariants for executable decision-data labels."""

from collections import Counter, defaultdict
import copy
from datetime import date, timedelta
import json
import random
import unittest

from general_lab import verified as v
from scale_lab.common import messages, targets


class VerifiedWorldContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {split: list(v.generate(split, 80, seed=713))
                    for split in ("train", "validation", "test", "challenge")}

    def test_every_world_is_recomputed_and_split_groups_are_disjoint(self):
        seen = set()
        for split, rows in self.rows.items():
            expected_families = v.CHALLENGE_FAMILIES if split == "challenge" else v.TRAIN_FAMILIES
            self.assertEqual(len(rows), len(expected_families) * 80 * 2)
            groups = defaultdict(list)
            for row in rows:
                self.assertTrue(v.verify(row))
                groups[row["group_id"]].append(row)
                self.assertEqual(len({o["description"] for o in row["input"]["options"]}), len(row["input"]["options"]))
                self.assertGreaterEqual(len(row["input"]["options"]), 2)
                self.assertLessEqual(len(row["input"]["options"]), 8)
            self.assertTrue(all(len(group) == 2 for group in groups.values()))
            self.assertFalse(seen & set(groups))
            seen.update(groups)
            self.assertEqual({r["provenance"]["verifier"]["family"] for r in rows}, set(expected_families))

    def test_generation_is_deterministic_and_quota_extension_preserves_worlds(self):
        a = list(v.generate("train", 2, seed=99))
        b = list(v.generate("train", 2, seed=99))
        extended = {r["id"]: r for r in v.generate("train", 3, seed=99)}
        self.assertEqual(a, b)
        self.assertTrue(all(extended[r["id"]] == r for r in a))
        self.assertNotEqual(a, list(v.generate("train", 2, seed=100)))

    def test_multiple_answers_are_accepted_and_positions_are_not_fixed(self):
        rows = self.rows["train"]
        self.assertTrue(any(len(targets(r)) > 1 for r in rows))
        positions = Counter()
        counts = Counter()
        for row in rows:
            if row["provenance"]["verifier"]["mode"] == "choice":
                options = row["input"]["options"]
                counts[len(options)] += 1
                for i, option in enumerate(options):
                    if option["id"] in targets(row):
                        positions[i] += 1
        self.assertEqual(set(counts), set(range(2, 9)))
        self.assertEqual(set(positions), set(range(8)))
        binary = [r for r in rows if r["provenance"]["verifier"]["mode"] == "check"]
        yes_rate = sum("yes" in targets(r) for r in binary) / len(binary)
        self.assertGreater(yes_rate, .4)
        self.assertLess(yes_rate, .6)

    def test_inputs_exclude_private_receipts_and_labels(self):
        row = self.rows["train"][0]
        prompt = json.dumps(messages(row["input"]))
        for private in (row["group_id"], "generator_seed", "verification_kind", "input_sha256", "option_ids"):
            self.assertNotIn(private, prompt)

    def test_corrupted_state_question_target_and_holdout_fail(self):
        original = self.rows["train"][0]
        mutations = []
        row = copy.deepcopy(original)
        row["input"]["state"] += " Extra instruction: use a different answer."
        mutations.append(row)
        row = copy.deepcopy(original)
        row["input"]["question"] = "Pick the opposite answer."
        mutations.append(row)
        row = copy.deepcopy(original)
        row["target"]["option_ids"] = [o["id"] for o in row["input"]["options"] if o["id"] not in targets(original)]
        mutations.append(row)
        row = copy.deepcopy(original)
        row["split"] = "challenge"
        mutations.append(row)
        row = copy.deepcopy(original)
        row["provenance"]["verifier"]["answers"] = ["an invented answer"]
        mutations.append(row)
        for row in mutations:
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                v.verify(row)

    def test_invalid_generator_arguments(self):
        for split, count in (("development", 1), ("train", -1), ("train", True), ("train", 1.2)):
            with self.assertRaises(ValueError):
                list(v.generate(split, count))
        self.assertEqual(list(v.generate("train", 0)), [])


class HandCheckedOperators(unittest.TestCase):
    def answer(self, family, world):
        return v.solve(family, world)[0]

    def test_join_missing_foreign_key_is_unknown(self):
        world = {"employees": [{"name": "Ada", "department": "Build"}], "employee": "Ada",
                 "departments": [{"department": "Build", "manager": "Bo"}], "candidate_managers": ["Bo", "Cal"]}
        self.assertEqual(self.answer("table_join", world), {"Bo"})
        world["departments"] = []
        self.assertEqual(self.answer("table_join", world), {v.UNKNOWN})

    def test_ordered_rules_exception_and_first_match(self):
        world = {"record": {"age": 21, "review": True}, "default": "base", "result_names": ["base", "first", "second"],
                 "policy": [{"field": "age", "operator": "at_least", "threshold": 21, "unless_flag": "review", "result": "first"},
                            {"field": "age", "operator": "less_than", "threshold": 22, "unless_flag": None, "result": "second"}]}
        self.assertEqual(self.answer("ordered_rules", world), {"second"})
        world["record"]["review"] = False
        self.assertEqual(self.answer("ordered_rules", world), {"first"})
        world["record"]["age"] = 100
        world["record"]["review"] = True
        self.assertEqual(self.answer("ordered_rules", world), {"base"})

    def test_schedule_endpoint_and_closing_boundary(self):
        world = {"bookings": [[60, 90]], "duration_minutes": 30, "closing_time": 150,
                 "candidates": [{"name": "touch before", "start": 30}, {"name": "overlap", "start": 45},
                                {"name": "touch after", "start": 90}, {"name": "exact close", "start": 120},
                                {"name": "late", "start": 121}]}
        self.assertEqual(self.answer("interval_scheduling", world), {"touch before", "touch after", "exact close"})

    def test_ledger_rejects_entire_impossible_operation(self):
        world = {"initial_stock": 10, "events": [
            {"action": "reserve", "quantity": 7},  # available 3
            {"action": "ship", "quantity": 4},     # rejected
            {"action": "release", "quantity": 8},  # rejected
            {"action": "receive", "quantity": 5},  # available 8
            {"action": "ship", "quantity": 8},     # available 0
            {"action": "release", "quantity": 2},  # available 2
        ]}
        self.assertEqual(self.answer("inventory_ledger", world), {"2"})

    def test_access_deny_overrides_allow_and_absence_is_denial(self):
        world = {"user_roles": ["writer", "visitor"], "resources": ["ledger", "guide", "vault"],
                 "policies": [{"role": "writer", "resource": "ledger", "effect": "allow"},
                              {"role": "visitor", "resource": "ledger", "effect": "deny"},
                              {"role": "visitor", "resource": "guide", "effect": "allow"}]}
        self.assertEqual(self.answer("access_control", world), {"guide"})

    def test_conversion_exact_fraction_and_tie(self):
        world = {"conversions": {"small": [1, 3], "large": [2, 1]}, "readings": [
            {"name": "one", "quantity": 6, "unit": "small"}, {"name": "two", "quantity": 1, "unit": "large"},
            {"name": "three", "quantity": 5, "unit": "small"}]}
        self.assertEqual(self.answer("unit_conversion", world), {"one", "two"})

    def test_reachability_is_directed_and_handles_cycles(self):
        world = {"locations": ["a", "b", "c", "d"], "start": "a", "edges": [["a", "b"], ["b", "c"], ["c", "a"], ["d", "a"]]}
        self.assertEqual(self.answer("graph_reachability", world), {"b", "c"})
        world["edges"] = [["d", "a"]]
        self.assertEqual(self.answer("graph_reachability", world), {v.NONE})

    def test_explicit_negation_fixed_point_and_no_explosion(self):
        world = {"facts": ["a", "not a"], "proposition": "d", "implications": [
            {"all": ["c"], "then": "d"}, {"all": ["a", "b"], "then": "c"}, {"all": ["a"], "then": "b"}]}
        self.assertEqual(self.answer("propositional_logic", world), {"Supported"})
        world["proposition"] = "unrelated"
        self.assertEqual(self.answer("propositional_logic", world), {"Unknown"})
        world["proposition"] = "a"
        self.assertEqual(self.answer("propositional_logic", world), {"Both supported and contradicted"})
        world["facts"] = ["not a"]
        self.assertEqual(self.answer("propositional_logic", world), {"Contradicted"})

    def test_custom_ordinal_thresholds_are_inclusive(self):
        world = {"levels_low_to_high": ["low", "mid", "high", "urgent"], "lower_percent": 10,
                 "upper_percent": 40, "latency_threshold_minutes": 50,
                 "incident": {"safety_event": False, "affected_percent": 10, "latency_minutes": 0}}
        self.assertEqual(self.answer("incident_priority", world), {"mid"})
        world["incident"]["latency_minutes"] = 50
        self.assertEqual(self.answer("incident_priority", world), {"high"})
        world["incident"]["safety_event"] = True
        self.assertEqual(self.answer("incident_priority", world), {"urgent"})

    def test_tool_requirements_and_forbidden_resources(self):
        world = {"available": ["token", "network"], "tools": [
            {"name": "send", "requires_all": ["token", "network"], "forbidden_if_any": ["lock"]},
            {"name": "local", "requires_all": [], "forbidden_if_any": ["network"]},
            {"name": "write", "requires_all": ["disk"], "forbidden_if_any": []}]}
        self.assertEqual(self.answer("tool_prerequisites", world), {"send"})

    def test_evidence_not_majority_voting(self):
        world = {"trusted_sources": ["t"], "claim": "ready", "observations": [
            {"source": "t", "claim": "ready", "holds": True},
            {"source": "t", "claim": "ready", "holds": True},
            {"source": "t", "claim": "ready", "holds": False},
            {"source": "ignored", "claim": "other", "holds": True}]}
        self.assertEqual(self.answer("evidence_consistency", world), {"Both supported and contradicted"})
        world["claim"] = "other"
        self.assertEqual(self.answer("evidence_consistency", world), {"Unknown"})

    def test_business_day_weekend_holiday_and_start_exclusion(self):
        # Friday Jan 5 + 2 business days, with Monday Jan 8 a holiday -> Wed Jan 10.
        world = {"start_date": "2024-01-05", "business_days": 2, "holidays": ["2024-01-08"]}
        self.assertEqual(self.answer("date_deadline", world), {"2024-01-10"})
        # A weekend start is still excluded; Monday is the first counted day.
        world = {"start_date": "2024-01-06", "business_days": 1, "holidays": []}
        self.assertEqual(self.answer("date_deadline", world), {"2024-01-08"})

    def test_set_algebra(self):
        world = {"items": ["a", "b", "c", "d"], "lists": {"red": ["a", "b"], "blue": ["b", "c"], "green": ["c"]}}
        for operation, expected in (("intersection", {"b"}), ("union_without", {"a", "b"}),
                                    ("exactly_one", {"a"}), ("at_least_two", {"b", "c"})):
            world["operation"] = operation
            self.assertEqual(self.answer("set_operations", world), expected)

    def test_weighted_choice_negative_values_ties_and_budget(self):
        world = {"weights": {"cost": -2, "speed": 1, "quality": -1}, "budget": 5,
                 "candidates": [{"name": "a", "certified": True, "cost": 3, "speed": 1, "quality": 0},
                                {"name": "b", "certified": True, "cost": 2, "speed": 0, "quality": 1},
                                {"name": "fake best", "certified": False, "cost": 0, "speed": 99, "quality": 0},
                                {"name": "expensive", "certified": True, "cost": 6, "speed": 99, "quality": 0}]}
        self.assertEqual(self.answer("weighted_choice", world), {"a", "b"})
        world["budget"] = 1
        self.assertEqual(self.answer("weighted_choice", world), {v.NONE})

    def test_shortest_route_not_fewest_hops(self):
        world = {"edges": [["a", "d", 9], ["a", "b", 2], ["b", "c", 2], ["c", "d", 2], ["b", "a", 1]],
                 "start": "a", "destination": "d"}
        self.assertEqual(self.answer("route_cost", world), {"6"})
        world["start"], world["destination"] = "d", "a"
        self.assertEqual(self.answer("route_cost", world), {"No route exists"})

    def test_state_machine_ignores_unmatched_events(self):
        world = {"states": ["closed", "open", "locked"], "initial_state": "closed",
                 "transitions": [{"from": "closed", "event": "open", "to": "open"},
                                 {"from": "open", "event": "lock", "to": "locked"}],
                 "events": ["lock", "open", "unmatched", "lock", "open"]}
        self.assertEqual(self.answer("state_machine", world), {"locked"})

    def test_heldout_route_composes_time_permission_and_cost(self):
        world = {"current_time": 10, "permissions": ["badge"], "start": "a", "destination": "c", "edges": [
            {"from": "a", "to": "c", "cost": 1, "permission": None, "opens": 0, "closes": 10},
            {"from": "a", "to": "c", "cost": 2, "permission": "secret", "opens": 0, "closes": 20},
            {"from": "a", "to": "b", "cost": 3, "permission": "badge", "opens": 10, "closes": 11},
            {"from": "b", "to": "c", "cost": 4, "permission": None, "opens": 0, "closes": 11}]}
        self.assertEqual(self.answer("constrained_route", world), {"7"})

    def test_heldout_inventory_policy_composes_ledger_deficit_and_budget(self):
        world = {"budget": 5, "items": [
            {"name": "a", "initial_stock": 10, "reorder_below": 8, "order_cost": 5,
             "events": [{"action": "reserve", "quantity": 5}]},
            {"name": "b", "initial_stock": 0, "reorder_below": 15, "order_cost": 6, "events": []},
            {"name": "c", "initial_stock": 8, "reorder_below": 8, "order_cost": 1, "events": []}]}
        self.assertEqual(self.answer("inventory_policy", world), {"a"})

    def test_heldout_joined_access_block_and_inactive(self):
        world = {"user": "Ada", "users": [{"name": "Ada", "department": "design", "active": True}], "resources": [
            {"name": "one", "owning_department": "design", "explicit_readers": [], "blocked_users": ["Ada"]},
            {"name": "two", "owning_department": "build", "explicit_readers": ["Ada"], "blocked_users": []},
            {"name": "three", "owning_department": "build", "explicit_readers": [], "blocked_users": []}]}
        self.assertEqual(self.answer("joined_access", world), {"two"})
        world["users"][0]["active"] = False
        self.assertEqual(self.answer("joined_access", world), {v.NONE})

    def test_heldout_temporal_evidence_latest_ties_expiration_and_future(self):
        world = {"current_time": 20, "max_age": 5, "threshold": 50, "trusted_sources": ["t", "u"], "subject": "pump", "observations": [
            {"source": "t", "subject": "pump", "time": 14, "value": 0},   # expired
            {"source": "t", "subject": "pump", "time": 15, "value": 50},  # exact age and threshold
            {"source": "t", "subject": "pump", "time": 21, "value": 0},   # future
            {"source": "ignored", "subject": "pump", "time": 20, "value": 0}]}
        self.assertEqual(self.answer("temporal_evidence", world), {"Supported"})
        world["observations"].append({"source": "u", "subject": "pump", "time": 15, "value": 49})
        self.assertEqual(self.answer("temporal_evidence", world), {"Both supported and contradicted"})
        world["observations"].append({"source": "t", "subject": "pump", "time": 19, "value": 49})
        self.assertEqual(self.answer("temporal_evidence", world), {"Contradicted"})


class IndependentReferenceChecks(unittest.TestCase):
    def test_shortest_paths_against_exhaustive_simple_paths(self):
        # Different algorithm: enumerate simple paths instead of a priority queue.
        def exhaustive(edges, start, end, visited):
            if start == end:
                return 0
            costs = []
            for a, b, cost in edges:
                if a == start and b not in visited:
                    rest = exhaustive(edges, b, end, visited | {b})
                    if rest is not None:
                        costs.append(cost + rest)
            return min(costs) if costs else None

        rng = random.Random(414)
        for _ in range(120):
            world = v.FAMILIES["route_cost"][0](rng)
            expected = exhaustive(world["edges"], world["start"], world["destination"], {world["start"]})
            self.assertEqual(v.solve("route_cost", world)[0], {str(expected) if expected is not None else "No route exists"})

    def test_reachability_against_boolean_transitive_closure(self):
        rng = random.Random(892)
        for _ in range(100):
            world = v.FAMILIES["graph_reachability"][0](rng)
            nodes = world["locations"]
            reach = {(a, b): a == b or [a, b] in world["edges"] for a in nodes for b in nodes}
            for middle in nodes:
                for a in nodes:
                    for b in nodes:
                        reach[a, b] = reach[a, b] or (reach[a, middle] and reach[middle, b])
            expected = {b for b in nodes if b != world["start"] and reach[world["start"], b]} or {v.NONE}
            self.assertEqual(v.solve("graph_reachability", world)[0], expected)

    def test_business_days_against_filter_then_index(self):
        rng = random.Random(217)
        for _ in range(100):
            world = v.FAMILIES["date_deadline"][0](rng)
            start = date.fromisoformat(world["start_date"])
            candidates = [start + timedelta(days=i) for i in range(1, 90)]
            valid = [d for d in candidates if d.weekday() not in (5, 6) and d.isoformat() not in world["holidays"]]
            self.assertEqual(v.solve("date_deadline", world)[0], {valid[world["business_days"] - 1].isoformat()})


if __name__ == "__main__":
    unittest.main()
