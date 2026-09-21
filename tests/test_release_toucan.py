import copy
import json
import unittest

from release_lab import toucan as t
from release_lab.toucan_audit import check_disjoint, check_witness, require_use
from scale_lab.common import digest


def fixture(server="server-0", target="inspect"):
    params = {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"], "additionalProperties": False}
    defs = [{"name": name, "description": "Inspect current state" if name == "inspect" else "Commit the change", "parameters": params} for name in ("inspect", "commit")]
    return {"uuid": "fixture", "question": "Check the item before changing it.",
            "messages": [{"role": "user", "content": "Check the item before changing it."},
                         {"role": "assistant", "content": "SECRET REASONING", "function_call": {"name": target, "arguments": '{"item":"SECRET ARGUMENT"}'}},
                         {"role": "function", "name": target, "content": "SECRET FUTURE RESULT"}],
            "available_tools": [{"type": "function", "function": d} for d in defs],
            "metadata": {"mcp_servers": [{"server_id": server, "remote_server_response": {"tools": [
                {"name": d["name"], "description": d["description"], "input_schema": d["parameters"]} for d in defs]}}]},
            "response_quality_assessment": {"score": "SECRET JUDGE"}}


def make(raw):
    w = {"file": "fixture", "row": 0, "row_sha256": digest(raw)}
    r = t.inventory(raw, w)
    return t.example(raw, r, {"roles": {s: "train" for s in r["servers"]}, "blocked": {}})


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)

    def encode(self, prompt, **kwargs):
        return list(prompt.encode())


class ReleaseToucanTests(unittest.TestCase):
    def test_future_reasoning_argument_and_judge_never_visible(self):
        raw = fixture()
        row = make(raw)
        self.assertNotIn("SECRET", json.dumps(row["input"]))
        raw["messages"][2]["content"] = "different future result"
        raw["messages"][1]["content"] = "different thoughts"
        raw["messages"][1]["function_call"]["arguments"] = '{"item":"new argument"}'
        raw["response_quality_assessment"] = {}
        changed = make(raw)
        self.assertEqual(row["input"], changed["input"])
        self.assertEqual(row["target"], changed["target"])

    def test_target_and_source_order_do_not_affect_option_order(self):
        raw = fixture()
        row = make(raw)
        raw["available_tools"].reverse()
        raw["messages"][1]["function_call"]["name"] = "commit"
        raw["uuid"] = "other-teacher"
        self.assertEqual(row["input"], make(raw)["input"])

    def test_parallel_and_clarification_rejected(self):
        raw = fixture()
        call = raw["messages"][1].pop("function_call")
        raw["messages"][1]["tool_calls"] = [{"function": call}, {"function": call}]
        with self.assertRaisesRegex(t.Exclude, "parallel_first_calls"):
            make(raw)
        raw = fixture()
        raw["messages"].insert(1, {"role": "user", "content": "clarification"})
        with self.assertRaisesRegex(t.Exclude, "clarification_history"):
            make(raw)

    def test_required_type_and_extra_arguments(self):
        for args in ({}, {"item": 7}, {"item": "x", "extra": True}):
            raw = fixture()
            raw["messages"][1]["function_call"]["arguments"] = args
            with self.assertRaisesRegex(t.Exclude, "argument_schema_violation"):
                make(raw)

    def test_schema_refs_and_unknown_constraints_rejected(self):
        for s in ({"$ref": "https://example.invalid/schema"}, {"$ref": "#/definitions/x"}, {"type": "object", "nullable": True}, {"type": "string", "format": "unknown"}):
            with self.assertRaises(t.Exclude):
                t.checked_validator(json.dumps(s))
        with self.assertRaisesRegex(t.Exclude, "unsupported_schema_dialect"):
            t.checked_validator('{"$schema":"https://example.invalid/custom"}')

    def test_keyword_like_property_and_enum_values_are_data(self):
        schema = {"type": "object", "properties": {"$ref": {"enum": [{"$ref": "data"}]}}, "required": ["$ref"]}
        t.checked_validator(json.dumps(schema)).validate({"$ref": {"$ref": "data"}})

    def test_schema_pattern_not_whitespace_normalized(self):
        a = t.schema_key("x", "a  tool", {"pattern": "a  b"})
        b = t.schema_key("x", "a tool", {"pattern": "a b"})
        self.assertNotEqual(a, b)

    def test_duplicates_collapse_and_conflicts_reject_every_teacher(self):
        one = make(fixture())
        duplicate = copy.deepcopy(one)
        duplicate["lineage"][0]["row"] = 1
        kept, excluded = t.consolidate([one, duplicate])
        self.assertEqual((len(kept), len(kept[0]["lineage"]), len(excluded)), (1, 2, 0))
        conflict = make(fixture(target="commit"))
        kept, excluded = t.consolidate([one, conflict])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(excluded), 2)

    def test_server_alias_blocks_whole_lower_priority_group(self):
        train = next(str(i) for i in range(1000) if t.role_for(str(i)) == "train")
        reserve = next(str(i) for i in range(1000) if t.role_for(str(i)) == "reserved_transfer")
        rows = [{"servers": [train], "request_key": "first", "schemas": [{"server": train, "key": "shared"}]},
                {"servers": [reserve], "request_key": "second", "schemas": [{"server": reserve, "key": "shared"}]}]
        owners = t.ownership(rows)
        self.assertIn(train, owners["blocked"])
        self.assertNotIn(reserve, owners["blocked"])
        with self.assertRaisesRegex(t.Exclude, "blocked_server_group"):
            t.row_role({"servers": [train]}, owners)

    def test_request_collision_and_cross_role_compositions(self):
        a = next(str(i) for i in range(1000) if t.role_for(str(i)) == "train")
        b = next(str(i) for i in range(1000) if t.role_for(str(i)) == "development")
        records = [{"servers": [s], "schemas": [], "request_key": "same"} for s in (a, b)]
        self.assertIn(a, t.ownership(records)["blocked"])
        owners = {"roles": {a: "train", b: "development"}, "blocked": {}}
        with self.assertRaisesRegex(t.Exclude, "cross_role_composition"):
            t.row_role({"servers": [a, b]}, owners)

    def test_legacy_overlap_blocks_entire_server(self):
        records = [{"servers": ["0"], "schemas": [], "request_key": "same"}]
        self.assertIn("0", t.ownership(records, {"same"})["blocked"])

    def test_overlength_and_token_duplicates(self):
        row = make(fixture())
        rows, bad = t.tokenize([row, row], TinyTokenizer())
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["lineage"]), 2)
        row["input"]["state"] = "x" * 5000
        rows, bad = t.tokenize([row], TinyTokenizer())
        self.assertEqual(rows, [])
        self.assertEqual(bad[0]["reason"], "over_4096_tokens")

    def test_saved_export_witness_and_mutations(self):
        raw = fixture()
        row = t.tokenize([make(raw)], TinyTokenizer())[0][0]
        check_witness(row, raw, row["lineage"][0], TinyTokenizer())
        for field, value in (("target", {"option_id": "commit"}), ("input_ids", [123]), ("request_key", "changed")):
            bad = {**row, field: value}
            with self.assertRaises(AssertionError):
                check_witness(bad, raw, row["lineage"][0], TinyTokenizer())

    def test_usage_guard_rejects_tampering_and_reserved_training(self):
        row = make(fixture())
        manifest = {"rows": {row["id"]: {"sha256": digest(row), "role": "train"}}}
        require_use(row, manifest, "train")
        with self.assertRaises(ValueError):
            require_use(row, manifest, "reserved_transfer")
        with self.assertRaises(ValueError):
            require_use({**row, "target_kind": "success_probability"}, manifest, "train")

    def test_saved_cross_role_collision_rejected(self):
        row = t.tokenize([make(fixture())], TinyTokenizer())[0][0]
        owners = {"roles": {row["servers"][0]: "train"}, "blocked": {}}
        check_disjoint([row], owners)
        other = {**row, "role": "development", "servers": ["new"]}
        owners["roles"]["new"] = "development"
        with self.assertRaises(AssertionError):
            check_disjoint([row, other], owners)


if __name__ == "__main__":
    unittest.main()
