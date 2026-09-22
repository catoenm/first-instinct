"""Offline provenance, execution and tokenization checks for outcome data v2."""

from collections import Counter
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from general_lab import outcome_data as data
from general_lab import outcome_prepare as prepared
from scale_lab.common import LABELS, MODELS, file_hash, messages, shuffled_input, write_json, write_rows


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class RecordingTokenizer:
    """Lossless character tokenizer; no model or network is ever loaded."""
    def __init__(self):
        self.prompts = []
        self.batch_arguments = []

    def apply_chat_template(self, turns, **kwargs):
        if kwargs != {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}:
            raise AssertionError("Unexpected chat-template behavior")
        return json.dumps(turns, sort_keys=True)

    def encode(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False}:
            raise AssertionError("Unexpected encoding behavior")
        return [ord(char) + 100 for char in text]

    def __call__(self, prompts, **kwargs):
        self.batch_arguments.append(kwargs)
        if kwargs != {"add_special_tokens": False, "truncation": False, "padding": False}:
            raise AssertionError("Tokenization must preserve every character")
        self.prompts.extend(prompts)
        return {"input_ids": [self.encode(prompt, add_special_tokens=False) for prompt in prompts]}


class OutcomeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.raw = cls.base / "raw"
        with redirect_stdout(io.StringIO()):
            cls.manifest = data.make_data(cls.raw, train_worlds=24, validation_worlds=24, test_worlds=24)
        cls.general = cls.base / "general"
        cls.general.mkdir()
        for split in ("train", "validation", "test"):
            rows = [{"id": f"general-{split}-{index}", "task": "alpha" if index % 2 else "beta",
                     "input_ids": [1, 2, 3 + index], "option_ids": ["yes", "no"], "target_indices": [index % 2]}
                    for index in range(12)]
            write_rows(cls.general / f"{split}.jsonl", rows)
        write_json(cls.general / "manifest.json", {"model_alias": "qwen35-9b", "model": MODELS["qwen35-9b"],
                   "outputs": {f"{split}.jsonl": file_hash(cls.general / f"{split}.jsonl")
                               for split in ("train", "validation", "test")}})

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(dir=self.base)
        self.addCleanup(self.work.cleanup)
        self.path = Path(self.work.name)

    def tokenize(self, raw=None, general=None, output=None, max_tokens=20000):
        tokenizer = RecordingTokenizer()
        output = output or self.path / "prepared"
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as factory, redirect_stdout(io.StringIO()):
            manifest = prepared.prepare(raw or self.raw, output, general or self.general,
                                        max_tokens=max_tokens, replay_limit=5)
        factory.assert_called_once_with(MODELS["qwen35-9b"]["id"], revision=MODELS["qwen35-9b"]["revision"],
                                        token=False, trust_remote_code=False)
        return manifest, tokenizer, output

    def test_generated_files_are_deterministic_and_hashed(self):
        other = self.path / "raw"
        with redirect_stdout(io.StringIO()):
            repeated = data.make_data(other, train_worlds=24, validation_worlds=24, test_worlds=24)
        self.assertEqual(self.manifest, repeated)
        for filename, checksum in self.manifest["outputs"].items():
            self.assertEqual(file_hash(self.raw / filename), checksum)
            self.assertEqual((self.raw / filename).read_bytes(), (other / filename).read_bytes())
        with self.assertRaises(FileExistsError):
            data.make_data(other, 1, 1, 1)

    def test_split_owners_root_prefixes_and_empirical_class_coverage(self):
        groups = {split: set(self.manifest["groups"][split]) for split in data.SEEDS}
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(groups[first] & groups[second])
        for split in data.SEEDS:
            self.assertEqual({row["split"] for row in read_rows(self.raw / f"{split}.jsonl")}, {split})
            for name in data.ENVIRONMENTS:
                module = data.environment(name)
                actual = {row["target"]["option_id"] for row in read_rows(self.raw / f"{split}.jsonl")
                          if row["task"] == name + "/outcome"}
                self.assertEqual(actual, set(module.terminal_utilities()))
            for root in read_rows(self.raw / f"{split}-roots.jsonl"):
                module = data.environment(root["environment"])
                scenario = module.Scenario(**root["scenario"])
                self.assertEqual(module.split_owner(scenario), split)
                self.assertEqual(root["group_id"], root["environment"] + ":" + module.mechanism_id(scenario))
                with module.EpisodeAdapter(scenario, module.Tape(**root["tape"])) as episode:
                    visited = []
                    for step in root["steps"]:
                        visited.append(asdict(episode.observe()))
                        self.assertEqual(episode.input(), step["input"])
                        self.assertEqual(step["action_probability"], 1 / len(episode.legal_actions()))
                        result = episode.step(step["action"])
                        self.assertEqual(result["reward_cents"], step["reward_cents"])
                    self.assertEqual(data.serializable(episode.truth_receipt()), root["truth_receipt"])
                    self.assertIn(root["selected_observation"], data.serializable(visited))

    def test_each_pair_uses_one_fresh_executed_rollout_and_accounts_for_singletons(self):
        different_from_actor = Counter()
        for split in data.SEEDS:
            roots = {row["id"]: row for row in read_rows(self.raw / f"{split}-roots.jsonl")}
            rows = {row["id"]: row for row in read_rows(self.raw / f"{split}.jsonl")}
            audit = {row["id"]: row for row in read_rows(self.raw / f"{split}-audit.jsonl")}
            counts = Counter()
            for receipt in read_rows(self.raw / f"{split}-replay.jsonl"):
                root = roots[receipt["root_id"]]
                name = receipt["environment"]
                module = data.environment(name)
                scenario = module.Scenario(**root["scenario"])
                observation = module.observation_from_dict(root["selected_observation"])
                fresh_seed = data.seed_for(root["id"], receipt["action"], "fresh-conditional-outcome")
                bundle = module.forecast_bundle(scenario, observation, receipt["action"], fresh_seed)
                self.assertEqual(data.serializable(bundle["replay"]), receipt["replay"])
                self.assertEqual(receipt["id"], data.digest([root["id"], receipt["action"]]))
                private = bundle["replay"]
                truth = private["result"]["truth_receipt"] if name == "retry" else private["truth_receipt"]
                different_from_actor[name] += data.serializable(truth["tape"]) != root["tape"]
                label = module.OUTCOMES[truth["completed_jobs"]] if name == "retry" else truth["outcome"]
                self.assertEqual(bundle["outcome_target"], label)
                self.assertEqual(int(bundle["cost_target"]), truth["spent_cents"] - observation.spent_cents)
                for kind in ("outcome", "cost"):
                    item, target = bundle[kind + "_input"], bundle[kind + "_target"]
                    counts[name + "/" + kind + "/" + target] += 1
                    key = receipt["id"] + "-" + kind
                    if len(item["options"]) == 1:
                        counts[name + "/known_" + kind] += 1
                        self.assertNotIn(key, rows)
                    else:
                        counts[name + "/" + kind + "_rows"] += 1
                        self.assertEqual(rows[key]["input"], item)
                        self.assertEqual(rows[key]["target"], {"option_id": target})
                        self.assertNotIn("replay", rows[key])
                        self.assertEqual(rows[key]["label_origin"], "executed_independent_conditional_draw")
                if split != "train":
                    self.assertEqual(audit[receipt["id"]]["outcome_target"], bundle["outcome_target"])
                    self.assertEqual(audit[receipt["id"]]["cost_target"], bundle["cost_target"])
            for key, value in counts.items():
                self.assertEqual(self.manifest["counts"][split][key], value)
            for name in data.ENVIRONMENTS:
                self.assertGreater(counts[name + "/known_cost"], 0)
            if split == "train":
                self.assertEqual(audit, {})
        self.assertTrue(all(different_from_actor[name] > 0 for name in data.ENVIRONMENTS))

    def test_no_exact_forecast_is_used_to_generate_training_targets(self):
        seen = Counter()
        with ExitStack() as stack:
            for name in data.ENVIRONMENTS:
                module = data.environment(name)
                original = module.exact_forecast
                def checked(scenario, observation, action, name=name, module=module, original=original):
                    owner = module.split_owner(scenario)
                    self.assertNotEqual(owner, "train")
                    seen[name + "/" + owner] += 1
                    return original(scenario, observation, action)
                stack.enter_context(patch.object(module, "exact_forecast", side_effect=checked))
            with redirect_stdout(io.StringIO()):
                data.make_data(self.path / "raw", 2, 2, 2)
        self.assertEqual(set(seen), {name + "/" + split for name in data.ENVIRONMENTS for split in ("validation", "test")})

    def test_generator_rejects_cross_split_mechanism_collision(self):
        module = data.environment("retry")
        with patch.object(module, "mechanism_id", return_value="forged-common-mechanism"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Cross-split mechanism"):
                data.make_data(self.path / "raw", 1, 1, 1)

    def test_programmatic_world_counts_fail_before_creating_output(self):
        for invalid in (0, -1, True, 1.5):
            output = self.path / str(invalid)
            with self.subTest(count=invalid), self.assertRaises(ValueError):
                data.make_data(output, invalid, 1, 1)
            self.assertFalse(output.exists())

    def test_tokenization_is_complete_deterministic_and_targets_follow_shuffled_options(self):
        manifest, tokenizer, output = self.tokenize()
        second, second_tokenizer, other = self.tokenize(output=self.path / "prepared-again")
        self.assertEqual(manifest, second)
        self.assertEqual(tokenizer.prompts, second_tokenizer.prompts)
        self.assertEqual(manifest["label_token_ids"], [ord(label) + 100 for label in LABELS])
        for filename, checksum in manifest["outputs"].items():
            self.assertEqual(file_hash(output / filename), checksum)
            self.assertEqual((output / filename).read_bytes(), (other / filename).read_bytes())
        for split in data.SEEDS:
            raw = {row["id"]: row for row in read_rows(self.raw / f"{split}.jsonl")}
            encoded = read_rows(output / f"{split}.jsonl")
            self.assertEqual(len(encoded), len(raw))
            self.assertEqual(manifest["counts"][split], len(raw))
            for row in encoded:
                item = shuffled_input(raw[row["id"]]["input"], "outcome-v2:" + row["id"])
                expected_prompt = json.dumps(messages(item), sort_keys=True)
                self.assertEqual(row["input_ids"], [ord(char) + 100 for char in expected_prompt])
                self.assertEqual(row["option_ids"][row["target_indices"][0]], raw[row["id"]]["target"]["option_id"])

    def test_general_replay_is_train_only_and_retention_is_validation_only(self):
        manifest, _, output = self.tokenize()
        replay, retention = read_rows(output / "replay.jsonl"), read_rows(output / "retention.jsonl")
        self.assertEqual(len(replay), 5)
        self.assertTrue(all(row["id"].startswith("general-train-") for row in replay))
        self.assertTrue(all(row["id"].startswith("general-validation-") for row in retention))
        self.assertEqual(Counter(row["task"] for row in retention), {"alpha": 4, "beta": 4})
        self.assertEqual(manifest["replay"]["split"], "train")
        self.assertEqual(manifest["retention"]["split"], "validation")
        self.assertNotIn("general-test-", (output / "replay.jsonl").read_text() + (output / "retention.jsonl").read_text())

    def test_private_top_level_fields_and_receipts_never_enter_prompts(self):
        raw = self.path / "raw"
        shutil.copytree(self.raw, raw)
        rows = read_rows(raw / "train.jsonl")
        rows[0]["verifier_probability"] = "SECRET_VERIFIER_TARGET"
        rows[0]["input"]["truth_receipt"] = "SECRET_DATABASE_TAPE"
        rows[0]["private_source_id"] = "SECRET_SOURCE_ID"
        write_rows(raw / "train.jsonl", rows)
        manifest = json.loads((raw / "manifest.json").read_text())
        manifest["outputs"]["train.jsonl"] = file_hash(raw / "train.jsonl")
        write_json(raw / "manifest.json", manifest)
        _, tokenizer, _ = self.tokenize(raw=raw)
        all_prompts = "\n".join(tokenizer.prompts)
        for secret in ("SECRET_VERIFIER_TARGET", "SECRET_DATABASE_TAPE", "SECRET_SOURCE_ID"):
            self.assertNotIn(secret, all_prompts)

    def test_overlength_rows_fail_instead_of_truncating_or_disappearing(self):
        with self.assertRaisesRegex(ValueError, "Overlength input.*no truncation or silent exclusion"):
            self.tokenize(max_tokens=20)
        self.assertFalse((self.path / "prepared" / "manifest.json").exists())

    def test_raw_replay_and_retention_checksum_mismatches_fail_closed(self):
        for area, filename, message in (("raw", "train.jsonl", "Raw data checksum"),
                                        ("general", "train.jsonl", "Replay model or file checksum"),
                                        ("general", "validation.jsonl", "General validation checksum")):
            with self.subTest(area=area, filename=filename):
                copy = self.path / (area + "-" + filename)
                shutil.copytree(self.raw if area == "raw" else self.general, copy)
                with (copy / filename).open("a") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, message):
                    self.tokenize(raw=copy if area == "raw" else None,
                                  general=copy if area == "general" else None,
                                  output=self.path / ("prepared-" + area + filename))

    def test_preparer_rejects_wrong_split_rows_even_with_matching_file_hash(self):
        raw = self.path / "raw"
        shutil.copytree(self.raw, raw)
        rows = read_rows(raw / "train.jsonl")
        rows[0]["split"] = "test"
        write_rows(raw / "train.jsonl", rows)
        manifest = json.loads((raw / "manifest.json").read_text())
        manifest["outputs"]["train.jsonl"] = file_hash(raw / "train.jsonl")
        write_json(raw / "manifest.json", manifest)
        with self.assertRaises(ValueError):
            self.tokenize(raw=raw)

    def test_preparer_rejects_wrong_group_owner_and_cross_split_manifest(self):
        for mode in ("wrong-owner", "overlap"):
            with self.subTest(mode=mode):
                raw = self.path / mode
                shutil.copytree(self.raw, raw)
                manifest = json.loads((raw / "manifest.json").read_text())
                if mode == "wrong-owner":
                    rows = read_rows(raw / "train.jsonl")
                    rows[0]["group_id"] = manifest["groups"]["test"][0]
                    write_rows(raw / "train.jsonl", rows)
                    manifest["outputs"]["train.jsonl"] = file_hash(raw / "train.jsonl")
                else:
                    manifest["groups"]["train"].append(manifest["groups"]["test"][0])
                write_json(raw / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    self.tokenize(raw=raw, output=self.path / ("prepared-" + mode))


if __name__ == "__main__":
    unittest.main()
