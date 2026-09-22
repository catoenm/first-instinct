"""Portable release invariants, using tiny valid adapter fixtures and no models."""

import copy
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest

from general_lab.package import FROZEN, RUNTIME, package, safetensor_fingerprint
from scale_lab.common import MODELS, file_hash


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def adapter(directory, value, spec, tensor_name=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "adapter_config.json", {"base_model_name_or_path": spec["id"],
                                                  "revision": None, "peft_type": "LORA", "r": 1})
    name = tensor_name or "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    header = json.dumps({name: {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    (directory / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", value))


def prediction():
    return {"id": "one", "group_id": "group-one", "task": "task", "choice": "a",
            "probabilities": {"a": .7, "b": .3}, "target_ids": ["a"]}


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.project, self.data, self.run = (self.root / x for x in ("project", "data", "run"))
        for relative in (*RUNTIME, *FROZEN):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture for " + relative + "\n")
        self.spec = copy.deepcopy(MODELS["qwen35-9b"])
        self.manifest = {"model": self.spec, "max_tokens": 1536, "counts": {"train": 10},
                         "outputs": {split + ".jsonl": "a" * 64 for split in ("train", "validation", "test", "challenge")}}
        write_json(self.data / "manifest.json", self.manifest)
        (self.data / "train.jsonl").write_text("TRAINING_CORPUS_MUST_NOT_BE_PACKAGED")
        for checkpoint, value in (("best", 1.), ("latest", 2.)):
            adapter(self.run / checkpoint, value, self.spec)
        write_json(self.run / "tokenizer/tokenizer.json", {"version": "1.0"})
        write_json(self.run / "tokenizer/tokenizer_config.json", {"tokenizer_class": "FixtureTokenizer"})
        self.receipt = {"status": "complete", "model": self.spec, "steps": 2, "best_step": 1,
                        "data_manifest_sha256": file_hash(self.data / "manifest.json"),
                        "initial_trainable": {"sha256": "a" * 64}, "final_trainable": {"sha256": "b" * 64},
                        "code_sha256": {relative: file_hash(self.project / relative) for relative in
                                        ("scale_lab/common.py", "scale_lab/model.py", "scale_lab/infer.py")}}
        write_json(self.run / "run.json", self.receipt)
        write_json(self.run / "baseline-metrics.json", {"macro_log_loss": 1.})
        write_json(self.run / "validation-step-1.jsonl", prediction())
        (self.run / "training.jsonl").write_text('{"step": 1, "loss": 0.5}\n')
        (self.run / "optimizer.pt").write_bytes(b"OPTIMIZER_SECRET")
        (self.run / "credentials.json").write_text('{"api_key":"PRIVATE"}')
        (self.run / "best/model.safetensors").write_bytes(b"BASE_WEIGHTS_MUST_NOT_BE_PACKAGED")
        (self.run / "best/credentials.txt").write_text("PRIVATE")

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, name="release.tar.gz", **kwargs):
        return package(self.run, self.data, self.root / name, project_root=self.project, **kwargs)

    def rl(self, selected=0):
        path = self.root / "rl"
        for checkpoint, value in (("best", 1. if selected == 0 else 3.), ("latest", 4.)):
            adapter(path / checkpoint, value, self.spec)
        shutil.copytree(self.run / "tokenizer", path / "tokenizer")
        receipt = {"status": "complete", "model": self.spec, "updates": 2, "optimizer_steps": 4,
                   "selected_update": selected,
                   "language_parameter_audit": {"changed_elements": 1},
                   "pure_policy_language_gradient": {"nonzero_elements": 1},
                   "starting_adapter_sha256": {name: file_hash(self.run / "best" / name)
                                               for name in ("adapter_config.json", "adapter_model.safetensors")}}
        write_json(path / "run.json", receipt)
        for name in ("baseline-metrics.json", "best-metrics.json", "latest-metrics.json"):
            write_json(path / name, {"validation": {"reward": 1}})
        (path / "optimizer-steps.jsonl").write_text('{"optimizer_step":1}\n')
        return path, receipt

    def evaluation(self, foundation=False):
        path = self.root / ("base-eval" if foundation else "trained-eval")
        record = prediction()
        if foundation:
            record["probabilities"] = {"a": .6, "b": .4}
        write_json(path / "test-predictions.jsonl", record)
        write_json(path / "metrics.json", {"model": self.spec,
                   "checkpoint": "foundation" if foundation else "best",
                   "run_sha256": None if foundation else file_hash(self.run / "run.json"),
                   "prepared_manifest_sha256": file_hash(self.data / "manifest.json"), "splits": {"test": {"n": 1}}})
        return path

    def test_portable_allowlist_determinism_and_extracted_verification(self):
        first = self.build()
        second = self.build("second.tar.gz")
        self.assertEqual(first["archive_sha256"], second["archive_sha256"])
        names = first["files"]
        self.assertIn("runs/supervised/best/adapter_model.safetensors", names)
        self.assertIn("runs/supervised/latest/adapter_model.safetensors", names)
        self.assertIn("runs/supervised/tokenizer/tokenizer.json", names)
        for relative in ("general_lab/serve.py", "general_lab/web/index.html", "general_lab/web/app.js",
                         "general_lab/web/style.css", "examples/general-decisions.json", "docs/general-demo.md"):
            self.assertIn(relative, names)
        self.assertFalse(any("optimizer.pt" in x or "credentials" in x or x.endswith("train.jsonl") or x.endswith("/model.safetensors") for x in names))
        extracted = self.root / "extracted"
        with tarfile.open(self.root / "release.tar.gz") as archive:
            self.assertTrue(all(member.isfile() and not member.issym() for member in archive))
            archive.extractall(extracted, filter="data")
        readme = (extracted / "first-instinct-general-v1/README.md").read_text()
        self.assertIn("python -m general_lab.serve --run runs/supervised --device auto --max-tokens 1536", readme)
        self.assertIn("[demo guide](docs/general-demo.md)", readme)
        result = subprocess.run([sys.executable, "verify.py"], cwd=extracted / "first-instinct-general-v1", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified", result.stdout)
        (extracted / "first-instinct-general-v1/example.json").write_text("changed")
        result = subprocess.run([sys.executable, "verify.py"], cwd=extracted / "first-instinct-general-v1", capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_incomplete_failed_zero_step_and_unknown_revision_rejected(self):
        for field, value in (("status", "training"), ("status", "failed"), ("steps", 0), ("best_step", 3)):
            receipt = {**self.receipt, field: value}
            write_json(self.run / "run.json", receipt)
            with self.assertRaises(ValueError):
                self.build()
            self.assertFalse((self.root / "release.tar.gz").exists())
        write_json(self.run / "run.json", self.receipt)
        self.manifest["model"] = {**self.spec, "revision": "main"}
        write_json(self.data / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.build()

    def test_model_manifest_runtime_and_adapter_mismatch_rejected(self):
        receipt = {**self.receipt, "data_manifest_sha256": "c" * 64}
        write_json(self.run / "run.json", receipt)
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            self.build()
        write_json(self.run / "run.json", self.receipt)
        original = (self.project / "scale_lab/model.py").read_text()
        (self.project / "scale_lab/model.py").write_text("different")
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            self.build()
        (self.project / "scale_lab/model.py").write_text(original)
        config = json.loads((self.run / "best/adapter_config.json").read_text())
        config["revision"] = "not-the-pinned-revision"
        write_json(self.run / "best/adapter_config.json", config)
        with self.assertRaisesRegex(ValueError, "Adapter revision"):
            self.build()

    def test_bounded_stop_requires_explicit_opt_in(self):
        write_json(self.run / "run.json", {**self.receipt, "status": "bounded_stop"})
        with self.assertRaisesRegex(ValueError, "bounded stops"):
            self.build()
        result = self.build(allow_bounded_stop=True)
        self.assertEqual(result["runs"][0]["status"], "bounded_stop")

    def test_rl_zero_selection_honesty_and_changed_latest(self):
        path, receipt = self.rl()
        result = self.build(rl_runs={"hybrid": path})
        self.assertFalse(result["runs"][1]["selected_contains_updates"])
        self.assertIn("not an RL-trained selection", result["runs"][1]["selection_note"])
        self.assertIn("runs/rl-hybrid/optimizer-steps.jsonl", result["files"])
        receipt["selected_update"] = 1
        write_json(path / "run.json", receipt)
        with self.assertRaisesRegex(ValueError, "selected-update receipt"):
            self.build("bad-selection.tar.gz", rl_runs={"hybrid": path})
        receipt["selected_update"] = 0
        write_json(path / "run.json", receipt)
        adapter(path / "latest", 1., self.spec)
        with self.assertRaisesRegex(ValueError, "Latest reinforcement adapter is unchanged"):
            self.build("unchanged.tar.gz", rl_runs={"hybrid": path})

    def test_rl_new_domain_traces_survive_without_widening_allowlist(self):
        path, _ = self.rl()
        included = [f"{checkpoint}-new_domain-{kind}.jsonl"
                    for checkpoint in ("baseline", "best", "latest")
                    for kind in ("forecasts", "policy")]
        excluded = ["best-new_domain-predictions.jsonl", "best-new_domain-policy.jsonl.bak",
                    "best-new_domain_extra-policy.jsonl", "candidate-new_domain-policy.jsonl",
                    "private/best-new_domain-policy.jsonl", "optimizer.pt", "credentials.json"]
        for name in included + excluded:
            write_json(path / name, {"fixture": name})
        result = self.build(rl_runs={"hybrid": path})
        destination = "runs/rl-hybrid/"
        for name in included:
            self.assertIn(destination + name, result["files"])
        for name in excluded:
            self.assertNotIn(destination + name, result["files"])
        with tarfile.open(self.root / "release.tar.gz") as archive:
            prefix = "first-instinct-general-v1/" + destination
            for name in included:
                with archive.extractfile(prefix + name) as stream:
                    self.assertEqual(stream.read(), (path / name).read_bytes())
            for name in excluded:
                self.assertNotIn(prefix + name, archive.getnames())

    def test_rl_parent_and_policy_gradient_evidence_are_required(self):
        path, receipt = self.rl(selected=1)
        receipt["starting_adapter_sha256"]["adapter_model.safetensors"] = "f" * 64
        write_json(path / "run.json", receipt)
        with self.assertRaisesRegex(ValueError, "starting adapter"):
            self.build(rl_runs={"hybrid": path})
        receipt["starting_adapter_sha256"]["adapter_model.safetensors"] = file_hash(self.run / "best/adapter_model.safetensors")
        receipt["pure_policy_language_gradient"]["nonzero_elements"] = 0
        write_json(path / "run.json", receipt)
        with self.assertRaisesRegex(ValueError, "pure-policy-gradient"):
            self.build(rl_runs={"hybrid": path})

    def test_evaluations_reports_and_freeze_match_included_artifacts(self):
        base, trained = self.evaluation(True), self.evaluation()
        report = self.root / "report"
        write_json(report / "report.json", {"schema": "first-instinct-paired-report-v1", "inputs": {
            "base_sha256": file_hash(base / "test-predictions.jsonl"), "trained_sha256": file_hash(trained / "test-predictions.jsonl")}})
        (report / "report.md").write_text("# Measured report\n")
        freeze = self.root / "freeze.json"
        write_json(freeze, {"documentation_sha256": {name: file_hash(self.project / name) for name in FROZEN},
                            "prepared_manifest_sha256": file_hash(self.data / "manifest.json"), "commit": "locally-recorded"})
        result = self.build(evaluations={"base": base, "trained": trained}, reports={"general": report}, freeze=freeze)
        self.assertIn("reports/general/report.json", result["files"])
        self.assertIn("provenance/runtime-freeze.json", result["files"])
        (self.project / FROZEN[0]).write_text("changed after freeze")
        with self.assertRaisesRegex(ValueError, "Frozen documentation mismatch"):
            self.build("changed-docs.tar.gz", freeze=freeze)

    def test_report_rl_inputs_match_the_same_bundled_run(self):
        path, _ = self.rl()
        traces = [f"{checkpoint}-new_domain-{kind}.jsonl"
                  for checkpoint in ("baseline", "best", "latest")
                  for kind in ("forecasts", "policy")]
        for name in traces:
            write_json(path / name, {"fixture": name})
        other = self.root / "other-rl"
        shutil.copytree(path, other)
        other_receipt = json.loads((other / "run.json").read_text())
        other_receipt["config"] = {"seed": 53}
        write_json(other / "run.json", other_receipt)
        write_json(other / "best-new_domain-policy.jsonl", {"fixture": "different run"})
        write_json(other / "best-shift-policy.jsonl", {"fixture": "only in the other run"})
        base, trained = self.evaluation(True), self.evaluation()
        report = self.root / "report"
        names = ["run.json", "baseline-metrics.json", "best-metrics.json", "latest-metrics.json", *traces]
        hashes = {name: file_hash(path / name) for name in names}
        document = {"schema": "first-instinct-paired-report-v1", "inputs": {
            "base_sha256": file_hash(base / "test-predictions.jsonl"),
            "trained_sha256": file_hash(trained / "test-predictions.jsonl")},
            "reinforcement_learning": {"runs": [{"path": "/nonexistent/historical/source",
                                                   "input_sha256": hashes}]}}
        write_json(report / "report.json", document)
        (report / "report.md").write_text("# Measured report\n")
        options = {"rl_runs": {"hybrid": path, "other": other},
                   "evaluations": {"base": base, "trained": trained}, "reports": {"general": report}}
        result = self.build(**options)
        self.assertIn("reports/general/report.json", result["files"])
        invalid = {
            "missing-file": {**hashes, "latest-shift-forecasts.jsonl": "a" * 64},
            "wrong-hash": {**hashes, traces[0]: "a" * 64},
            "wrong-run": {**hashes, "run.json": "f" * 64},
            "missing-run": {name: checksum for name, checksum in hashes.items() if name != "run.json"},
            "other-run-file": {**hashes, "best-shift-policy.jsonl": file_hash(other / "best-shift-policy.jsonl")},
            "other-run-hash": {**hashes, "best-new_domain-policy.jsonl": file_hash(other / "best-new_domain-policy.jsonl")},
            "unsafe-path": {**hashes, "../supervised/run.json": file_hash(self.run / "run.json")},
        }
        for label, references in invalid.items():
            with self.subTest(label=label):
                document["reinforcement_learning"]["runs"][0]["input_sha256"] = references
                write_json(report / "report.json", document)
                with self.assertRaisesRegex(ValueError, "[Rr]eport reinforcement"):
                    self.build(label + ".tar.gz", **options)
                self.assertFalse((self.root / (label + ".tar.gz")).exists())
        document["reinforcement_learning"]["runs"][0]["input_sha256"] = hashes
        write_json(report / "report.json", document)
        with self.assertRaisesRegex(ValueError, "Report reinforcement run is not included"):
            self.build("unbundled-run.tar.gz", **{**options, "rl_runs": {"other": other}})

    def test_unrelated_partial_or_raw_evaluation_outputs_rejected(self):
        path = self.evaluation()
        receipt = json.loads((path / "metrics.json").read_text())
        receipt["run_sha256"] = "f" * 64
        write_json(path / "metrics.json", receipt)
        with self.assertRaisesRegex(ValueError, "included checkpoint"):
            self.build(evaluations={"trained": path})
        receipt["run_sha256"] = file_hash(self.run / "run.json")
        receipt["splits"]["test"]["n"] = 2
        write_json(path / "metrics.json", receipt)
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.build(evaluations={"trained": path})
        write_json(path / "test-predictions.jsonl", {**prediction(), "input_ids": [1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "raw data"):
            self.build(evaluations={"trained": path})

    def test_symlinks_secrets_base_tensors_and_oversize_release_rejected(self):
        path = self.run / "best/adapter_model.safetensors"
        backup = self.root / "adapter.safetensors"
        path.replace(backup)
        path.symlink_to(backup)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.build()
        path.unlink(); backup.replace(path)
        write_json(self.run / "run.json", {**self.receipt, "config": {"api_key": "NEVER-PUBLISH"}})
        with self.assertRaisesRegex(ValueError, "Credential-like"):
            self.build()
        write_json(self.run / "run.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "size/file limit"):
            self.build(max_bytes=10)
        adapter(self.run / "best", 1., self.spec, tensor_name="model.layers.0.weight")
        with self.assertRaisesRegex(ValueError, "Only language low-rank"):
            self.build()

    def test_existing_archive_is_never_overwritten(self):
        self.build()
        original = file_hash(self.root / "release.tar.gz")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.build()
        self.assertEqual(file_hash(self.root / "release.tar.gz"), original)


if __name__ == "__main__":
    unittest.main()
