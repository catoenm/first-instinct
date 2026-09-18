"""Build an allowlisted portable general-model release, without publishing it.

Only adapters, tokenizers, receipts, named evaluations, documentation, and a
minimal inference runtime are eligible. The training corpus and base weights
are never copied, even if they exist beside the requested inputs.
"""

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile

from scale_lab.common import MODELS, ROOT, file_hash

RUNTIME = (
    "scale_lab/__init__.py", "scale_lab/common.py", "scale_lab/model.py", "scale_lab/infer.py",
    "general_lab/__init__.py", "general_lab/interface.py", "general_lab/serve.py",
    "general_lab/web/index.html", "general_lab/web/app.js", "general_lab/web/style.css",
    "examples/general-decisions.json", "docs/general-demo.md",
    "requirements-scale.txt", "requirements-scale-cuda.txt", "requirements-multitask.txt",
    "requirements-decision-lock.txt", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "licenses/Apache-2.0.txt", "licenses/CC-BY-4.0.txt", "licenses/CC0-1.0.txt",
)
FROZEN = (
    "docs/general-training-v1-protocol.md", "docs/general-data-sources.md",
    "docs/general-probes.md", "data/general/probes.jsonl",
)
TOKENIZER = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json", "tokenizer.model",
)
RUN_FILES = ("run.json", "training.jsonl", "optimizer-steps.jsonl", "rollouts.jsonl",
             "baseline-metrics.json", "baseline-predictions.jsonl", "best-metrics.json", "latest-metrics.json")
RUN_PATTERN = re.compile(r"(?:validation-step-[0-9]+|(?:baseline|best|latest)-(?:validation|test|shift|challenge|new_domain)-(?:forecasts|policy)|validation-update-[0-9]+-(?:forecasts|policy))\.jsonl$")
SPLITS = {"validation", "test", "challenge", "probes", "probes_reversed"}
SHA = re.compile(r"[0-9a-f]{64}$")
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")
SENSITIVE = {"apikey", "apitoken", "accesstoken", "authorization", "password", "privatekey", "secretkey", "credentials"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path):
    path = Path(path).absolute()
    require(not any(part in {".local", ".git", ".ssh", ".env"} for part in path.parts), "Private input path is forbidden")
    require(not any(parent.is_symlink() for parent in (path, *path.parents)), "Symlink inputs are forbidden")
    require(path.is_file(), f"Required regular file is missing: {path.name}")
    return path


def safe_json(path):
    value = json.loads(safe_path(path).read_text())
    def check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = re.sub(r"[^a-z]", "", str(key).casefold())
                require(normalized not in SENSITIVE or child in (None, "", False), "Credential-like field in release metadata")
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, float):
            require(math.isfinite(item), "Nonfinite release metadata")
    check(value)
    return value


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def safetensor_fingerprint(path):
    """Inspect adapter names/shapes and hash tensors without importing torch."""
    path = safe_path(path)
    size = path.stat().st_size
    require(8 < size <= 2 * 1024 ** 3, "Unexpected adapter file size")
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        require(0 < length <= min(16 * 1024 ** 2, size - 8), "Invalid adapter header length")
        header = json.loads(stream.read(length))
        require(isinstance(header, dict), "Invalid adapter header")
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        require(bool(tensors), "Adapter has no tensors")
        fingerprint = hashlib.sha256()
        spans = []
        for name, tensor in sorted(tensors.items()):
            require(re.search(r"\.lora_[AB](?:\.[^.]+)?\.weight$", name) is not None,
                    "Only language low-rank adapter tensors may be packaged")
            require("visual" not in name and "vision" not in name, "Unexpected vision adapter tensor")
            dtype, shape, offsets = tensor.get("dtype"), tensor.get("shape"), tensor.get("data_offsets")
            require(dtype in {"F32", "F16", "BF16"} and isinstance(shape, list) and len(shape) == 2
                    and all(positive_int(x) for x in shape), "Invalid adapter tensor shape or dtype")
            require(isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(x, int) and x >= 0 for x in offsets),
                    "Invalid adapter tensor offsets")
            begin, end = offsets
            require(end - begin == math.prod(shape) * (4 if dtype == "F32" else 2)
                    and end <= size - 8 - length, "Adapter tensor size mismatch")
            spans.append((begin, end))
            fingerprint.update(json.dumps([name, dtype, shape], separators=(",", ":")).encode())
            stream.seek(8 + length + begin)
            remaining = end - begin
            while remaining:
                chunk = stream.read(min(1024 ** 2, remaining))
                require(bool(chunk), "Truncated adapter tensor")
                fingerprint.update(chunk)
                remaining -= len(chunk)
        spans.sort()
        require(spans[0][0] == 0 and spans[-1][1] == size - 8 - length
                and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), "Adapter tensor offsets have gaps or overlap")
    return {"tensor_sha256": fingerprint.hexdigest(), "tensor_count": len(tensors)}


@dataclass
class Entry:
    source: Path | None
    content: bytes | None
    sha256: str
    bytes: int


class Files:
    def __init__(self, max_bytes):
        self.entries = {}
        self.max_bytes = max_bytes
        self.total = 0

    def add(self, name, source=None, content=None):
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and not any(p.startswith(".") for p in path.parts), "Unsafe artifact path")
        require(name not in self.entries, "Duplicate artifact path")
        if source is not None:
            source = safe_path(source)
            size, checksum = source.stat().st_size, file_hash(source)
        else:
            content = content.encode() if isinstance(content, str) else content
            require(isinstance(content, bytes), "Missing artifact content")
            size, checksum = len(content), hashlib.sha256(content).hexdigest()
        self.total += size
        require(self.total <= self.max_bytes and len(self.entries) < 4096, "Release exceeds its declared size/file limit")
        self.entries[name] = Entry(source, content, checksum, size)
        return checksum


def add_adapter(files, source, destination, spec):
    config = safe_json(source / "adapter_config.json")
    require(config.get("base_model_name_or_path") == spec["id"], "Adapter foundation does not match run")
    require(config.get("revision") in (None, spec["revision"]), "Adapter revision does not match run")
    require(config.get("peft_type") == "LORA" and positive_int(config.get("r")), "Unsupported adapter type")
    require(not config.get("modules_to_save"), "Full saved modules are not eligible for an adapter-only release")
    facts = safetensor_fingerprint(source / "adapter_model.safetensors")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        files.add(destination + "/" + name, source / name)
    return facts


def add_run(files, source, destination, spec, manifest_sha, allow_bounded_stop, supervised=None):
    source = Path(source)
    receipt = safe_json(source / "run.json")
    require(receipt.get("status") == "complete" or (allow_bounded_stop and receipt.get("status") == "bounded_stop"),
            "Run is not complete; bounded stops require --allow-bounded-stop")
    require(receipt.get("model") == spec, "Run model/revision does not match prepared data")
    is_rl = supervised is not None
    step = receipt.get("updates" if is_rl else "steps")
    selected = receipt.get("selected_update" if is_rl else "best_step")
    require(positive_int(step), "A release requires completed training updates")
    require(isinstance(selected, int) and not isinstance(selected, bool) and 0 <= selected <= step, "Invalid selected checkpoint step")
    if not is_rl:
        require(receipt.get("data_manifest_sha256") == manifest_sha, "Supervised data manifest mismatch")
        initial, final = receipt.get("initial_trainable", {}), receipt.get("final_trainable", {})
        require(SHA.fullmatch(str(initial.get("sha256", ""))) and SHA.fullmatch(str(final.get("sha256", "")))
                and initial["sha256"] != final["sha256"], "Supervised run lacks a changed-parameter receipt")
        require((source / "baseline-metrics.json").is_file(), "Supervised baseline evaluation is missing")
        if selected:
            require((source / f"validation-step-{selected}.jsonl").is_file(), "Selected supervised validation is missing")
    else:
        require(positive_int(receipt.get("optimizer_steps")), "Reinforcement run has no optimizer steps")
        require(receipt.get("language_parameter_audit", {}).get("changed_elements", 0) > 0
                and receipt.get("pure_policy_language_gradient", {}).get("nonzero_elements", 0) > 0,
                "Reinforcement run lacks language-update and pure-policy-gradient evidence")
        expected = receipt.get("starting_adapter_sha256", {})
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            require(expected.get(name) == file_hash(Path(supervised["path"]) / "best" / name),
                    "Reinforcement starting adapter is not the supplied supervised selection")
        replay = receipt.get("replay")
        if replay:
            require(replay.get("manifest_sha256") == manifest_sha, "Reinforcement replay manifest mismatch")
        if receipt["status"] == "complete":
            require(all((source / filename).is_file() for filename in
                        ("baseline-metrics.json", "best-metrics.json", "latest-metrics.json")),
                    "Completed reinforcement run is missing checkpoint evaluation receipts")
    facts = {checkpoint: add_adapter(files, source / checkpoint, destination + "/" + checkpoint, spec)
             for checkpoint in ("best", "latest")}
    if is_rl:
        start = supervised["adapters"]["best"]["tensor_sha256"]
        require(facts["latest"]["tensor_sha256"] != start, "Latest reinforcement adapter is unchanged")
        require((facts["best"]["tensor_sha256"] == start) == (selected == 0), "Reinforcement selected-update receipt disagrees with selected adapter")
    for name in TOKENIZER:
        path = source / "tokenizer" / name
        if path.exists() or path.is_symlink():
            files.add(destination + "/tokenizer/" + name, path)
    require(destination + "/tokenizer/tokenizer_config.json" in files.entries
            and destination + "/tokenizer/tokenizer.json" in files.entries, "Portable tokenizer files are missing")
    for path in sorted(source.iterdir()):
        if path.name in RUN_FILES or RUN_PATTERN.fullmatch(path.name):
            if path.suffix == ".json":
                safe_json(path)
            files.add(destination + "/" + path.name, path)
    selected_note = ("Selected update zero: best/ is the supervised starting adapter, not an RL-trained selection. "
                     "latest/ contains the completed reinforcement updates.") if is_rl and selected == 0 else (
                     "Selected step zero: best/ is the unchanged supervised starting point; latest/ contains completed updates.") if selected == 0 else "The selected checkpoint contains training updates."
    return {"path": str(source), "artifact_path": destination, "run_sha256": file_hash(source / "run.json"),
            "status": receipt["status"], "completed_updates": step, "selected_update": selected,
            "selected_contains_updates": selected > 0, "selection_note": selected_note, "adapters": facts,
            "parameter_evidence_note": "Gradient/change counts are receipt claims; tensor fingerprints independently compare packaged adapter values."}


def validate_report_rl_inputs(report, files, runs):
    if "reinforcement_learning" not in report:
        return
    section = report["reinforcement_learning"]
    require(isinstance(section, dict) and isinstance(section.get("runs"), list),
            "Invalid report reinforcement run references")
    for reference in section["runs"]:
        hashes = reference.get("input_sha256") if isinstance(reference, dict) else None
        require(isinstance(hashes, dict) and SHA.fullmatch(str(hashes.get("run.json", ""))),
                "Report reinforcement run receipt hash is missing or invalid")
        candidates = [run for run in runs if run["run_sha256"] == hashes["run.json"]]
        require(candidates, "Report reinforcement run is not included in this release")
        require(all(isinstance(filename, str) and PurePosixPath(filename).name == filename
                    and not filename.startswith(".") and SHA.fullmatch(str(checksum))
                    for filename, checksum in hashes.items()),
                "Invalid report reinforcement input filename or checksum")
        # Receipt identity determines the bundled run; historical source paths
        # in a report are not trusted or used to read additional files.
        require(any(all((entry := files.entries.get(run["artifact_path"] + "/" + filename))
                        is not None and entry.sha256 == checksum for filename, checksum in hashes.items())
                    for run in candidates),
                "Report reinforcement input is missing or checksum mismatched in its bundled run")


def prediction_count(path):
    count = 0
    with safe_path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            require(set(row) == {"id", "group_id", "task", "choice", "probabilities", "target_ids"},
                    "Prediction output contains unexpected fields; raw data is not eligible")
            probabilities = row["probabilities"]
            require(isinstance(probabilities, dict) and len(probabilities) >= 2
                    and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in probabilities.values())
                    and abs(sum(probabilities.values()) - 1) < 1e-4, "Invalid prediction probabilities")
            require(row["choice"] in probabilities and bool(row["target_ids"])
                    and set(row["target_ids"]) <= set(probabilities), "Invalid prediction choices or targets")
            count += 1
    return count


def add_evaluation(files, label, directory, spec, manifest_sha, run_hashes, probe_sha):
    require(NAME.fullmatch(label), "Invalid evaluation name")
    directory = Path(directory)
    receipt = safe_json(directory / "metrics.json")
    require(receipt.get("model") == spec and receipt.get("prepared_manifest_sha256") == manifest_sha,
            "Evaluation model or data manifest mismatch")
    checkpoint, run_hash = receipt.get("checkpoint"), receipt.get("run_sha256")
    require((checkpoint == "foundation" and run_hash is None)
            or (checkpoint in {"best", "latest"} and run_hash in run_hashes), "Evaluation does not identify an included checkpoint")
    splits = receipt.get("splits", {})
    require(bool(splits) and set(splits) <= SPLITS, "Unsupported or missing evaluation splits")
    if any(split.startswith("probes") for split in splits):
        require(receipt.get("probes_sha256") == probe_sha, "Evaluation probes do not match frozen release probes")
    root = "evaluations/" + label
    for split, measured in sorted(splits.items()):
        path = directory / (split + "-predictions.jsonl")
        require(prediction_count(path) == measured.get("n") and positive_int(measured.get("n")), "Evaluation prediction count mismatch")
        files.add(root + "/" + path.name, path)
    files.add(root + "/metrics.json", directory / "metrics.json")
    return {"artifact_path": root, "checkpoint": checkpoint, "run_sha256": run_hash, "splits": sorted(splits)}


VERIFY = '''"""Verify the extracted release files against artifact-manifest.json."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "artifact-manifest.json").read_text())
for name, item in manifest["files"].items():
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
        raise SystemExit("Missing or changed release file: " + name)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != item["sha256"]:
        raise SystemExit("Checksum mismatch: " + name)
print("Verified", len(manifest["files"]), "release files")
'''


def readme(spec, manifest, runs, freeze):
    lines = ["# First Instinct: portable general decision adapters", "",
             f"Foundation: `{spec['id']}` at revision `{spec['revision']}`.",
             "The foundation weights are downloaded separately from Hugging Face. This archive contains trained low-rank adapters, not a full foundation model.", "",
             "Verify the externally published archive checksum before extraction, then run `python verify.py` inside the extracted directory. The internal manifest checks files; it is not an independent signature.", "",
             "Create a Python environment and install `requirements-scale.txt` on a Mac or `requirements-scale-cuda.txt` with an appropriate CUDA runtime. The recorded run receipts identify the training package versions.", "",
             "```sh", "python -m scale_lab.infer --run runs/supervised --input example.json --device auto",
             "```", "", f"The prepared training input limit was {manifest['max_tokens']} tokens. Supply `--max-tokens {manifest['max_tokens']}` to enforce the same limit.",
             "`Predictor(run=...)` reads `run.json` and `best/`. Reinforcement run directories have the same layout. The bundled tokenizer is a reproducibility snapshot; the existing Predictor loads the tokenizer from the pinned foundation revision.",
             "To inspect latest weights without changing the selection, use `load_model(spec, device, adapter=Path(run) / 'latest')` from `scale_lab.model`. Latest is not necessarily the validation-selected checkpoint.", "",
             "Run the browser demo from this extracted directory:", "", "```sh",
             f"python -m general_lab.serve --run runs/supervised --device auto --max-tokens {manifest['max_tokens']}",
             "```", "",
             "Open [the local demo](http://127.0.0.1:8766/) and select **Run** to try the editable parcel example. For the broader question interface, use `python -m general_lab.interface --run runs/supervised --input examples/general-decisions.json --device auto`. See the [demo guide](docs/general-demo.md) for details and limitations.", "",
             "## Included runs", ""]
    for run in runs:
        lines.append(f"- `{run['artifact_path']}`: {run['status']}; {run['completed_updates']} updates; selected {run['selected_update']}. {run['selection_note']}")
    lines += ["", "The original run receipts are unchanged; absolute training paths inside their configs are historical provenance, not paths needed for inference.",
              "The frozen documentation is a packaging snapshot. " + ("Its hashes match the supplied locally recorded pre-run freeze receipt; Git history can support chronology, but this is not an externally trusted timestamp." if freeze else "No pre-run documentation freeze receipt was supplied; this bundle does not independently establish when those documents were written."), "",
              "## Scope", "", "Choice probabilities are conditional on the offered answers. They are not automatically calibrated success probabilities. Public benchmark exposure and authored-probe limits are documented in docs/.",
              "No optimizer state, critic pickle files, base weights, raw upstream corpus, tokenized training rows, credentials, or .local files are included. This is an inference-and-evidence release, not an exact-resume checkpoint.", ""]
    return "\n".join(lines)


def package(supervised_run, data, output, *, rl_runs=None, evaluations=None, reports=None,
            source_audit=None, freeze=None, project_root=ROOT, name="first-instinct-general-v1",
            allow_bounded_stop=False, max_bytes=4 * 1024 ** 3):
    """Return an external archive receipt; no network, publication, or mutation of runs."""
    require(NAME.fullmatch(name), "Invalid release directory name")
    require(positive_int(max_bytes), "max_bytes must be positive")
    output, data, project_root = Path(output), Path(data), Path(project_root)
    sidecar = Path(str(output) + ".manifest.json")
    require(not output.exists() and not sidecar.exists(), "Refusing to overwrite an existing release")
    files = Files(max_bytes)
    manifest = safe_json(data / "manifest.json")
    manifest_sha = file_hash(data / "manifest.json")
    spec = manifest.get("model")
    require(spec in MODELS.values(), "Foundation must match a supported immutable model revision")
    require(positive_int(manifest.get("max_tokens")) and positive_int(manifest.get("counts", {}).get("train")), "Invalid prepared manifest")
    require(all(SHA.fullmatch(str(manifest.get("outputs", {}).get(split + ".jsonl", "")))
                for split in ("train", "validation", "test", "challenge")), "Prepared split checksums are missing")
    files.add("provenance/prepared-manifest.json", data / "manifest.json")
    for relative in (*RUNTIME, *FROZEN):
        files.add(relative, project_root / relative)
    supervised = add_run(files, supervised_run, "runs/supervised", spec, manifest_sha, allow_bounded_stop)
    runs = [supervised]
    for label, path in sorted((rl_runs or {}).items()):
        require(NAME.fullmatch(label), "Invalid reinforcement run name")
        runs.append(add_run(files, path, "runs/rl-" + label, spec, manifest_sha, allow_bounded_stop, supervised))
    # The inference implementation must be the same code that produced the run.
    recorded_code = safe_json(Path(supervised_run) / "run.json").get("code_sha256", {})
    for relative in ("scale_lab/common.py", "scale_lab/model.py", "scale_lab/infer.py"):
        require(recorded_code.get(relative) == files.entries[relative].sha256, "Inference runtime differs from recorded supervised code")
    if source_audit:
        expected = manifest.get("source_manifests", {}).get("public_source_audit_sha256")
        require(expected == file_hash(safe_path(source_audit)), "Public source audit does not match prepared provenance")
        files.add("provenance/public-source-audit.json", source_audit)
    if freeze:
        frozen = safe_json(freeze)
        hashes = frozen.get("documentation_sha256", {})
        require(set(FROZEN[:3]) <= set(hashes), "Freeze receipt must identify protocol, source, and probe documents")
        for relative, checksum in hashes.items():
            require(relative in FROZEN and files.entries[relative].sha256 == checksum, "Frozen documentation mismatch")
        if frozen.get("prepared_manifest_sha256") is not None:
            require(frozen["prepared_manifest_sha256"] == manifest_sha, "Frozen prepared manifest mismatch")
        files.add("provenance/runtime-freeze.json", freeze)
    evaluation_receipts = []
    for label, directory in sorted((evaluations or {}).items()):
        evaluation_receipts.append(add_evaluation(files, label, directory, spec, manifest_sha,
                                                  {run["run_sha256"] for run in runs}, files.entries["data/general/probes.jsonl"].sha256))
    for label, directory in sorted((reports or {}).items()):
        require(NAME.fullmatch(label), "Invalid report name")
        directory = Path(directory)
        report = safe_json(directory / "report.json")
        require(report.get("schema") == "first-instinct-paired-report-v1", "Unsupported report schema")
        predictions = {entry.sha256 for path, entry in files.entries.items() if path.startswith("evaluations/") and path.endswith("-predictions.jsonl")}
        require(all(report.get("inputs", {}).get(key) in predictions for key in ("base_sha256", "trained_sha256")),
                "Report input predictions are not included in this release")
        validate_report_rl_inputs(report, files, runs[1:])
        for filename in ("report.json", "report.md"):
            files.add("reports/" + label + "/" + filename, directory / filename)
    # Preserve receipt bytes, but do not publish packaging-host absolute paths in the new index.
    portable_runs = [{key: value for key, value in run.items() if key != "path"} for run in runs]
    files.add("README.md", content=readme(spec, manifest, portable_runs, freeze))
    files.add("verify.py", content=VERIFY)
    files.add("example.json", content=json.dumps({"state": "Entry requires a badge. Ivo has a badge.",
              "question": "Does Ivo meet the entry requirement?", "options": [
                  {"id": "yes", "description": "The requirement is met"}, {"id": "no", "description": "The requirement is not met"}]}, indent=2) + "\n")
    index = {"schema": "first-instinct-general-release-v1", "model": spec, "prepared_manifest_sha256": manifest_sha,
             "runs": portable_runs, "evaluations": evaluation_receipts,
             "documentation_freeze_supplied": bool(freeze),
             "files": {path: {"sha256": entry.sha256, "bytes": entry.bytes} for path, entry in sorted(files.entries.items())},
             "checksum_scope": "Payload files; artifact-manifest.json and SHA256SUMS are covered by the external archive checksum."}
    files.add("artifact-manifest.json", content=json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n")
    files.add("SHA256SUMS", content="".join(f"{entry.sha256}  {path}\n" for path, entry in sorted(files.entries.items())))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".general-release-", suffix=".part", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
            for relative, entry in sorted(files.entries.items()):
                info = tarfile.TarInfo(name + "/" + relative)
                info.size, info.mode, info.mtime = entry.bytes, 0o644, 0
                if entry.source:
                    with entry.source.open("rb") as stream:
                        archive.addfile(info, stream)
                    require(file_hash(entry.source) == entry.sha256, "Release input changed during packaging")
                else:
                    archive.addfile(info, io.BytesIO(entry.content))
        # Inspect the archive itself, not only files that may have changed while copied.
        with tarfile.open(temporary, "r:gz") as archive:
            for member in archive:
                relative = member.name.removeprefix(name + "/")
                require(member.isfile() and relative in files.entries, "Unexpected archive member")
                actual = hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest()
                require(actual == files.entries[relative].sha256, "Packaged file checksum mismatch")
        require(not output.exists(), "Release output appeared during packaging")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {"schema": "first-instinct-general-archive-v1", "archive_name": output.name,
               "directory_name": name, "archive_bytes": output.stat().st_size, "archive_sha256": file_hash(output),
               "files": {path: {"sha256": entry.sha256, "bytes": entry.bytes} for path, entry in sorted(files.entries.items())},
               "runs": portable_runs, "prepared_manifest_sha256": manifest_sha}
    with sidecar.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return receipt


def named_paths(values):
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        require(separator and NAME.fullmatch(label) and bool(path), "Use NAME=PATH for named release inputs")
        require(label not in result, "Duplicate named release input")
        result[label] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="first-instinct-general-v1")
    parser.add_argument("--rl-run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--evaluation", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--allow-bounded-stop", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=4 * 1024 ** 3)
    args = parser.parse_args()
    receipt = package(args.supervised_run, args.data, args.output, rl_runs=named_paths(args.rl_run),
                      evaluations=named_paths(args.evaluation), reports=named_paths(args.report),
                      source_audit=args.source_audit, freeze=args.freeze, project_root=args.project_root,
                      name=args.name, allow_bounded_stop=args.allow_bounded_stop, max_bytes=args.max_bytes)
    print(json.dumps({key: receipt[key] for key in ("archive_name", "archive_bytes", "archive_sha256")}, indent=2))


if __name__ == "__main__":
    main()
