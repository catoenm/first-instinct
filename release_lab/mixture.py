"""Assemble existing admitted questions without changing their learning roles."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from scale_lab.common import MODELS, digest, file_hash, write_json, write_rows
from .toucan import ROOT
from .toucan_audit import load_qualified, require_use as require_tool_use
from .retail_questions import require_use as require_retail_use

GENERAL = ROOT / "output/general-qwen35-9b-v2"
TOOLS = ROOT / "output/release-tool-data-v1"
FORECASTS = ROOT / "output/decision-source-registry-v1-qualified"
APPWORLD = ROOT / "output/appworld-supervised-v2-data"
RETAIL = ROOT / "output/release-retail-questions-v1"
PARENT = ROOT / "output/general-supervised-complete-v1/runs/supervised-01/best/adapter_model.safetensors"
PARENT_SHA = "882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a"


def read(path):
    return json.loads(path.read_text())


def sources():
    specs = []
    for split, role in (("train", "train"), ("validation", "development"), ("test", "known_regression"), ("challenge", "known_regression")):
        specs.append({"name": "general_" + split, "kind": "general", "path": GENERAL / f"{split}.jsonl", "role": role})
    for role in ("train", "development", "reserved_transfer"):
        specs.append({"name": "tools_" + role, "kind": "tools", "path": TOOLS / f"{role}-private.jsonl", "role": role})
    for split, role in (("train_candidate", "train"), ("development", "development"), ("reserved_transfer", "known_transfer"), ("exposed_diagnostic", "diagnostic")):
        specs.append({"name": "forecasts_" + split, "kind": "forecasts", "path": FORECASTS / f"{split}-forecasts.jsonl", "role": role, "original_role": split})
    for name, role in (("new_forecast", "train"), ("new_decision", "train"), ("development_forecast", "development"), ("development_change", "development"), ("development_decision", "development")):
        specs.append({"name": "appworld_" + name, "kind": "appworld", "path": APPWORLD / f"{name}.jsonl", "role": role})
    specs.append({"name": "retail", "kind": "retail", "path": RETAIL / "train-private.jsonl", "role": "train"})
    return specs


def manifests():
    g = read(GENERAL / "manifest.json")
    if g["model"] != MODELS["qwen35-9b"] or file_hash(PARENT) != PARENT_SHA:
        raise ValueError("General tokenizer/model identity or parent checkpoint changed")
    t = load_qualified(TOOLS)
    f = read(FORECASTS / "report.json")
    a = read(APPWORLD / "freeze.json")
    r = read(RETAIL / "admission.json")
    u = read(RETAIL / "usage-private.json")
    if f["status"] != "qualified_source_index" or a["model"] != g["model"] or t["model"] != g["model"]:
        raise ValueError("Unqualified source or model mismatch")
    if r["status"] != "qualified_paired_questions" or u["admission_sha256"] != file_hash(RETAIL / "admission.json"):
        raise ValueError("Retail admission binding differs")
    if t["label_token_ids"] != g["label_token_ids"] or a["label_token_ids"] != g["label_token_ids"]:
        raise ValueError("Inconsistent decision label tokens")
    for s in sources():
        expected = {"general": g["outputs"], "tools": t["outputs"], "forecasts": f["files"],
                    "appworld": a["files"], "retail": r["files"]}[s["kind"]][s["path"].name]
        if file_hash(s["path"]) != expected:
            raise ValueError("Source data changed: " + s["name"])
    return {"general": g, "tools": t, "forecasts": f, "appworld": a, "retail": r, "retail_usage": u}


def manifest_paths():
    return [GENERAL / "manifest.json", TOOLS / "manifest-private.json", TOOLS / "admission.json",
            FORECASTS / "report.json", FORECASTS / "preparation-freeze.json", APPWORLD / "freeze.json",
            RETAIL / "admission.json", RETAIL / "freeze.json", RETAIL / "usage-private.json",
            ROOT / "output/telecom-questions-v1-usage/usage-private.json"]


def freeze(out):
    manifests()
    paths = manifest_paths() + [s["path"] for s in sources()]
    paths += sorted((ROOT / "release_lab").glob("*.py"))
    paths += [ROOT / "docs/release-mixture-v1-protocol.md", ROOT / "tests/test_release_mixture.py", ROOT / "scale_lab/common.py"]
    out.mkdir(parents=True, exist_ok=False)
    plan = {"version": "release-mixture-v1", "model": MODELS["qwen35-9b"], "max_tokens": 4096,
            "parent_adapter_path": str(PARENT), "parent_adapter_sha256": PARENT_SHA,
            "files": {str(p.relative_to(ROOT)): file_hash(p) for p in paths}, "training_presentations": 0}
    write_json(out / "freeze.json", plan)
    return {"status": "frozen_before_assembly", "files": len(plan["files"])}


def normalize(row, spec, refs):
    kind, role = spec["kind"], spec["role"]
    n = len(row["option_ids"])
    if not 2 <= n <= 36 or len(set(row["option_ids"])) != n:
        raise ValueError("Invalid source option menu")
    if kind == "tools":
        require_tool_use(row, refs["tools"], role)
    elif kind == "retail":
        require_retail_use(row, refs["retail_usage"], role)
    elif kind == "forecasts":
        if row["role"] != spec["original_role"] or refs["forecasts"]["by_family"][row["family"]]["role"] != row["role"]:
            raise ValueError("Forecast mechanism usage changed")
    elif kind == "appworld" and row["role"] != role:
        raise ValueError("Protected application source role changed")
    soft = row.get("soft_target")
    acceptable = row.get("target_indices", [])
    if soft is not None:
        if (len(soft) != n or acceptable or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in soft)
                or not math.isclose(sum(soft), 1., abs_tol=1e-6, rel_tol=0)):
            raise ValueError("Invalid distribution target")
        contract = "categorical_distribution"
    else:
        if not acceptable or len(set(acceptable)) != len(acceptable) or any(type(i) is not int or not 0 <= i < n for i in acceptable):
            raise ValueError("Invalid acceptable action target")
        contract = "acceptable_set"
    if kind in ("forecasts", "retail") and contract != "categorical_distribution":
        raise ValueError("Verified distribution lost its target contract")
    if kind in ("general", "tools") and contract != "acceptable_set":
        raise ValueError("Behavior annotations changed target contract")
    if not row["input_ids"] or any(type(i) is not int or i < 0 for i in row["input_ids"]):
        raise ValueError("Invalid token sequence")
    if len(row["input_ids"]) > 4096:
        return None
    group = row.get("group_id", row.get("group", row.get("family")))
    if not isinstance(group, str) or not group:
        raise ValueError("Missing source ownership group")
    family = row.get("family", kind)
    task = row.get("task", "consequence_" + family)
    source_ref = {"pool": spec["name"], "row_id": row["id"], "row_sha256": digest(row),
                  "source_file_sha256": file_hash(spec["path"]) if "sha256" not in spec else spec["sha256"],
                  "group": kind + ":" + group, "family": family, "task": task}
    key = digest(row["input_ids"])
    return {"id": key, "role": role, "input_ids": row["input_ids"], "option_ids": row["option_ids"],
            "target_contract": contract, "target_indices": acceptable, "soft_target": soft,
            "source_refs": [source_ref], "task": task, "family": family,
            "group_id": kind + ":" + group}


def target_signature(row):
    # Option identifiers do not enter the prompt; equality concerns aligned
    # target positions and the target contract, not arbitrary external IDs.
    return digest({"contract": row["target_contract"], "n": len(row["option_ids"]),
                   "acceptable": sorted(row["target_indices"]), "probabilities": row["soft_target"]})


def merge_rows(rows):
    unique, roles = {}, defaultdict(set)
    for row in rows:
        for ref in row["source_refs"]:
            roles[ref["group"]].add(row["role"])
        old = unique.get(row["id"])
        if old is None:
            unique[row["id"]] = row
        elif old["role"] != row["role"] or target_signature(old) != target_signature(row):
            raise ValueError("Exact token collision in role/target: " + row["id"])
        else:
            old["source_refs"].extend(row["source_refs"])
    bad = {g: sorted(v) for g, v in roles.items() if len(v) > 1}
    if bad:
        raise ValueError("Source groups cross usage roles: " + json.dumps(bad, sort_keys=True)[:2000])
    return sorted(unique.values(), key=lambda r: r["id"])


def assemble(out):
    plan = read(out / "freeze.json")
    for path, sha in plan["files"].items():
        if file_hash(ROOT / path) != sha:
            raise ValueError("Frozen assembly input changed: " + path)
    if (out / "assembly.json").exists() or (out / "train-private.jsonl").exists():
        raise ValueError("Preserve prior pack attempt")
    refs = manifests()
    prepared, rejected, census = [], [], {}
    for spec in sources():
        spec["sha256"] = file_hash(spec["path"])
        counts = Counter()
        for line in spec["path"].open():
            raw = json.loads(line)
            counts["source_questions"] += 1
            row = normalize(raw, spec, refs)
            if row is None:
                rejected.append({"source": spec["name"], "source_id": raw["id"], "source_row_sha256": digest(raw),
                                 "reason": "over_4096_tokens_no_truncation", "tokens": len(raw["input_ids"])})
                counts["overlength"] += 1
            else:
                prepared.append(row)
                counts["eligible_questions"] += 1
                counts["eligible_tokens"] += len(row["input_ids"])
        census[spec["name"]] = dict(counts)
    write_rows(out / "excluded-private.jsonl", rejected)
    try:
        rows = merge_rows(prepared)
    except ValueError as exc:
        write_json(out / "REJECTED.json", {"status": "rejected_source_collision", "reason": str(exc), "census": census,
                                          "training_presentations": 0, "freeze_sha256": file_hash(out / "freeze.json")})
        raise
    counts = {}
    for role in sorted({r["role"] for r in rows}):
        subset = [r for r in rows if r["role"] == role]
        write_rows(out / (role + "-private.jsonl"), subset)
        counts[role] = {"unique_token_questions": len(subset), "source_presentations": sum(len(r["source_refs"]) for r in subset),
                        "source_groups": len({ref["group"] for r in subset for ref in r["source_refs"]}),
                        "tokens": sum(len(r["input_ids"]) for r in subset),
                        "target_contracts": dict(Counter(r["target_contract"] for r in subset))}
    result = {"status": "assembled_not_training_ready", "model": plan["model"], "max_tokens": 4096,
              "parent_adapter_sha256": PARENT_SHA, "source_census": census, "by_role": counts,
              "exact_agreement_duplicates_collapsed": len(prepared) - len(rows), "overlength_exclusions": len(rejected),
              "label_token_ids": refs["general"]["label_token_ids"], "training_presentations": 0, "optimizer_steps": 0,
              "new_world_executions": 0, "freeze_sha256": file_hash(out / "freeze.json"),
              "files": {p.name: file_hash(p) for p in out.glob("*-private.jsonl")},
              "remaining_gates": ["independent saved-pack audit", "exact sampling/weights and evaluation cohorts",
                                  "qualified pilot trainer, context/padding and restart checks", "fresh billing, price, recovery and provider stop deadline"]}
    write_json(out / "assembly.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "assemble"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(freeze(a.out) if a.mode == "freeze" else assemble(a.out), indent=2))


if __name__ == "__main__":
    main()
