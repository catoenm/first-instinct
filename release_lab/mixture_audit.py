"""Verify a saved mixed pack against each frozen source without reconverting it."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from scale_lab.common import digest, file_hash, write_json
from .mixture import ROOT, PARENT, PARENT_SHA, sources


def signature(contract, n, acceptable, probabilities):
    return digest({"contract": contract, "n": n, "acceptable": sorted(acceptable), "probabilities": probabilities})


def validate_prepared(row, expected_role):
    if row["role"] != expected_role or row["id"] != digest(row["input_ids"]):
        raise ValueError("Pack input identity or role differs")
    if not 1 <= len(row["input_ids"]) <= 4096 or any(type(i) is not int or i < 0 for i in row["input_ids"]):
        raise ValueError("Invalid complete token input")
    n = len(row["option_ids"])
    if not 2 <= n <= 36 or len(set(row["option_ids"])) != n:
        raise ValueError("Invalid option menu")
    hard, q = row["target_indices"], row["soft_target"]
    kind = row["target_contract"]
    if kind == "categorical_distribution":
        if hard or not isinstance(q, list) or len(q) != n or any(not math.isfinite(v) or v < 0 for v in q) or not math.isclose(sum(q), 1., abs_tol=1e-6, rel_tol=0):
            raise ValueError("Invalid outcome probability contract")
    elif kind == "acceptable_set":
        if q is not None or not hard or len(set(hard)) != len(hard) or any(type(i) is not int or not 0 <= i < n for i in hard):
            raise ValueError("Invalid acceptable-action contract")
    else:
        raise ValueError("Unknown target contract")
    if not row["source_refs"] or len({(r["pool"], r["row_id"]) for r in row["source_refs"]}) != len(row["source_refs"]):
        raise ValueError("Missing or duplicate source witness")
    first = row["source_refs"][0]
    if row["group_id"] != first["group"] or row["task"] != first["task"] or row["family"] != first["family"]:
        raise ValueError("Primary metadata differs from source witness")
    return signature(kind, n, hard, q)


def audit(folder):
    manifest = json.loads((folder / "assembly.json").read_text())
    plan = json.loads((folder / "freeze.json").read_text())
    if manifest["freeze_sha256"] != file_hash(folder / "freeze.json") or file_hash(PARENT) != PARENT_SHA:
        raise ValueError("Mixture freeze or original parent changed")
    for path, sha in plan["files"].items():
        if file_hash(ROOT / path) != sha:
            raise ValueError("Assembly input changed: " + path)
    for name, sha in manifest["files"].items():
        if file_hash(folder / name) != sha:
            raise ValueError("Saved pack changed: " + name)
    audit_plan = {"assembly_sha256": file_hash(folder / "assembly.json"),
                  "auditor_sha256": file_hash(__file__),
                  "tests_sha256": file_hash(ROOT / "tests/test_release_mixture_audit.py"), "training_presentations": 0}
    with (folder / "audit-plan.json").open("x") as stream:
        json.dump(audit_plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    witnesses, groups, ids = {}, defaultdict(set), set()
    actual = {}
    for role in manifest["by_role"]:
        count, presentations, tokens = 0, 0, 0
        contracts, role_groups = Counter(), set()
        for line in (folder / f"{role}-private.jsonl").open():
            row = json.loads(line)
            target = validate_prepared(row, role)
            if row["id"] in ids:
                raise ValueError("Duplicate token input in pack")
            ids.add(row["id"])
            count += 1
            tokens += len(row["input_ids"])
            contracts[row["target_contract"]] += 1
            for position, ref in enumerate(row["source_refs"]):
                key = (ref["pool"], ref["row_id"])
                if key in witnesses:
                    raise ValueError("Source row represented more than once")
                witnesses[key] = {"ref": ref, "role": role, "token_sha256": row["id"], "target": target,
                                  "primary_option_ids": row["option_ids"] if position == 0 else None}
                groups[ref["group"]].add(role)
                role_groups.add(ref["group"])
                presentations += 1
        actual[role] = {"unique_token_questions": count, "source_presentations": presentations, "source_groups": len(role_groups),
                        "tokens": tokens, "target_contracts": dict(contracts)}
    if actual != manifest["by_role"] or any(len(v) > 1 for v in groups.values()):
        raise ValueError("Census or whole-group ownership differs")
    rejected = {}
    for line in (folder / "excluded-private.jsonl").open():
        r = json.loads(line)
        key = (r["source"], r["source_id"])
        if key in rejected or key in witnesses:
            raise ValueError("Exclusion duplicated or also retained")
        rejected[key] = r
    retained = excluded = source_rows = 0
    for spec in sources():
        sha = file_hash(spec["path"])
        for line in spec["path"].open():
            raw = json.loads(line)
            source_rows += 1
            key = (spec["name"], raw["id"])
            if len(raw["input_ids"]) > 4096:
                r = rejected.pop(key)
                if r != {"source": spec["name"], "source_id": raw["id"], "source_row_sha256": digest(raw),
                         "reason": "over_4096_tokens_no_truncation", "tokens": len(raw["input_ids"])}:
                    raise ValueError("Whole-question exclusion changed")
                excluded += 1
                continue
            w = witnesses.pop(key)
            group = raw.get("group_id", raw.get("group", raw.get("family")))
            family = raw.get("family", spec["kind"])
            task = raw.get("task", "consequence_" + family)
            expected_ref = {"pool": spec["name"], "row_id": raw["id"], "row_sha256": digest(raw),
                            "source_file_sha256": sha, "group": spec["kind"] + ":" + group, "family": family, "task": task}
            is_distribution = spec["kind"] in ("forecasts", "retail") or (spec["kind"] == "appworld" and raw.get("soft_target") is not None)
            target = signature("categorical_distribution" if is_distribution else "acceptable_set", len(raw["option_ids"]),
                               [] if is_distribution else raw["target_indices"], raw["soft_target"] if is_distribution else None)
            if (w["ref"] != expected_ref or w["role"] != spec["role"] or w["token_sha256"] != digest(raw["input_ids"])
                    or w["target"] != target or w["primary_option_ids"] is not None and w["primary_option_ids"] != raw["option_ids"]):
                raise ValueError("Saved question differs from its source witness")
            retained += 1
    if witnesses or rejected or excluded != manifest["overlength_exclusions"]:
        raise ValueError("Unaccounted source or exported question")
    result = {"status": "qualified_data_pack_not_training_runtime", "by_role": actual,
              "checked_source_rows": source_rows, "retained_source_rows": retained, "whole_overlength_exclusions": excluded,
              "assembly_sha256": file_hash(folder / "assembly.json"), "audit_plan_sha256": file_hash(folder / "audit-plan.json"),
              "training_presentations": 0, "optimizer_steps": 0,
              "remaining_gates": manifest["remaining_gates"][1:]}
    write_json(folder / "data-admission.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    audit(p.parse_args().data)


if __name__ == "__main__":
    main()
