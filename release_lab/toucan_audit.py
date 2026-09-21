"""Check saved release examples against pinned source rows before admitting them."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scale_lab.common import digest, encode, file_hash, messages, read_rows, write_json
from . import toucan as source


def check_witness(row, raw, witness, tokenizer):
    """Reconstruct the prompt directly, without calling the example converter."""
    assert witness in row["lineage"] and digest(raw) == witness["row_sha256"]
    users, chosen = [], None
    for m in source.parse(raw["messages"]):
        if m["role"] == "user":
            users.append(m["content"])
        if m["role"] != "assistant":
            continue
        if m.get("function_call"):
            assert not m.get("tool_calls")
            chosen = m["function_call"]
        elif m.get("tool_calls"):
            assert len(m["tool_calls"]) == 1
            chosen = m["tool_calls"][0]["function"]
        if chosen is not None:
            break
    assert chosen is not None and len(users) == 1
    assert source.request_key(users[0]) == source.request_key(raw["question"])
    definitions = [t.get("function", t) for t in source.parse(raw["available_tools"])]
    assert 2 <= len(definitions) <= 36 and len({d["name"] for d in definitions}) == len(definitions)
    options = [{"id": d["name"], "description": json.dumps({k: d.get(k, "") for k in ("name", "description", "parameters")}, sort_keys=True, ensure_ascii=False)} for d in definitions]
    options.sort(key=lambda o: digest({"seed": source.SEED, "request": source.request_key(users[0]), "option": o}))
    item = {"state": users[0], "question": source.QUESTION, "options": options}
    assert row["input"] == item
    assert row["id"] == digest(messages(item))
    assert row["target"] == {"option_id": chosen["name"]}
    assert row["target_kind"] == "observed_first_tool_imitation"
    assert row["option_ids"] == [o["id"] for o in options]
    assert row["target_indices"] == [row["option_ids"].index(chosen["name"])]
    assert row["input_ids"] == encode(tokenizer, item, 4096)
    assert row["token_sha256"] == digest(row["input_ids"])
    assert row["request_key"] == source.request_key(users[0])
    assert row["group_id"] == "toucan_request:" + row["request_key"]
    servers, remote = source.remote_schemas(raw)
    tools = source.offered_tools(raw, remote)
    assert set(servers) <= set(row["servers"])
    assert row["schema_keys"] == sorted({t["schema_key"] for t in tools})
    assert row["semantic_key"] == digest({"request": row["request_key"], "menu": sorted(t["schema_key"] for t in tools)})
    assert row["chosen_schema"] == next(t["schema_key"] for t in tools if t["name"] == chosen["name"])
    for t in tools:
        validator = source.checked_validator(json.dumps(t["parameters"], sort_keys=True))
        if t["name"] == chosen["name"]:
            args = source.parse(chosen["arguments"])
            assert isinstance(args, dict)
            validator.validate(args)


def check_disjoint(rows, owners):
    maps = {kind: defaultdict(set) for kind in ("servers", "schema_keys", "request_key", "token_sha256", "semantic_key")}
    for row in rows:
        assert row["role"] in source.ROLES
        for server in row["servers"]:
            assert server not in owners["blocked"] and owners["roles"][server] == row["role"]
        for kind, groups in maps.items():
            for key in row[kind] if isinstance(row[kind], list) else [row[kind]]:
                groups[key].add(row["role"])
    assert all(len(roles) == 1 for groups in maps.values() for roles in groups.values())
    return {kind: len(groups) for kind, groups in maps.items()}


def audit(folder):
    from transformers import AutoTokenizer
    manifest = json.loads((folder / "manifest-private.json").read_text())
    freeze_path = Path(manifest["freeze_path"])
    assert file_hash(freeze_path) == manifest["freeze_sha256"]
    source.verify_freeze(freeze_path)
    for name, sha in manifest["outputs"].items():
        assert file_hash(folder / name) == sha
    assert manifest["training_presentations"] == 0
    rows = []
    for role in source.ROLES:
        subset = read_rows(folder / f"{role}-private.jsonl")
        assert all(r["role"] == role for r in subset)
        rows.extend(subset)
    assert len(rows) == len(manifest["rows"]) == len({r["id"] for r in rows})
    assert len(rows) == len({r["token_sha256"] for r in rows})
    for row in rows:
        assert manifest["rows"][row["id"]] == {"sha256": digest(row), "role": row["role"]}
    owners = json.loads((folder / "ownership-private.json").read_text())
    disjoint = check_disjoint(rows, owners)
    witnesses = {}
    for row in rows:
        for w in row["lineage"]:
            key = (w["file"], w["row"])
            assert key not in witnesses
            witnesses[key] = (row, w)
    exclusions = {}
    for rejected in read_rows(folder / "excluded-private.jsonl"):
        for w in rejected["lineage"]:
            key = (w["file"], w["row"])
            assert key not in witnesses and key not in exclusions
            exclusions[key] = w
    assert len(witnesses) + len(exclusions) == 35227
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["id"], revision=manifest["model"]["revision"], local_files_only=True, trust_remote_code=False)
    records, checked, source_count = [], 0, 0
    for raw, witness in source.raw_rows():
        source_count += 1
        key = (witness["file"], witness["row"])
        try:
            records.append(source.inventory(raw, witness))
        except (source.Exclude, ValueError, KeyError, TypeError, AttributeError):
            pass
        if key in witnesses:
            row, old = witnesses[key]
            assert old == witness
            check_witness(row, raw, witness, tokenizer)
            checked += 1
        else:
            assert exclusions[key] == witness
    assert source_count == 35227 and checked == len(witnesses)
    old_requests, old_schemas, old_tokens, _ = source.legacy_index()
    assert owners == source.ownership(records, old_requests, old_schemas)
    assert not ({r["request_key"] for r in rows} & old_requests)
    assert not ({s for r in rows for s in r["schema_keys"]} & old_schemas)
    assert not ({r["token_sha256"] for r in rows} & old_tokens)
    counts = {role: {"questions": sum(r["role"] == role for r in rows),
                     "servers": len({s for r in rows if r["role"] == role for s in r["servers"]})} for role in source.ROLES}
    sufficient = (counts["train"]["questions"] >= 1000 and counts["train"]["servers"] >= 10
                  and all(counts[role]["questions"] >= 200 and counts[role]["servers"] >= 5 for role in source.ROLES[1:]))
    result = {"status": "qualified_supervised_behavior_only" if sufficient else "insufficient_grouped_coverage",
              "integrity_passed": True, "coverage_passed": sufficient,
              "manifest_sha256": file_hash(folder / "manifest-private.json"),
              "freeze_sha256": manifest["freeze_sha256"], "checked_source_witnesses": checked,
              "checked_questions": len(rows), "unique_keys": disjoint, "by_role": counts,
              "source_trajectory_accounting_complete": True, "training_presentations": 0,
              "limitations": ["Exact overlap checks do not detect semantic paraphrases or foundation pretraining exposure.",
                              "First-action teacher agreement is not verified task success or optimality.",
                              "Contiguous cached shards, not a random sample of all TOUCAN.",
                              "No none/clarify or outcome labels were manufactured."]}
    with (folder / "admission.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return result


def load_qualified(folder):
    admission = json.loads((folder / "admission.json").read_text())
    if admission["status"] != "qualified_supervised_behavior_only" or not admission["integrity_passed"] or not admission["coverage_passed"]:
        raise ValueError("Unqualified release corpus")
    path = folder / "manifest-private.json"
    if file_hash(path) != admission["manifest_sha256"]:
        raise ValueError("Candidate manifest changed after audit")
    manifest = json.loads(path.read_text())
    if manifest["freeze_sha256"] != admission["freeze_sha256"] or file_hash(manifest["freeze_path"]) != manifest["freeze_sha256"]:
        raise ValueError("Admission freeze mismatch")
    for name, sha in manifest["outputs"].items():
        if file_hash(folder / name) != sha:
            raise ValueError("Qualified data changed")
    return manifest


def require_use(row, manifest, requested_role):
    entry = manifest["rows"].get(row["id"])
    if not entry or entry["sha256"] != digest(row) or entry["role"] != requested_role or row["role"] != requested_role:
        raise ValueError("Row or usage role differs from admitted manifest")
    if requested_role not in source.ROLES or row["target_kind"] != "observed_first_tool_imitation":
        raise ValueError("Unsupported target contract")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    audit(p.parse_args().data)


if __name__ == "__main__":
    main()
