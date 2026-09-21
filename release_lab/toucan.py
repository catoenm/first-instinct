"""Conservative first-action imitation from pinned, local TOUCAN witnesses.

No model, network, source tool execution, success inference or quality-score use.
Ownership is computed from requests and declared schemas before reading labels.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import sys

from jsonschema import Draft7Validator, Draft202012Validator
from scale_lab.common import MODELS, digest, encode, file_hash, label_token_ids, messages, write_json, write_rows

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260921
ROLES = ("train", "development", "reserved_transfer")
PRIORITY = {r: i for i, r in enumerate(ROLES)}
REVISION = "0df3cf37f2abefb380370cfb02eabea2a35ae782"
SOURCE_RECEIPTS = (
    "output/toucan-audit-v3/source.json",
    "output/toucan-audit-oss-v1/source.json",
    "output/toucan-audit-qwen-v1/source.json",
)
LEGACY_DIRS = (
    "output/general-mixture-v3", "output/general-qwen35-9b-v2",
    "output/multitask_reproduction", "output/dataset_reproduction_v2",
    "output/scale-full-v2", "output/decision-source-registry-v1-qualified",
    "output/appworld-supervised-v2-data", "output/appworld-transfer-evaluation-v2-data",
    "output/contextual-shell-training-v1-data", "output/telecom-questions-v1-usage",
)
QUESTION = "Which available tool should be called first to begin fulfilling the user's request?"
PACKAGES = ("jsonschema", "referencing", "jsonschema-specifications", "rpds-py", "attrs", "pyarrow", "transformers", "tokenizers", "huggingface-hub")


class Exclude(ValueError):
    pass


def parse(value):
    def unique(pairs):
        obj = {}
        for key, val in pairs:
            if key in obj:
                raise Exclude("duplicate_json_key")
            obj[key] = val
        return obj
    def invalid(_):
        raise Exclude("nonfinite_json")
    return json.loads(value, object_pairs_hook=unique, parse_constant=invalid) if isinstance(value, str) else value


def request_key(text):
    # Whitespace only; preserve case, punctuation and numbers.
    return digest(" ".join(text.split()))


def schema_key(name, description, parameters):
    # Local tool names remove only a witnessed server prefix. Schema values,
    # including regex patterns, enum strings and required fields, stay exact.
    return digest({"name": name, "description": " ".join(description.split()), "parameters": parameters})


def role_for(server):
    bucket = int(digest({"seed": SEED, "server": str(server)})[:16], 16) % 100
    return "train" if bucket < 80 else "development" if bucket < 90 else "reserved_transfer"


def source_paths():
    from huggingface_hub import hf_hub_download
    found = []
    for receipt in SOURCE_RECEIPTS:
        spec = json.loads((ROOT / receipt).read_text())
        assert spec["revision"] == REVISION
        path = Path(hf_hub_download(spec["dataset"], spec["file"], repo_type="dataset", revision=REVISION, local_files_only=True))
        if file_hash(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError("Pinned source mismatch")
        found.append((spec, path))
    return found


def raw_rows():
    import pyarrow.parquet as pq
    for spec, path in source_paths():
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for raw in batch.to_pylist():
                witness = {"file": spec["file"], "source_sha256": spec["sha256"], "row": offset,
                           "uuid": raw["uuid"], "row_sha256": digest(raw)}
                yield raw, witness
                offset += 1


def remote_schemas(raw):
    metadata = parse(raw["metadata"])
    servers = metadata.get("mcp_servers", [])
    if not servers or any(s.get("server_id") is None for s in servers):
        raise Exclude("missing_server_attribution")
    ids = [str(s["server_id"]) for s in servers]
    if len(ids) != len(set(ids)):
        raise Exclude("duplicate_server_attribution")
    result = []
    for s in servers:
        for t in s.get("remote_server_response", {}).get("tools", []):
            if isinstance(t.get("name"), str) and isinstance(t.get("description", ""), str) and isinstance(t.get("input_schema"), dict):
                result.append({"server": str(s["server_id"]), "name": t["name"],
                               "description": t.get("description", ""), "parameters": t["input_schema"]})
    return sorted(ids), result


def offered_tools(raw, remote):
    tools = parse(raw["available_tools"])
    if not isinstance(tools, list) or not 2 <= len(tools) <= 36:
        raise Exclude("offered_tool_count")
    result = []
    for t in tools:
        if t.get("type", "function") != "function":
            raise Exclude("nonfunction_tool")
        f = t.get("function", t)
        if not isinstance(f.get("name"), str) or not f["name"] or not isinstance(f.get("description", ""), str) or not isinstance(f.get("parameters"), dict):
            raise Exclude("malformed_tool")
        # Require exact parameters/description and a witnessed unprefixed name.
        matches = [x for x in remote if (f["name"] == x["name"] or f["name"].endswith("-" + x["name"]))
                   and f.get("description", "") == x["description"] and f["parameters"] == x["parameters"]]
        if len(matches) != 1:
            raise Exclude("unresolved_tool_attribution")
        x = matches[0]
        result.append({"name": f["name"], "description": f.get("description", ""), "parameters": f["parameters"],
                       "server": x["server"], "schema_key": schema_key(x["name"], x["description"], x["parameters"])})
    if len({t["name"] for t in result}) != len(result):
        raise Exclude("duplicate_offered_name")
    return result


def inventory(raw, witness):
    ids, remote = remote_schemas(raw)
    if not isinstance(raw.get("question"), str) or not raw["question"].strip():
        raise Exclude("missing_public_question")
    return {"witness": witness, "servers": ids, "request_key": request_key(raw["question"]),
            "schemas": [{"server": t["server"], "key": schema_key(t["name"], t["description"], t["parameters"])} for t in remote]}


def ownership(records, old_requests=(), old_schemas=()):
    """Reserve first; remove whole lower-priority servers, never relabel rows.

    Co-occurrence is not a union: compositions spanning roles are excluded.
    Exact aliases and requests constrain every surviving member server. Removed
    servers stay removed even if a later removal makes a collision disappear.
    """
    roles = {s: role_for(s) for r in records for s in r["servers"]}
    blocked = defaultdict(set)
    aliases = defaultdict(set)
    for r in records:
        for s in r["schemas"]:
            aliases[s["key"]].add(s["server"])
            if s["key"] in old_schemas:
                blocked[s["server"]].add("legacy_schema")
        if r["request_key"] in old_requests:
            for server in r["servers"]:
                blocked[server].add("legacy_request")
    rounds = 0
    while True:
        rounds += 1
        before = set(blocked)
        joins = [("schema_alias", members - set(blocked)) for members in aliases.values()]
        requests = defaultdict(set)
        for r in records:
            if any(s in blocked for s in r["servers"]) or len({roles[s] for s in r["servers"]}) != 1:
                continue
            requests[r["request_key"]].update(r["servers"])
        joins += [("repeated_request", members) for members in requests.values()]
        for reason, members in joins:
            if not members:
                continue
            highest = max(PRIORITY[roles[s]] for s in members)
            for s in members:
                if PRIORITY[roles[s]] < highest:
                    blocked[s].add(reason)
        if before == set(blocked):
            break
    return {"roles": roles, "blocked": {s: sorted(v) for s, v in sorted(blocked.items())}, "rounds": rounds}


def row_role(record, owners):
    if any(s in owners["blocked"] for s in record["servers"]):
        raise Exclude("blocked_server_group")
    roles = {owners["roles"][s] for s in record["servers"]}
    if len(roles) != 1:
        raise Exclude("cross_role_composition")
    return next(iter(roles))


# Deliberately limited schema support. Rejection is preferable to silently
# ignoring unknown assertion keywords. References and formats require a later
# extension. Annotation values are preserved, never executed.
COMMON_KEYS = {"$schema", "$comment", "title", "description", "default", "examples", "readOnly", "writeOnly",
               "type", "enum", "const", "multipleOf", "maximum", "exclusiveMaximum", "minimum", "exclusiveMinimum",
               "maxLength", "minLength", "pattern", "items", "maxItems", "minItems", "uniqueItems", "contains",
               "maxProperties", "minProperties", "required", "properties", "patternProperties", "additionalProperties",
               "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "definitions", "$defs"}


@lru_cache(maxsize=8192)
def checked_validator(serialized):
    schema = parse(serialized)
    uri = schema.get("$schema", "http://json-schema.org/draft-07/schema#")
    if uri in ("http://json-schema.org/draft-07/schema#", "https://json-schema.org/draft-07/schema", "https://json-schema.org/draft-07/schema#"):
        cls = Draft7Validator
        allowed = COMMON_KEYS | {"additionalItems", "dependencies"}
    elif uri in ("https://json-schema.org/draft/2020-12/schema", "https://json-schema.org/draft/2020-12/schema#"):
        cls = Draft202012Validator
        allowed = COMMON_KEYS | {"prefixItems", "minContains", "maxContains", "dependentRequired", "dependentSchemas", "unevaluatedItems", "unevaluatedProperties"}
    else:
        raise Exclude("unsupported_schema_dialect")
    def walk(s):
        if isinstance(s, bool):
            return
        if not isinstance(s, dict):
            raise Exclude("invalid_schema")
        if any(k in s for k in ("$ref", "$dynamicRef", "$recursiveRef")):
            raise Exclude("unsupported_schema_reference")
        if set(s) - allowed:
            raise Exclude("unsupported_schema_keyword")
        if "$schema" in s and s is not schema and s["$schema"] != uri:
            raise Exclude("nested_schema_dialect")
        for k in ("properties", "patternProperties", "definitions", "$defs", "dependentSchemas"):
            for v in s.get(k, {}).values():
                walk(v)
        for k in ("allOf", "anyOf", "oneOf", "prefixItems"):
            for v in s.get(k, []):
                walk(v)
        for k in ("not", "if", "then", "else", "additionalProperties", "additionalItems", "contains", "unevaluatedItems", "unevaluatedProperties"):
            if k in s:
                walk(s[k])
        if "items" in s:
            for v in s["items"] if isinstance(s["items"], list) else [s["items"]]:
                walk(v)
        for v in s.get("dependencies", {}).values():
            if not isinstance(v, list):
                walk(v)
    try:
        cls.check_schema(schema)
        walk(schema)
    except Exclude:
        raise
    except Exception as exc:
        raise Exclude("invalid_schema") from exc
    return cls(schema)


def calls(message):
    if message.get("role") != "assistant":
        return []
    if message.get("function_call") and message.get("tool_calls"):
        raise Exclude("ambiguous_call_representation")
    if message.get("function_call"):
        return [message["function_call"]]
    return [x["function"] for x in message.get("tool_calls", [])]


def first_call(raw):
    history = parse(raw["messages"])
    users = []
    for m in history:
        if m.get("role") == "user":
            users.append(m.get("content"))
        if m.get("role") in ("tool", "function"):
            raise Exclude("orphan_preceding_response")
        chosen = calls(m)
        if chosen:
            if len(chosen) != 1:
                raise Exclude("parallel_first_calls")
            if len(users) != 1 or not isinstance(users[0], str):
                raise Exclude("unsupported_clarification_history")
            if request_key(users[0]) != request_key(raw["question"]):
                raise Exclude("question_history_mismatch")
            return users[0], chosen[0]
    raise Exclude("no_tool_call")


def render_input(state, tools):
    options = [{"id": t["name"], "description": json.dumps({k: t[k] for k in ("name", "description", "parameters")}, sort_keys=True, ensure_ascii=False)} for t in tools]
    options.sort(key=lambda o: digest({"seed": SEED, "request": request_key(state), "option": o}))
    return {"state": state, "question": QUESTION, "options": options}


def example(raw, record, owners):
    role = row_role(record, owners)
    _, remote = remote_schemas(raw)
    tools = offered_tools(raw, remote)
    validators = {t["name"]: checked_validator(json.dumps(t["parameters"], sort_keys=True)) for t in tools}
    state, call = first_call(raw)
    if call.get("name") not in validators:
        raise Exclude("undeclared_first_call")
    args = parse(call.get("arguments"))
    if not isinstance(args, dict):
        raise Exclude("nonobject_arguments")
    try:
        validators[call["name"]].validate(args)
    except Exception as exc:
        raise Exclude("argument_schema_violation") from exc
    item = render_input(state, tools)
    return {"id": digest(messages(item)), "group_id": "toucan_request:" + record["request_key"],
            "input": item, "role": role, "servers": record["servers"],
            "request_key": record["request_key"], "schema_keys": sorted({t["schema_key"] for t in tools}),
            "semantic_key": digest({"request": record["request_key"], "menu": sorted(t["schema_key"] for t in tools)}),
            "chosen_schema": next(t["schema_key"] for t in tools if t["name"] == call["name"]),
            "target": {"option_id": call["name"]}, "target_kind": "observed_first_tool_imitation",
            "lineage": [record["witness"]], "task": "toucan_first_tool", "family": "toucan"}


def consolidate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["semantic_key"]].append(r)
    retained, excluded = [], []
    for group in groups.values():
        if len({r["chosen_schema"] for r in group}) != 1:
            excluded.extend({"lineage": r["lineage"], "reason": "conflicting_teacher_choices"} for r in group)
            continue
        visible = defaultdict(list)
        for r in group:
            visible[r["id"]].append(r)
        for duplicates in visible.values():
            if len({r["role"] for r in duplicates}) != 1:
                raise ValueError("Cross-role visible collision escaped ownership")
            r = dict(duplicates[0])
            r["lineage"] = [w for d in duplicates for w in d["lineage"]]
            r["servers"] = sorted({s for d in duplicates for s in d["servers"]})
            retained.append(r)
    return sorted(retained, key=lambda r: r["id"]), excluded


def tokenize(rows, tokenizer, old_tokens=()):
    kept, excluded, groups = [], [], defaultdict(list)
    for r in rows:
        try:
            ids = encode(tokenizer, r["input"], 4096)
        except ValueError:
            excluded.append({"lineage": r["lineage"], "reason": "over_4096_tokens"})
            continue
        token_hash = digest(ids)
        if token_hash in old_tokens:
            excluded.append({"lineage": r["lineage"], "reason": "legacy_token_overlap"})
            continue
        r = {**r, "input_ids": ids, "token_sha256": token_hash, "option_ids": [o["id"] for o in r["input"]["options"]]}
        r["target_indices"] = [r["option_ids"].index(r["target"]["option_id"])]
        groups[token_hash].append(r)
    for group in groups.values():
        if len({r["role"] for r in group}) != 1 or len({tuple(r["target_indices"]) for r in group}) != 1:
            excluded.extend({"lineage": r["lineage"], "reason": "token_collision_role_or_target"} for r in group)
            continue
        r = dict(group[0])
        r["lineage"] = [w for d in group for w in d["lineage"]]
        r["servers"] = sorted({s for d in group for s in d["servers"]})
        kept.append(r)
    return sorted(kept, key=lambda r: r["id"]), excluded


def legacy_files():
    files = []
    for name in LEGACY_DIRS:
        folder = ROOT / name
        if not folder.is_dir():
            raise ValueError(f"Missing prior corpus: {name}")
        files += sorted(folder.rglob("*.jsonl"))
        files += sorted(folder.glob("*manifest*.json"))
        files += sorted(folder.glob("*usage*.json"))
    return sorted(set(files))


def legacy_index():
    requests, schemas, token_keys = set(), set(), set()
    counts = Counter()
    for path in legacy_files():
        if path.suffix != ".jsonl":
            continue
        for line in path.open():
            r = json.loads(line)
            if isinstance(r.get("input_ids"), list):
                token_keys.add(digest(r["input_ids"]))
                counts["token_rows"] += 1
            item = r.get("input")
            if not isinstance(item, dict) or not isinstance(item.get("state"), str):
                continue
            counts["visible_rows"] += 1
            requests.add(request_key(item["state"]))
            for o in item.get("options", []):
                desc = o.get("description", "")
                if "\nTool schema: " not in desc:
                    continue
                d, packed = desc.split("\nTool schema: ", 1)
                try:
                    params = json.loads(packed)["parameters"]
                except (ValueError, KeyError, TypeError):
                    counts["unparsed_legacy_schema"] += 1
                    continue
                if d.startswith("Call the offered tool: "):
                    d = d.removeprefix("Call the offered tool: ")
                name = r.get("provenance", {}).get("function_name") if r.get("family") == "glaive" else o["id"]
                if not isinstance(name, str):
                    counts["unresolved_legacy_schema_name"] += 1
                    continue
                schemas.add(schema_key(name, d, params))
    return requests, schemas, token_keys, dict(counts)


def runtime():
    return {"python": sys.version, "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def frozen_files():
    fixed = [ROOT / r for r in SOURCE_RECEIPTS]
    fixed += [ROOT / "docs/release-tool-data-v1-protocol.md", ROOT / "docs/release-candidate-v1-plan.md",
              ROOT / "scale_lab/common.py", ROOT / "results/toucan-audit-v2/comparison.json",
              ROOT / "output/telecom-questions-v1-usage/usage-private.json"]
    fixed += sorted((ROOT / "release_lab").glob("*.py"))
    fixed += [ROOT / "tests/test_release_toucan.py"]
    fixed += [ROOT / ".local/release-data-venv/lib/python3.14/site-packages/release-base.pth"]
    return sorted(set(fixed + legacy_files()))


def freeze(path):
    from huggingface_hub import hf_hub_download
    spec = MODELS["qwen35-9b"]
    cache = Path(hf_hub_download(spec["id"], "tokenizer_config.json", revision=spec["revision"], local_files_only=True)).parent
    tokenizer_files = {str(p): file_hash(p) for p in cache.iterdir() if p.name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt")}
    if not any(Path(p).name == "tokenizer_config.json" for p in tokenizer_files):
        raise ValueError("Missing tokenizer witness")
    record = {"protocol": "release-tool-data-v1", "files": {str(p.relative_to(ROOT)): file_hash(p) for p in frozen_files()},
              "sources": [s for s, _ in source_paths()], "tokenizer_files": tokenizer_files,
              "runtime": runtime(), "model": spec, "training_presentations": 0}
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return record


def verify_freeze(path):
    f = json.loads(path.read_text())
    for name, sha in f["files"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f"Frozen file changed: {name}")
    for name, sha in f["tokenizer_files"].items():
        if file_hash(name) != sha:
            raise ValueError("Frozen tokenizer changed")
    if runtime() != f["runtime"] or [s for s, _ in source_paths()] != f["sources"]:
        raise ValueError("Frozen runtime/source changed")
    return f


def build(out, freeze_path):
    from transformers import AutoTokenizer
    f = verify_freeze(freeze_path)
    out.mkdir(parents=True, exist_ok=False)
    records, rejected, raw_count = [], [], 0
    for raw, witness in raw_rows():
        raw_count += 1
        try:
            records.append(inventory(raw, witness))
        except (Exclude, ValueError, KeyError, TypeError, AttributeError) as exc:
            rejected.append({"lineage": [witness], "reason": str(exc) if isinstance(exc, Exclude) else "malformed_inventory"})
    if raw_count != 35227:
        raise ValueError("Unexpected source row census")
    old_requests, old_schemas, old_tokens, old_counts = legacy_index()
    owners = ownership(records, old_requests, old_schemas)
    write_json(out / "ownership-private.json", owners)
    by_witness = {(r["witness"]["file"], r["witness"]["row"]): r for r in records}
    candidates = []
    for raw, witness in raw_rows():
        record = by_witness.get((witness["file"], witness["row"]))
        if record is None:
            continue
        try:
            candidates.append(example(raw, record, owners))
        except (Exclude, ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
            rejected.append({"lineage": [witness], "reason": str(exc) if isinstance(exc, Exclude) else "malformed_source"})
    structural_count = len(candidates)
    deduped, failures = consolidate(candidates)
    rejected.extend(failures)
    tokenizer = AutoTokenizer.from_pretrained(f["model"]["id"], revision=f["model"]["revision"], local_files_only=True, trust_remote_code=False)
    rows, failures = tokenize(deduped, tokenizer, old_tokens)
    rejected.extend(failures)
    for role in ROLES:
        write_rows(out / f"{role}-private.jsonl", [r for r in rows if r["role"] == role])
    write_rows(out / "excluded-private.jsonl", rejected)
    summary = {"status": "candidate_pending_saved_export_audit", "source_trajectories": raw_count,
               "inventory_rows": len(records), "source_unique_requests": len({r["request_key"] for r in records}),
               "source_unique_schemas": len({s["key"] for r in records for s in r["schemas"]}),
               "source_servers": len(owners["roles"]), "blocked_servers": len(owners["blocked"]),
               "ownership_rounds": owners["rounds"], "structurally_eligible_after_ownership": structural_count,
               "questions_before_length_filter": len(deduped), "candidate_questions": len(rows),
               "candidate_lineage_rows": sum(len(r["lineage"]) for r in rows),
               "excluded_trajectories_by_reason": dict(Counter({reason: sum(len(x["lineage"]) for x in rejected if x["reason"] == reason) for reason in {x["reason"] for x in rejected}})),
               "by_role": {role: {"questions": sum(r["role"] == role for r in rows),
                                  "unique_requests": len({r["request_key"] for r in rows if r["role"] == role}),
                                  "server_groups": len({s for r in rows if r["role"] == role for s in r["servers"]}),
                                  "tokens": sum(len(r["input_ids"]) for r in rows if r["role"] == role),
                                  "within_1536": sum(len(r["input_ids"]) <= 1536 for r in rows if r["role"] == role)} for role in ROLES},
               "legacy_index": {**old_counts, "unique_requests": len(old_requests), "unique_schemas": len(old_schemas), "unique_token_inputs": len(old_tokens)},
               "training_presentations": 0, "executed_source_tools": 0,
               "interpretation": "Observed first-action imitation; no optimality, verified success or calibrated-outcome labels."}
    write_json(out / "summary.json", summary)
    manifest = {"status": "candidate", "freeze_path": str(freeze_path.resolve()), "freeze_sha256": file_hash(freeze_path),
                "model": f["model"], "max_tokens": 4096, "label_token_ids": label_token_ids(tokenizer),
                "outputs": {p.name: file_hash(p) for p in sorted(out.iterdir())},
                "rows": {r["id"]: {"sha256": digest(r), "role": r["role"]} for r in rows}, "training_presentations": 0}
    write_json(out / "manifest-private.json", manifest)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "build"))
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.mode == "freeze":
        freeze(a.freeze)
        print(json.dumps({"freeze_sha256": file_hash(a.freeze)}))
    else:
        if a.out is None:
            p.error("build requires --out")
        build(a.out, a.freeze)


if __name__ == "__main__":
    main()
