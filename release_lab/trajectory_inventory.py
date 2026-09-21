"""Inventory later tool choices from existing training-owned TOUCAN witnesses.

This writes private candidates, not admitted training data or outcome labels.
No model, external tool, evaluator score or new dataset download is used.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

from release_lab import toucan as t
from scale_lab.common import digest, encode, file_hash, messages, write_json

QUESTION = "Given the user's request and the tool results observed so far, which available tool should be called next?"
PROTOCOL = "docs/trajectory-inventory-v1-protocol.md"


def later_choices(raw):
    """Require serial, linked, schema-valid calls; hide current/future actions.

    Assistant prose is omitted, including reasoning and final answers. Previous
    executed arguments and responses are public history. A current target's
    arguments and response never enter its own input.
    """
    _, remote = t.remote_schemas(raw)
    tools = t.offered_tools(raw, remote)
    validators = {x["name"]: t.checked_validator(json.dumps(x["parameters"], sort_keys=True)) for x in tools}
    history, systems, candidates = [], [], []
    user = None
    pending = None
    call_ids = set()
    for index, message in enumerate(t.parse(raw["messages"])):
        role = message.get("role")
        if role == "system":
            if user is not None or not isinstance(message.get("content"), str):
                raise t.Exclude("late_or_invalid_system_message")
            systems.append(message["content"])
        elif role == "user":
            if user is not None or not isinstance(message.get("content"), str):
                raise t.Exclude("unsupported_clarification_history")
            user = message["content"]
            if t.request_key(user) != t.request_key(raw["question"]):
                raise t.Exclude("question_history_mismatch")
        elif role == "assistant":
            chosen = t.calls(message)
            if not chosen:
                continue
            if user is None or len(chosen) != 1 or pending is not None:
                raise t.Exclude("nonserial_or_pre_user_call")
            call = chosen[0]
            name = call.get("name")
            if name not in validators:
                raise t.Exclude("undeclared_call")
            args = t.parse(call.get("arguments"))
            if not isinstance(args, dict):
                raise t.Exclude("nonobject_arguments")
            if not validators[name].is_valid(args):
                raise t.Exclude("argument_schema_violation")
            modern = bool(message.get("tool_calls"))
            call_id = message["tool_calls"][0].get("id") if modern else None
            if modern:
                if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                    raise t.Exclude("invalid_call_identity")
                call_ids.add(call_id)
            candidate = None
            if history:
                state = json.dumps(dict(request=user, source_system_messages=systems,
                                        observed_tool_history=history), sort_keys=True, ensure_ascii=False)
                if len(state.encode()) > 256*1024:
                    raise t.Exclude("public_history_over_256k_bytes")
                item = t.render_input(state, tools)
                item["question"] = QUESTION
                candidate = dict(input=item, target={"option_id": name}, message_index=index,
                                 preceding_tool_calls=len(history), prefix_sha256=digest(history))
            pending = dict(name=name, arguments=args, call_id=call_id, modern=modern, candidate=candidate)
        elif role in ("function", "tool"):
            if pending is None:
                raise t.Exclude("orphan_response")
            if pending["modern"]:
                if role != "tool" or message.get("tool_call_id") != pending["call_id"]:
                    raise t.Exclude("response_identity_mismatch")
            elif role != "function" or message.get("name") != pending["name"]:
                raise t.Exclude("response_identity_mismatch")
            if not isinstance(message.get("content"), (str, dict, list)):
                raise t.Exclude("unsupported_response_content")
            event = dict(tool=pending["name"], arguments=pending["arguments"], response=message["content"])
            if pending["candidate"] is not None:
                candidate = pending["candidate"]
                candidate["response_sha256"] = digest(message)
                candidates.append(candidate)
            history.append(copy.deepcopy(event))
            pending = None
        else:
            raise t.Exclude("unsupported_message_role")
    if pending is not None:
        raise t.Exclude("unanswered_final_call")
    if user is None:
        raise t.Exclude("missing_user")
    return candidates


def inventory(source, output):
    from transformers import AutoTokenizer
    manifest = json.loads((source / "manifest-private.json").read_text())
    admission = json.loads((source / "admission.json").read_text())
    if admission["status"] != "qualified_supervised_behavior_only" or (
            admission["manifest_sha256"] != file_hash(source / "manifest-private.json")):
        raise ValueError("Original tool admission changed")
    for name in ("train-private.jsonl", "ownership-private.json"):
        if file_hash(source / name) != manifest["outputs"][name]:
            raise ValueError("Original training ownership/data changed: " + name)
    owners = json.loads((source / "ownership-private.json").read_text())
    allowed = {}
    old_ids = set()
    for line in (source / "train-private.jsonl").open():
        row = json.loads(line)
        if row["role"] != "train":
            raise ValueError("Nontraining source")
        old_ids.add(row["token_sha256"])
        for witness in row["lineage"]:
            key = (witness["file"], witness["row"])
            if key in allowed:
                raise ValueError("Repeated admitted witness")
            allowed[key] = dict(witness=witness, group_id=row["group_id"], request_key=row["request_key"],
                                servers=row["servers"], first_question_id=row["id"])
    sources = [spec for spec, _ in t.source_paths()]
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["id"], revision=manifest["model"]["revision"], local_files_only=True)
    start = time.monotonic()
    rejected = Counter()
    summary = dict(status="inventory_running", admitted_source_trajectories=len(allowed), inspected_trajectories=0,
                   trajectories_with_later_candidates=0, candidate_presentations=0, input_tokens=0,
                   source_ownership_sha256=file_hash(source/"ownership-private.json"),
                   training_presentations=0, optimizer_steps=0, new_executed_branches=0, new_underlying_tasks=0)
    hashes = {"code": file_hash(Path(__file__)), "protocol": file_hash(t.ROOT/PROTOCOL),
              "original_admission": file_hash(source/"admission.json"), "original_manifest": file_hash(source/"manifest-private.json")}
    write_json(output/"plan.json", dict(version="trajectory-inventory-v1", hashes=hashes, sources=sources,
                max_tokens=4096, max_history_bytes=256*1024, role="train_only_candidates", model=manifest["model"]))
    seen, requests, servers, depths = {}, set(), set(), Counter()
    inspected = set()
    with (output/"candidates-private.jsonl").open("x") as out, (output/"exclusions-private.jsonl").open("x") as exclusions:
        for raw, witness in t.raw_rows():
            key = (witness["file"], witness["row"])
            if key not in allowed:
                continue
            parent = allowed[key]
            if witness != parent["witness"] or key in inspected:
                raise ValueError("Admitted source witness changed or repeated")
            inspected.add(key)
            summary["inspected_trajectories"] += 1
            record = t.inventory(raw, witness)
            if t.row_role(record, owners) != "train" or record["request_key"] != parent["request_key"]:
                raise ValueError("Ownership escaped the original training component")
            try:
                candidates = later_choices(raw)
            except (t.Exclude, KeyError, TypeError, ValueError, AttributeError) as exc:
                reason = str(exc) if isinstance(exc, t.Exclude) else type(exc).__name__
                rejected[reason] += 1
                exclusions.write(json.dumps(dict(witness=witness, reason=reason))+'\n')
                continue
            kept = 0
            for candidate in candidates:
                try:
                    ids = encode(tokenizer, candidate["input"], 4096)
                except ValueError:
                    rejected["later_question_over_4096_tokens"] += 1
                    continue
                token_hash = digest(ids)
                if token_hash in old_ids:
                    raise ValueError("Later question duplicates original first-action input")
                item = candidate["input"]
                option_ids = [o["id"] for o in item["options"]]
                target = option_ids.index(candidate["target"]["option_id"])
                targets = seen.setdefault(token_hash, set())
                targets.add(target)
                row = dict(candidate, id=digest(messages(item)), input_ids=ids, option_ids=option_ids,
                           target_indices=[target], token_sha256=token_hash, role="train_candidate",
                           group_id=parent["group_id"], request_key=parent["request_key"], servers=parent["servers"],
                           first_question_id=parent["first_question_id"], lineage=[witness],
                           target_kind="observed_next_tool_imitation", admitted=False)
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
                summary["candidate_presentations"] += 1
                summary["input_tokens"] += len(ids)
                depths[candidate["preceding_tool_calls"]] += 1
                requests.add(parent["request_key"])
                servers.update(parent["servers"])
                kept += 1
            summary["trajectories_with_later_candidates"] += kept > 0
            if summary["inspected_trajectories"] % 250 == 0:
                write_json(output/"progress.json", summary)
    if len(inspected) != len(allowed):
        raise ValueError("Missing admitted source witnesses")
    summary.update(status="inventory_complete_not_admitted", candidate_unique_token_inputs=len(seen),
                   conflicting_token_inputs=sum(len(v)>1 for v in seen.values()),
                   distinct_normalized_requests=len(requests), source_server_groups=len(servers),
                   preceding_calls=dict(sorted(depths.items())), exclusions=dict(sorted(rejected.items())),
                   elapsed_seconds=time.monotonic()-start,
                   candidate_sha256=file_hash(output/"candidates-private.jsonl"),
                   limitations=["Teacher actions are not verified optimal actions or outcome rewards.",
                                "Source responses are recorded feedback, not independently replayed outcomes.",
                                "No new families or independent tasks are created by later prefixes.",
                                "Independent source reconstruction, conflict filtering and full ownership audit remain required before training.",
                                "System prompts are retained as source data; assistant reasoning and future messages are absent."])
    write_json(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/release-tool-data-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
