"""Conservative first-decision extraction from a pinned synthetic tool dataset.

We preserve the source's first assistant action, including follow-up questions.
This is reference imitation, not execution-verified optimal behavior. Only
single-function conversations are admitted in this first converter.
"""

import ast
from collections import Counter, defaultdict
import json
import re
import urllib.request

from scale_lab.common import ROOT, digest, file_hash, text_key

REVISION = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
SHA256 = "e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad"
URL = f"https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/{REVISION}/glaive-function-calling-v2.json"


def fetch():
    path = ROOT / "output/source-cache" / (SHA256 + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=120) as response, temp.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if file_hash(temp) != SHA256:
            raise ValueError("Glaive download checksum mismatch")
        temp.rename(path)
    if file_hash(path) != SHA256:
        raise ValueError("Glaive source checksum mismatch")
    return path


def convert(row, index):
    system = row["system"]
    if "{" not in system:
        raise ValueError("no_function")
    raw = system[system.index("{"):].strip()
    tool, end = json.JSONDecoder().raw_decode(raw)
    if raw[end:].strip():
        raise ValueError("multiple_or_trailing_function_schemas")
    if not isinstance(tool, dict) or any(not isinstance(tool.get(k), str) for k in ("name", "description")):
        raise ValueError("invalid_function")
    pieces = re.split(r"(?:^|\s)(USER|ASSISTANT|FUNCTION RESPONSE):\s*", row["chat"])
    turns = [(pieces[i], pieces[i+1].strip()) for i in range(1, len(pieces) - 1, 2)]
    if len(turns) < 2 or turns[0][0] != "USER" or turns[1][0] != "ASSISTANT":
        raise ValueError("invalid_initial_turns")
    responses = []
    for role, content in turns[1:]:
        if role != "ASSISTANT":
            break
        responses.append(content)
    assistant = "\n".join(responses)
    function_text = tool["description"] + "\nTool schema: " + json.dumps({k: v for k, v in tool.items() if k not in ("name", "description")}, sort_keys=True)
    if "<functioncall>" in assistant:
        call_text = assistant.split("<functioncall>", 1)[1].split("<|endoftext|>", 1)[0].strip()
        try:
            call = json.loads(call_text)
        except json.JSONDecodeError:
            try:
                call = ast.literal_eval(call_text)
            except (ValueError, SyntaxError):
                raise ValueError("unsupported_call_format") from None
        if not isinstance(call, dict) or call.get("name") != tool["name"]:
            raise ValueError("call_function_mismatch")
        chosen = "tool"
    elif "?" in assistant:
        chosen = "clarify"
    else:
        # A declarative "Please provide the URL." is also a follow-up, but the
        # source has no intent label. Quarantine it rather than infer certainty
        # from a missing question mark. Keep only explicit inability responses.
        if not re.search(r"\b(can't|cannot|unable|don't have|do not have|not able|not equipped|no access)\b", assistant, re.I):
            raise ValueError("unclassified_noncall_response")
        chosen = "respond"
    return {"id": f"glaive-{index:06d}", "source_id": f"glaive-{index:06d}", "family": "glaive",
            "task": "tool_or_conversation", "group_id": "glaive-name:" + text_key(tool["name"]),
            "input": {"state": turns[0][1], "question": "What should the assistant do next to address the user's request?",
                      "options": [{"id": "tool", "description": "Call the offered tool: " + function_text},
                                  {"id": "clarify", "description": "Ask the user a follow-up question without calling a tool yet."},
                                  {"id": "respond", "description": "Respond to the user without a tool call or a follow-up question."}]},
            "target": {"option_id": chosen},
            "provenance": {"dataset": "glaiveai/glaive-function-calling-v2", "upstream_revision": REVISION,
                           "snapshot_sha256": SHA256, "source_row": index, "source_turn": 0,
                           "license": "Apache-2.0", "function_name": tool["name"],
                           "label_method": "synthetic_first_reference_action; question_mark_heuristic_for_followup",
                           "human_reviewed": False}}


def append(splits, audit, cap, seed):
    rows = json.loads(fetch().read_text())
    if isinstance(rows, dict):
        rows = rows.get("data", rows)
    if not isinstance(rows, list):
        raise ValueError("Expected a list of Glaive source rows")
    accepted = []
    for index, row in enumerate(rows):
        try:
            accepted.append(convert(row, index))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            audit["glaive_rejected:" + str(error)[:100]] += 1
    # Union same function name AND repeated requests, so repeated generic requests
    # cannot silently cross the split just because a different tool was offered.
    parents = list(range(len(accepted)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for i, row in enumerate(accepted):
        for key in (row["group_id"], "request:" + text_key(row["input"]["state"])):
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(accepted):
        groups[find(i)].append(row)
    counts = Counter()
    for group in sorted(groups.values(), key=lambda g: digest([seed, min(r["id"] for r in g)])):
        gid = min(r["id"] for r in group)
        bucket = int(digest([seed, gid])[:8], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        limit = cap if split == "train" else max(200, cap // 10)
        # Huge connected components are assigned intact, then sampled internally.
        # This preserves separation without losing all coverage to a single hub.
        group.sort(key=lambda r: digest([seed, r["id"]]))
        for row in group[:max(0, limit - counts[split])]:
            row["group_id"] = "glaive:" + gid
            splits[split].append(row)
            counts[split] += 1
    audit["glaive_source_rows"] = len(rows)
    audit["glaive_accepted_before_quotas"] = len(accepted)
    audit["glaive_largest_connected_group"] = max(map(len, groups.values()), default=0)
    for split, count in counts.items():
        audit["glaive_included:" + split] = count


if __name__ == "__main__":
    print(fetch())
