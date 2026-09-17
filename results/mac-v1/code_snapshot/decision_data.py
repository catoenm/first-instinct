"""Convert a saved ToolACE sample into inspectable, single-tool routing examples.

No model calls or tool execution. Reference labels are synthetic source labels,
not independently verified ground truth. This sample is for schema inspection.
"""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline import save_documents


ROOT = Path(__file__).resolve().parent
SNAPSHOT_SHA256 = "8657de928d9509729defe15832a1e804c6342c9b3c68837839e49c3f391d5cf5"
SNAPSHOT = ROOT / "data" / "decisions" / "raw" / f"{SNAPSHOT_SHA256}.json"
OUTPUT = ROOT / "output" / "decisions"
SOURCE_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=Team-ACE%2FToolACE"
    "&config=default&split=train&offset=0&length=100"
)
FIRST_TOOL_QUESTION = "Which available tool should be called first to begin fulfilling the user's request?"


def convert(item: dict) -> dict:
    if item.get("truncated_cells"):
        raise ValueError("truncated_source_row")
    row = item["row"]
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if marker not in row["system"]:
        raise ValueError("unknown_tool_list_format")
    tools, _ = json.JSONDecoder().raw_decode(row["system"].split(marker, 1)[1].lstrip())
    if not isinstance(tools, list) or len(tools) < 2:
        raise ValueError("fewer_than_two_candidates")
    if not all(isinstance(t, dict) and isinstance(t.get("name"), str)
               and isinstance(t.get("description"), str) for t in tools):
        raise ValueError("invalid_tool_schema")
    names = [t["name"] for t in tools]
    if len(names) != len(set(names)):
        raise ValueError("duplicate_tool_names")
    turns = row["conversations"]
    if len(turns) < 3 or [t["from"] for t in turns[:3]] != ["user", "assistant", "tool"]:
        raise ValueError("first_turn_not_a_tool_call")
    results = json.loads(turns[2]["value"])
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("not_a_single_tool_result")
    selected = results[0]["name"]
    if selected not in names:
        raise ValueError("target_missing_from_candidates")

    # Names in this dataset may contain spaces or slashes. Normalize only the
    # claimed outer function name for safe AST parsing; never execute the text.
    call = turns[1]["value"].strip()
    prefix = f"[{selected}("
    if not call.startswith(prefix):
        raise ValueError("call_result_name_mismatch")
    try:
        expression = ast.parse("[selected_tool(" + call[len(prefix):], mode="eval").body
    except SyntaxError as error:
        raise ValueError("unsupported_call_syntax") from error
    if (not isinstance(expression, ast.List) or len(expression.elts) != 1
            or not isinstance(expression.elts[0], ast.Call)
            or not isinstance(expression.elts[0].func, ast.Name)
            or expression.elts[0].func.id != "selected_tool"
            or sum(isinstance(node, ast.Call) for node in ast.walk(expression)) != 1):
        raise ValueError("not_a_single_simple_call")

    return {
        "id": f"toolace-sample-{item['row_idx']:05d}",
        "input": {"request": turns[0]["value"], "candidates": tools},
        "target": {"tool_name": selected},
        "provenance": {
            "dataset": "Team-ACE/ToolACE",
            "snapshot_sha256": SNAPSHOT_SHA256,
            "source_row": item["row_idx"],
            "source_turn": 0,
            "label_method": "synthetic_reference_call_crosschecked_with_synthetic_tool_result",
            "human_reviewed": False,
        },
    }


def main() -> None:
    content = SNAPSHOT.read_bytes()
    if hashlib.sha256(content).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("Source snapshot checksum mismatch")
    rows = json.loads(content)["rows"]
    examples, decisions = [], []
    for item in rows:
        try:
            example = convert(item)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            decisions.append({"source_row": item["row_idx"], "status": "excluded", "reason": str(error)})
        else:
            examples.append(example)
            decisions.append({"source_row": item["row_idx"], "status": "included"})

    save_documents(examples, OUTPUT / "examples.jsonl")
    save_documents([as_question(example) for example in examples], OUTPUT / "questions.jsonl")
    save_documents(decisions, OUTPUT / "decisions.jsonl")
    manifest = {
        "source_dataset": "https://huggingface.co/datasets/Team-ACE/ToolACE",
        "source_url": SOURCE_URL,
        "source_license": "Apache-2.0 (as declared by source dataset card)",
        "source_snapshot_sha256": SNAPSHOT_SHA256,
        "source_selection": "first 100 train rows, not a random sample",
        "upstream_revision": None,
        "revision_note": "Viewer response is pinned by its bytes; upstream commit is not recorded.",
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "included_rows": len(examples),
        "excluded_by_reason": dict(Counter(d["reason"] for d in decisions if d["status"] == "excluded")),
        "outputs": {name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
                    for name in ("examples.jsonl", "questions.jsonl", "decisions.jsonl")},
        "intended_use": "schema inspection; no train/test split and no performance claims",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Converted {len(examples)} / {len(rows)} source rows.")
    print("Excluded:", manifest["excluded_by_reason"])
    print(f"Inspect {OUTPUT / 'examples.jsonl'}")


def as_question(example: dict) -> dict:
    """Describe options in text; keep tool identifiers and labels out of that text."""
    options = []
    for tool in example["input"]["candidates"]:
        schema = {key: value for key, value in tool.items() if key not in {"name", "description"}}
        description = tool["description"]
        if schema:
            description += "\nTool schema: " + json.dumps(schema, sort_keys=True, ensure_ascii=False)
        options.append({"id": tool["name"], "description": description})
    return {
        "id": example["id"],
        "input": {
            "state": example["input"]["request"],
            "question": FIRST_TOOL_QUESTION,
            "options": options,
        },
        "target": {"option_id": example["target"]["tool_name"]},
        "provenance": example["provenance"],
    }


if __name__ == "__main__":
    main()
