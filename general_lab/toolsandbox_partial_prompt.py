"""Read-only cached-tokenizer audit; no model loading or inference."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from general_lab import toolsandbox_partial as pilot


def audit(tokenizer_path, guards, output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TORCH="0", USE_TF="0")
    from transformers import AutoTokenizer
    from scale_lab.common import messages
    tokenizer_path = Path(tokenizer_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                                              token=False, trust_remote_code=False)
    data = json.loads(Path(guards).read_text())
    observed = {(row["root_id"], row["world"]): row["phone_history"] for row in data["rows"]}
    rows = []
    for scenario in pilot.scenarios():
        for history_name, history in (("root", []), ("name_a", observed[(scenario["id"], 0)]),
                                      ("name_b", observed[(scenario["id"], 2)])):
            public = pilot.public_input(scenario, history)
            for action in pilot.actions(public):
                for kind in ("outcome", "cost"):
                    item = pilot.question(public, action, kind)
                    count = len(item["options"])
                    row = {"root_id": scenario["id"], "split": scenario["split"], "history": history_name,
                           "action": action, "marginal": kind, "options": count,
                           "input_sha256": pilot.base.digest(item), "known_singleton": count == 1}
                    if count == 1:
                        row.update(tokens=None, within_limits=True)
                    else:
                        prompt = tokenizer.apply_chat_template(messages(item), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
                        row["tokens"] = len(tokenizer.encode(prompt, add_special_tokens=False, truncation=False))
                        row["within_limits"] = row["tokens"] <= 1536 and count <= 36
                    rows.append(row)
    used = [row for row in rows if not row["known_singleton"]]
    result = {"status": "passed" if all(r["within_limits"] for r in rows) else "not_current_training_ready",
              "model_inference_calls": 0, "tokenizer": str(tokenizer_path),
              "tokenizer_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tokenizer_path.iterdir()
                                  if p.is_file() and (p.name.startswith("tokenizer") or p.name in ("vocab.json", "merges.txt", "chat_template.jinja"))},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest(),
              "guards_sha256": hashlib.sha256(Path(guards).read_bytes()).hexdigest(),
              "limits": {"tokens": 1536, "options": 36}, "question_count": len(rows),
              "tokenized_questions": len(used), "known_singleton_costs": len(rows) - len(used),
              "max_tokens": max(r["tokens"] for r in used), "min_tokens": min(r["tokens"] for r in used),
              "max_options": max(r["options"] for r in rows),
              "overlength_questions": sum(not r["within_limits"] for r in rows),
              "overlength_by_marginal": dict(Counter(r["marginal"] for r in rows if not r["within_limits"])),
              "method": "All declared roots/histories/actions, actual cached tokenizer/chat template; no truncation, label-based filtering, or model loading.",
              "rows": rows}
    pilot.write_json(output, result)
    return {k: v for k, v in result.items() if k not in ("rows", "tokenizer_files")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--guards", type=Path, default=pilot.OUTPUT / "guards.json")
    parser.add_argument("--output", type=Path, default=pilot.OUTPUT / "prompt-audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.tokenizer, args.guards, args.output), indent=2))
