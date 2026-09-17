"""Apply the exact same joint decision interface to a base model or saved adapter."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from scale_lab.common import MODELS, encode, label_token_ids
from scale_lab.model import device_name, evaluate, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, default="qwen35-4b")
    p.add_argument("--run", type=Path)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = p.parse_args()
    item = json.loads(args.input.read_text())
    spec = MODELS[args.model]
    if args.run:
        run = json.loads((args.run / "run.json").read_text())
        spec = run["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    ids = encode(tokenizer, item, args.max_tokens)
    model = load_model(spec, device_name(args.device), args.run / "best" if args.run else None)
    row = {"id": "request", "group_id": "request", "task": "request", "input_ids": ids,
           "option_ids": [o["id"] for o in item["options"]], "target_indices": []}
    result = evaluate(model, [row], label_token_ids(tokenizer), tokenizer.pad_token_id or tokenizer.eos_token_id, device_name(args.device), 1)[0]
    print(json.dumps({"choice": result["choice"], "probabilities": result["probabilities"],
                      "note": "Conditional option probabilities; not calibrated probabilities of success."}, indent=2))


if __name__ == "__main__":
    main()
