"""Apply the exact same joint decision interface to a base model or saved adapter."""

import argparse
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from scale_lab.common import MODELS, encode, label_token_ids
from scale_lab.model import device_name, evaluate, load_model


class Predictor:
    """Keep one model loaded for repeated, serialized decision requests."""

    def __init__(self, model_alias="qwen35-4b", run=None, max_tokens=2048, device="auto"):
        if max_tokens <= 0:
            raise ValueError("Token limit must be positive")
        self.spec, self.run = MODELS[model_alias], Path(run) if run else None
        self.selected_step = None
        if self.run:
            receipt = json.loads((self.run / "run.json").read_text())
            self.spec = receipt["model"]
            self.selected_step = receipt.get("best_step")
        self.max_tokens, self.device = max_tokens, device_name(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["id"], revision=self.spec["revision"], trust_remote_code=False, token=False)
        self.labels = label_token_ids(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.model = load_model(self.spec, self.device, self.run / "best" if self.run else None)

    def validate(self, item):
        if not isinstance(item, dict):
            raise ValueError("Request must be an object with state, question and options")
        return encode(self.tokenizer, item, self.max_tokens)

    def predict(self, item):
        started = time.perf_counter()
        ids = self.validate(item)
        row = {"id": "request", "group_id": "request", "task": "request", "input_ids": ids,
               "option_ids": [o["id"] for o in item["options"]], "target_indices": []}
        result = evaluate(self.model, [row], self.labels, self.pad, self.device, 1)[0]
        return {"choice": result["choice"], "probabilities": result["probabilities"],
                "milliseconds": (time.perf_counter() - started) * 1000, "input_tokens": len(ids),
                "note": "Conditional option probabilities; not calibrated probabilities of success."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, default="qwen35-4b")
    p.add_argument("--run", type=Path)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = p.parse_args()
    item = json.loads(args.input.read_text())
    predictor = Predictor(args.model, args.run, args.max_tokens, args.device)
    print(json.dumps(predictor.predict(item), indent=2))


if __name__ == "__main__":
    main()
