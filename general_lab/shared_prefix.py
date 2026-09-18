"""Experimental state-first prompts and independent cache branches.

This is an inference experiment, not the deployed or training serializer. The
state-first layout changes token order and therefore needs a quality comparison
before adoption. Cache reuse is checked against full forwards of that *same*
layout, not against the existing options-first prompts. No weights are updated.
"""

import argparse
import copy
import json
from pathlib import Path
import time

from scale_lab.common import LABELS, digest, encode, messages, validate_input, write_json


def state_first_messages(item):
    validate_input(item)
    public = {"state": item["state"], "question": item["question"],
              "options": [{"label": LABELS[i], "description": option["description"]}
                          for i, option in enumerate(item["options"])]}
    return [{"role": "system", "content": messages(item)[0]["content"]},
            {"role": "user", "content": json.dumps(public, ensure_ascii=False)}]


def encode_state_first(tokenizer, item, max_tokens):
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("Token limit must be a positive integer")
    prompt = tokenizer.apply_chat_template(state_first_messages(item), tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens or len(tokens) > max_tokens:
        raise ValueError(f"Input has {len(tokens)} tokens; limit {max_tokens}; no truncation")
    return tokens


def prepare(tokenizer, payload, max_tokens=1536, layout="state_first"):
    from .interface import requests
    if layout not in ("state_first", "legacy"):
        raise ValueError("Unknown prompt layout")
    encoder = encode_state_first if layout == "state_first" else encode
    return [{"id": identity, "type": kind,
             "input_ids": encoder(tokenizer, item, max_tokens),
             "option_ids": [option["id"] for option in item["options"]]}
            for identity, kind, item in requests(payload)]


def common_prefix(rows):
    """Longest exact token prefix, reserving a final token in every branch."""
    if not rows:
        raise ValueError("No questions supplied")
    for row in rows:
        tokens, options = row.get("input_ids"), row.get("option_ids")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Each question requires nonempty integer token IDs")
        if (not isinstance(options, list) or not 2 <= len(options) <= len(LABELS)
                or any(not isinstance(o, str) or not o for o in options)
                or len(set(options)) != len(options)):
            raise ValueError("Each question requires distinct option IDs")
    if len(rows) == 1:
        return 0
    limit = min(len(row["input_ids"]) for row in rows) - 1
    for index in range(limit):
        if any(row["input_ids"][index] != rows[0]["input_ids"][index] for row in rows[1:]):
            return index
    return limit


def plan(rows):
    prefix = common_prefix(rows)
    total = sum(len(row["input_ids"]) for row in rows)
    forwarded = total - (len(rows) - 1) * prefix
    return {"questions": len(rows), "common_prefix_tokens": prefix,
            "independent_input_tokens": total, "cached_forward_input_tokens": forwarded,
            "avoided_input_tokens": total - forwarded,
            "avoided_fraction": (total - forwarded) / total,
            "note": "Logical input-token accounting only; cache copying and attention still cost time and memory."}


def _synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def predict_tokens(model, rows, label_ids, *, reuse_prefix):
    """Text-only, serial, unpadded inference; clone all cache state per question.

    The hybrid Qwen cache contains both attention and recurrent/convolution
    state. Reusing one mutable cache across branches would leak earlier answers.
    No cache survives this request. This function is not safe to call concurrently
    on a model instance that other code is mutating.
    """
    import torch
    accounting = plan(rows)
    if model.training:
        raise ValueError("Cache experiment requires evaluation mode")
    count = max(len(row["option_ids"]) for row in rows)
    if (not isinstance(label_ids, list) or len(label_ids) < count
            or any(type(t) is not int or t < 0 for t in label_ids)
            or len(set(label_ids)) != len(label_ids)):
        raise ValueError("Labels must be distinct nonnegative token IDs")
    device = next(model.parameters()).device
    labels = torch.tensor(label_ids[:count], device=device)
    prefix = accounting["common_prefix_tokens"] if reuse_prefix else 0
    _synchronize(device)
    started = time.perf_counter()
    results, cache = [], None
    with torch.inference_mode():
        if prefix:
            output = model(input_ids=torch.tensor([rows[0]["input_ids"][:prefix]], device=device),
                           use_cache=True, logits_to_keep=1, return_dict=True)
            cache = output.past_key_values
            if cache is None or cache.get_seq_length() != prefix:
                raise ValueError("Model did not return the complete reusable prefix cache")
            del output
        for row in rows:
            # Deep copy includes recurrent state; tensor views or shallow copies
            # are insufficient for this hybrid architecture.
            branch = copy.deepcopy(cache) if cache is not None else None
            kwargs = {"past_key_values": branch} if branch is not None else {}
            output = model(input_ids=torch.tensor([row["input_ids"][prefix:]], device=device),
                           use_cache=branch is not None, logits_to_keep=1, return_dict=True, **kwargs)
            scores = output.logits[0, -1].index_select(0, labels[:len(row["option_ids"])]).float()
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite option logits")
            probs = scores.softmax(-1).cpu().tolist()
            results.append({"id": row["id"], "probabilities": dict(zip(row["option_ids"], probs)),
                            "choice": row["option_ids"][max(range(len(probs)), key=probs.__getitem__)]})
            del output, branch
    _synchronize(device)
    return {"predictions": results, "seconds": time.perf_counter() - started,
            "prefix_tokens_used": prefix,
            "forward_input_tokens": accounting["independent_input_tokens"] - (len(rows) - 1) * prefix,
            "model_calls": len(rows) + bool(prefix), "cache_lifetime": "one request"}


def compare(reference, candidate):
    left, right = reference["predictions"], candidate["predictions"]
    if len(left) != len(right) or not left:
        raise ValueError("Prediction coverage differs")
    deltas, agreements = [], []
    for a, b in zip(left, right):
        if a["id"] != b["id"] or a["probabilities"].keys() != b["probabilities"].keys():
            raise ValueError("Question or option identity differs")
        deltas.extend(abs(a["probabilities"][key] - b["probabilities"][key]) for key in a["probabilities"])
        agreements.append(a["choice"] == b["choice"])
    return {"max_probability_delta": max(deltas), "choice_agreement": sum(agreements) / len(agreements),
            "questions": len(left)}


def main():
    from scale_lab.common import MODELS, file_hash
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--model", choices=MODELS, default="qwen35-9b")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--inference", action="store_true", help="Explicitly load a model and compare both prompt layouts")
    parser.add_argument("--probability-tolerance", type=float, default=.02)
    args = parser.parse_args()
    if not 0 < args.probability_tolerance < 1:
        parser.error("Probability tolerance must be between zero and one")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing experiment report")
    from transformers import AutoTokenizer
    spec = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"],
                                              trust_remote_code=False, token=False, local_files_only=True)
    payload = json.loads(args.input.read_text())
    rows = {layout: prepare(tokenizer, payload, args.max_tokens, layout)
            for layout in ("legacy", "state_first")}
    report = {"purpose": "supplementary prompt/cache mechanics experiment; not training or checkpoint selection",
              "model": spec, "input_sha256": file_hash(args.input), "code_sha256": file_hash(Path(__file__)),
              "model_inference": args.inference, "weights_updated": False,
              "layouts": {layout: {"plan": plan(items), "encoded_inputs_sha256": digest(items)}
                          for layout, items in rows.items()}}
    if args.inference:
        from scale_lab.infer import Predictor
        predictor = Predictor(args.model, args.run, args.max_tokens, args.device)
        if predictor.spec != spec:
            raise ValueError("Requested tokenizer and saved model differ")
        # This explicit opt-in path is supplementary only. Full forwards of the
        # new layout distinguish a prompt-order effect from a caching effect.
        legacy = predict_tokens(predictor.model, rows["legacy"], predictor.labels, reuse_prefix=False)
        uncached = predict_tokens(predictor.model, rows["state_first"], predictor.labels, reuse_prefix=False)
        cached = predict_tokens(predictor.model, rows["state_first"], predictor.labels, reuse_prefix=True)
        report["inference"] = {"legacy": legacy, "state_first_uncached": uncached,
                               "state_first_cached": cached,
                               "cache_comparison": compare(uncached, cached),
                               "prompt_order_comparison": compare(legacy, uncached),
                               "timing_note": "Single observations include warmup/order effects; not a latency benchmark."}
        report["cache_probability_tolerance"] = args.probability_tolerance
        report["cache_check_passed"] = compare(uncached, cached)["max_probability_delta"] <= args.probability_tolerance
        report["adapter_files_sha256"] = ({p.name: file_hash(p) for p in sorted((args.run / 'best').iterdir())
                                          if p.is_file()} if args.run else None)
        import torch
        import transformers
        report["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                             "device": args.device, "dtype": str(next(predictor.model.parameters()).dtype)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"model_inference": report["model_inference"],
                      "plans": {key: value["plan"] for key, value in report["layouts"].items()}}, indent=2))
    if report.get("cache_check_passed") is False:
        raise SystemExit("Cache comparison exceeded tolerance; report preserved")


if __name__ == "__main__":
    main()
