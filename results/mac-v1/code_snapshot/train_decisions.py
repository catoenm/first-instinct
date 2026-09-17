"""Repeatedly train on one tiny batch: a learning-mechanics exercise, not an evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from decision_model import (
    MODEL_ID, MODEL_REVISION, OptionScorer, choose_device, encode_options,
    load_encoder, padded_features, tokenize_options,
)


ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurements(scores, targets, rows):
    probabilities = scores.softmax(dim=-1)
    predictions = []
    for i, row in enumerate(rows):
        options = row["input"]["options"]
        predictions.append({
            "id": row["id"],
            "target": row["target"]["option_id"],
            "target_probability": probabilities[i, targets[i]].item(),
            "choice": options[scores[i].argmax().item()]["id"],
            "scores": scores[i, :len(options)].tolist(),
            "probabilities": {option["id"]: probabilities[i, j].item()
                              for j, option in enumerate(options)},
        })
    return {
        "loss": F.cross_entropy(scores, targets).item(),
        "correct": (scores.argmax(dim=-1) == targets).sum().item(),
        "count": len(rows),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "output/decisions/questions.jsonl")
    parser.add_argument("--output", type=Path, help="A new directory; existing runs are never overwritten")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    if min(args.examples, args.steps, args.max_tokens) < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("examples, steps, max-tokens, and learning-rate must be positive and finite")
    if args.max_tokens > 512:
        parser.error("This exercise limits the selected encoder to 512 tokens per option input")
    run = args.output or ROOT / "output/decision_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    device = choose_device(args.device)
    print(f"Loading {MODEL_ID} on {device}; training the scoring layer on cpu.", flush=True)
    tokenizer, encoder = load_encoder(device)

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    rng.shuffle(rows)
    selected, encodings, selection_log = [], [], []
    for row in rows:
        rng.shuffle(row["input"]["options"])
        target = row["target"]["option_id"]
        if target not in [option["id"] for option in row["input"]["options"]]:
            raise ValueError(f"{row['id']}: target is missing from options")
        try:
            tokens = tokenize_options(tokenizer, row["input"], args.max_tokens)
        except ValueError as error:
            selection_log.append({"id": row["id"], "status": "excluded", "reason": str(error)})
            continue
        chosen = len(selected) < args.examples
        selection_log.append({"id": row["id"], "status": "selected" if chosen else "unused",
                              "token_counts": list(map(len, tokens["input_ids"]))})
        if chosen:
            selected.append(row)
            encodings.append(tokens)
    save_json(run / "selection.json", selection_log)
    if len(selected) < args.examples:
        raise ValueError(f"Only {len(selected)} eligible examples; requested {args.examples}. See selection.json.")
    save_json(run / "training_examples.json", selected)
    save_json(run / "example_input.json", selected[0]["input"])
    save_json(run / "token_inspection.json", {
        "id": selected[0]["id"],
        "note": "These are actual token IDs and token pieces for every option of the first example.",
        "options": [{"option_id": option["id"], "token_ids": ids,
                     "token_pieces": tokenizer.convert_ids_to_tokens(ids)}
                    for option, ids in zip(selected[0]["input"]["options"], encodings[0]["input_ids"])],
    })

    # The encoder is frozen and deterministic, so compute each representation once.
    feature_rows = []
    for index, tokens in enumerate(encodings):
        feature_rows.append(encode_options(encoder, tokenizer, tokens, device))
        print(f"Encoded example {index + 1}/{len(selected)} ({len(tokens['input_ids'])} options).", flush=True)
    features, mask = padded_features(feature_rows)
    targets = torch.tensor([[option["id"] for option in row["input"]["options"]].index(
        row["target"]["option_id"]) for row in selected])
    scorer = OptionScorer(features.shape[-1])
    with torch.no_grad():
        before = measurements(scorer(features, mask), targets, selected)
    print(f"Step 0: loss={before['loss']:.4f} (uniform probabilities)", flush=True)

    # This is the complete training loop. No trainer framework hides the update.
    trace = []
    first_update = None
    training_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        scorer.zero_grad()
        scores = scorer(features, mask)
        loss = F.cross_entropy(scores, targets)  # Mean -log(probability of target).
        loss.backward()  # Derivatives: how changing each weight changes the loss.
        gradient = scorer.score.weight.grad
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite loss/gradient; reduce the learning rate")
        old_weights = scorer.score.weight.detach().clone() if step == 1 else None
        with torch.no_grad():
            scorer.score.weight -= args.learning_rate * gradient
            updated = scorer(features, mask)
            updated_loss = F.cross_entropy(updated, targets).item()
            if not torch.isfinite(updated).masked_select(mask).all():
                raise FloatingPointError("Non-finite scores after update")
            trace.append({
                "step": step, "loss_before": loss.item(), "loss_after": updated_loss,
                "gradient_norm": gradient.norm().item(),
                "target_probabilities_after": updated.softmax(-1).gather(1, targets[:, None]).flatten().tolist(),
            })
            if step == 1:
                first_update = {"weights_before": old_weights.flatten().tolist(),
                                "gradient": gradient.flatten().tolist(),
                                "weights_after": scorer.score.weight.flatten().tolist()}
        if step in {1, 10, 50, 100, args.steps}:
            print(f"Step {step}: loss={updated_loss:.4f}, gradient norm={gradient.norm().item():.4f}", flush=True)
    training_seconds = time.perf_counter() - training_started
    with torch.no_grad():
        final_scores = scorer(features, mask)
        after = measurements(final_scores, targets, selected)

    # Check that the saved learned weights reproduce the in-memory model.
    save_file({key: value.detach().contiguous() for key, value in scorer.state_dict().items()},
              run / "scorer.safetensors")
    restored = OptionScorer(features.shape[-1])
    restored.load_state_dict(load_file(run / "scorer.safetensors"))
    torch.testing.assert_close(restored(features, mask), final_scores, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in encoder.parameters())
    save_json(run / "before_after.json", {"before": before, "after": after})
    save_json(run / "first_update.json", first_update)
    (run / "training_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    manifest = {
        "experiment": "memorize_one_tiny_batch_with_frozen_encoder",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "encoder_device": device, "scorer_device": "cpu", "encoder_frozen": True,
        "seed": args.seed, "steps": args.steps, "learning_rate": args.learning_rate,
        "max_tokens": args.max_tokens, "selection": "seeded shuffle; first eligible examples; options shuffled",
        "examples": len(selected), "training_data_sha256": checksum(args.data),
        "versions": {package: importlib.metadata.version(package) for package in
                     ("torch", "transformers", "tokenizers", "sentencepiece", "safetensors", "huggingface-hub")},
        "python": platform.python_version(), "platform": platform.platform(),
        "code_sha256": {name: checksum(ROOT / name) for name in
                        ("decision_model.py", "train_decisions.py", "decision_data.py")},
        "training_seconds": training_seconds, "elapsed_seconds": time.perf_counter() - started,
        "before_loss": before["loss"], "after_loss": after["loss"],
        "after_training_correct": after["correct"],
        "checkpoint_roundtrip_exact": True,
        "limitations": "Same examples used for training and inspection. No held-out evaluation or calibration. Synthetic reference labels.",
        "artifacts_sha256": {path.name: checksum(path) for path in sorted(run.iterdir()) if path.is_file()},
    }
    save_json(run / "manifest.json", manifest)
    print(f"\nTraining examples correct: {after['correct']}/{after['count']}; this is memorization, not held-out accuracy.")
    for old, new in zip(before["predictions"], after["predictions"]):
        print(f"  {old['id']}: target probability {old['target_probability']:.1%} -> {new['target_probability']:.1%}")
    print(f"Run saved to: {run}")


if __name__ == "__main__":
    main()
