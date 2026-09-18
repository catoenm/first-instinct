"""Check base-model padding mechanics using training inputs only.

Compare eight independently selected single-row forwards against two mixed
batches: ordinary left padding and padding rounded to 64. Dropout is off, no
labels are used, and no weights are updated. This is not an accuracy evaluation.
Exit nonzero if any absolute option-probability change exceeds 0.02.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from scale_lab.common import digest, file_hash, write_json
from scale_lab.model import batch, load_model, score

ROW_COUNT = 8
MAX_INPUT_TOKENS = 700
SCAN_LIMIT = 8192
PROBABILITY_TOLERANCE = .02


def select_rows(path):
    """Take length quantiles of a bounded training-only prefix, without labels."""
    eligible = []
    scanned = 0
    with Path(path).open() as stream:
        for line in stream:
            if scanned >= SCAN_LIMIT:
                break
            row = json.loads(line)
            scanned += 1
            tokens = row["input_ids"]
            if 0 < len(tokens) <= MAX_INPUT_TOKENS:
                # batch() constructs a target mask, which this check ignores.
                # Supply a dummy index instead of consuming a gold target.
                eligible.append({"id": row["id"], "input_ids": tokens,
                                 "option_ids": row["option_ids"], "target_indices": [0]})
    eligible.sort(key=lambda row: (len(row["input_ids"]), row["id"]))
    # Ensure rounding actually introduces padding; otherwise this portion of
    # the check could pass without exercising a different padded shape.
    nonmultiple = [len(row["input_ids"]) for row in eligible if len(row["input_ids"]) % 64]
    if not nonmultiple:
        raise ValueError("Training prefix has no input that exercises padding to 64")
    maximum = max(nonmultiple)
    eligible = [row for row in eligible if len(row["input_ids"]) <= maximum]
    if len(eligible) < ROW_COUNT:
        raise ValueError(f"Need at least {ROW_COUNT} eligible training rows in the first {SCAN_LIMIT}")
    indices = [round(i * (len(eligible) - 1) / (ROW_COUNT - 1)) for i in range(ROW_COUNT)]
    selected = [eligible[i] for i in indices]
    if len({row["id"] for row in selected}) != ROW_COUNT:
        raise ValueError("Selected training rows have duplicate identifiers")
    if len({len(row["input_ids"]) for row in selected}) < 2:
        raise ValueError("Need different sequence lengths to exercise mixed-batch masks")
    for row in selected:
        options = row["option_ids"]
        if not 2 <= len(options) <= 36 or len(set(options)) != len(options):
            raise ValueError("Invalid prepared option identifiers")
    return selected, {"method": "eight length quantiles of a bounded training-only prefix",
                      "rows_scanned": scanned, "scan_limit": SCAN_LIMIT,
                      "eligible_rows": len(eligible), "maximum_input_tokens": MAX_INPUT_TOKENS,
                      "selection_uses_targets_or_model_predictions": False}


@torch.inference_mode()
def capture(model, rows, label_ids, pad_id, device, pad_multiple):
    inputs, labels, option_mask, _ = batch(rows, label_ids, pad_id, device, pad_multiple)
    # Check that the actual masked input still has exactly the selected tokens.
    for index, row in enumerate(rows):
        real = inputs["input_ids"][index][inputs["attention_mask"][index].bool()].cpu().tolist()
        if real != row["input_ids"]:
            raise ValueError("Padding changed the non-padding input tokens")
    logits = score(model, inputs, labels, option_mask)
    probabilities = logits.softmax(-1)
    if not torch.isfinite(logits[option_mask]).all() or not torch.isfinite(probabilities).all():
        raise ValueError("Nonfinite option scores in padding check")
    logits, probabilities = logits.cpu().tolist(), probabilities.cpu().tolist()
    outputs = []
    for row, values, probs in zip(rows, logits, probabilities):
        count = len(row["option_ids"])
        values, probs = values[:count], probs[:count]
        outputs.append({"id": row["id"], "option_ids": row["option_ids"],
                        "logits": values, "probabilities": probs,
                        "choice": row["option_ids"][max(range(count), key=probs.__getitem__)]})
    return outputs, inputs["input_ids"].shape[-1]


def compare(reference, actual):
    if len(reference) != len(actual):
        raise ValueError("Comparison row counts differ")
    details = []
    for before, after in zip(reference, actual):
        if before["id"] != after["id"] or before["option_ids"] != after["option_ids"]:
            raise ValueError("Comparison changed row identity or the offered choices")
        details.append({"id": before["id"], "same_choice_set_and_order": True,
                        "max_probability_delta": max(abs(a - b) for a, b in zip(before["probabilities"], after["probabilities"])),
                        "max_logit_delta": max(abs(a - b) for a, b in zip(before["logits"], after["logits"])),
                        "choice_agrees": before["choice"] == after["choice"]})
    return {"rows": len(details), "same_choice_sets_and_order": True,
            "max_probability_delta": max(row["max_probability_delta"] for row in details),
            "max_logit_delta": max(row["max_logit_delta"] for row in details),
            "choice_agreement": sum(row["choice_agrees"] for row in details) / len(details),
            "per_row": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared training-data directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_path = args.data / "train.jsonl"
    train_hash = file_hash(train_path)
    if train_hash != manifest["outputs"]["train.jsonl"]:
        raise ValueError("Training-data checksum mismatch")
    rows, selection = select_rows(train_path)
    spec = manifest["model"]
    report = {"status": "loading", "purpose": "mechanical padding/attention-mask equivalence; no accuracy or calibration claim",
              "model": spec, "device": args.device, "weights_updated": False, "uses_training_rows_only": True,
              "gold_targets_used": False, "data_manifest_sha256": file_hash(manifest_path),
              "train_jsonl_sha256": train_hash, "code_sha256": file_hash(Path(__file__)),
              "selection": selection, "probability_tolerance": PROBABILITY_TOLERANCE,
              "pass_rule": "Every absolute probability delta must be <= 0.02; choice ties may flip within tolerance.",
              "inputs": [{"id": row["id"], "tokens": len(row["input_ids"]), "option_ids": row["option_ids"],
                          "input_sha256": digest({"input_ids": row["input_ids"], "option_ids": row["option_ids"]})}
                         for row in rows]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    try:
        torch.set_float32_matmul_precision("high")
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        labels = manifest["label_token_ids"]
        model = load_model(spec, args.device, training=False)
        model.eval()
        report["dtype"] = str(next(model.parameters()).dtype)
        report["model_training"] = model.training
        outputs, lengths = {}, {}
        outputs["mixed_pad1"], lengths["mixed_pad1"] = capture(model, rows, labels, pad, args.device, 1)
        outputs["mixed_pad64"], lengths["mixed_pad64"] = capture(model, rows, labels, pad, args.device, 64)
        singles, single_lengths = [], []
        for row in rows:
            output, length = capture(model, [row], labels, pad, args.device, 1)
            singles.extend(output)
            single_lengths.append(length)
        outputs["single_pad1"], lengths["single_pad1"] = singles, single_lengths
        comparisons = {"mixed_pad1_vs_mixed_pad64": compare(outputs["mixed_pad1"], outputs["mixed_pad64"]),
                       "single_vs_mixed_pad1": compare(singles, outputs["mixed_pad1"]),
                       "single_vs_mixed_pad64": compare(singles, outputs["mixed_pad64"])}
        maximum = max(value["max_probability_delta"] for value in comparisons.values())
        report.update(status="passed" if maximum <= PROBABILITY_TOLERANCE else "failed_tolerance",
                      comparisons=comparisons, max_probability_delta=maximum,
                      max_logit_delta=max(value["max_logit_delta"] for value in comparisons.values()),
                      all_choices_agree=all(value["choice_agreement"] == 1 for value in comparisons.values()),
                      same_choice_sets_and_order=True, padded_sequence_lengths=lengths, outputs=outputs,
                      seconds=time.monotonic() - started)
        write_json(args.output, report)
    except BaseException as error:
        report.update(status="error", error=type(error).__name__, error_message=str(error), seconds=time.monotonic() - started)
        write_json(args.output, report)
        raise
    print(json.dumps({key: report[key] for key in ("status", "max_probability_delta", "max_logit_delta", "all_choices_agree", "seconds")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
