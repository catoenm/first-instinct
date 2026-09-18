"""Joint option scoring using a pretrained model's single-label token logits.

All described options enter one context. We read the last-position logits for
the permitted label tokens, without generating an explanation or decoding text.
"""

import math
import torch
from torch.nn import functional as F


def device_name(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def float_output_head(model):
    """Keep native vocabulary weights, with finer precision for probabilities.

    Casting logits after a BF16 projection cannot undo its rounding. Cast the
    frozen, untied vocabulary projection and its input before multiplication;
    gradients still propagate into language features and their adapters.
    """
    head = model.get_output_embeddings()
    if head.weight.dtype == torch.float32:
        return
    if head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        raise ValueError("Output precision conversion requires an untied vocabulary head")
    head.float()
    head.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(), *inputs[1:]))


def load_model(spec, device, adapter=None, training=False):
    from transformers import AutoModelForCausalLM, Qwen3_5ForConditionalGeneration
    cls = Qwen3_5ForConditionalGeneration if spec["kind"] == "qwen3_5" else AutoModelForCausalLM
    # Retain the supported checkpoint layout; vision weights are unused and frozen.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, loading_info = cls.from_pretrained(spec["id"], revision=spec["revision"],
        trust_remote_code=False, use_safetensors=True, token=False, dtype=dtype,
        attn_implementation="sdpa", output_loading_info=True)
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise ValueError(f"Incomplete checkpoint load: {loading_info}")
    model.to(device)
    model.config.use_cache = False
    model.requires_grad_(False)
    float_output_head(model)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    elif training:
        from peft import LoraConfig, get_peft_model
        # Match only language modules; qwen3.5 linear-attention and full-attention
        # projections are both adapted. Feed-forward layers are included too.
        names = [name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.Linear) and "visual" not in name and name != "lm_head"]
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                                target_modules=names, task_type="CAUSAL_LM"))
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train(training)
    return model


def batch(rows, label_ids, pad_id, device, pad_to_multiple=1):
    if pad_to_multiple <= 0:
        raise ValueError("Padding multiple must be positive")
    length = max(len(r["input_ids"]) for r in rows)
    length = math.ceil(length / pad_to_multiple) * pad_to_multiple
    option_count = max(len(r["option_ids"]) for r in rows)
    inputs, attention, options, valid = [], [], [], []
    for row in rows:
        n = len(row["input_ids"])
        # Left padding ensures logits_to_keep=1 selects a real last token.
        inputs.append([pad_id] * (length - n) + row["input_ids"])
        attention.append([0] * (length - n) + [1] * n)
        options.append([i < len(row["option_ids"]) for i in range(option_count)])
        valid.append([i in row["target_indices"] for i in range(option_count)])
    return {"input_ids": torch.tensor(inputs, device=device),
            "attention_mask": torch.tensor(attention, device=device)}, \
           torch.tensor(label_ids[:option_count], device=device), \
           torch.tensor(options, device=device), torch.tensor(valid, device=device)


def score(model, inputs, label_ids, option_mask):
    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    return logits.index_select(-1, label_ids).float().masked_fill(~option_mask, -torch.inf)


def loss_for(logits, acceptable):
    if not acceptable.any(dim=-1).all():
        raise ValueError("Each example must have an acceptable option")
    # Optimize total mass assigned to any acceptable answer. Do not arbitrarily
    # penalize the second action when both executions achieve the same result.
    losses = torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~acceptable, -torch.inf), -1)
    return losses.mean()


@torch.no_grad()
def evaluate(model, rows, label_ids, pad_id, device, batch_size=4, pad_to_multiple=1):
    was_training = model.training
    model.eval()
    predictions = []
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            inputs, labels, mask, valid = batch(chunk, label_ids, pad_id, device, pad_to_multiple)
            probabilities = score(model, inputs, labels, mask).softmax(-1).cpu().tolist()
            for row, probs in zip(chunk, probabilities):
                probs = probs[:len(row["option_ids"])]
                predictions.append({"id": row["id"], "group_id": row["group_id"], "task": row["task"],
                                    "choice": row["option_ids"][max(range(len(probs)), key=probs.__getitem__)],
                                    "probabilities": dict(zip(row["option_ids"], probs)),
                                    "target_ids": [row["option_ids"][i] for i in row["target_indices"]]})
    finally:
        model.train(was_training)
    return predictions
