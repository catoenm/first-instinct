"""A text encoder plus one learned score per described answer option."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file


MODEL_ID = "microsoft/deberta-v3-small"
MODEL_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"


def text_pairs(item: dict) -> list[tuple[str, str]]:
    """Only state, question, and descriptions reach the encoder. Never the label."""
    for field in ("state", "question"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    options = item.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("At least two options are required")
    identifiers = []
    for option in options:
        if not isinstance(option, dict) or any(
            not isinstance(option.get(key), str) or not option[key].strip()
            for key in ("id", "description")
        ):
            raise ValueError("Every option needs a nonempty id and description")
        identifiers.append(option["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Option identifiers must be unique")
    context = f"Question: {item['question']}\nState: {item['state']}"
    return [(context, "Option: " + option["description"]) for option in options]


def tokenize_options(tokenizer, item: dict, max_tokens: int) -> dict:
    pairs = text_pairs(item)
    encoded = tokenizer(
        [pair[0] for pair in pairs], text_pair=[pair[1] for pair in pairs],
        padding=False, truncation=False,
    )
    longest = max(map(len, encoded["input_ids"]))
    if longest > max_tokens:
        raise ValueError(f"Option input needs {longest} tokens; limit is {max_tokens}. No truncation applied.")
    return encoded


def load_encoder(device: str, model_id=MODEL_ID, revision=MODEL_REVISION, checkpoint=None):
    location = checkpoint if checkpoint is not None else model_id
    settings = {"local_files_only": True} if checkpoint is not None else {"revision": revision, "token": False}
    tokenizer = AutoTokenizer.from_pretrained(location, trust_remote_code=False, **settings)
    encoder = AutoModel.from_pretrained(
        location, use_safetensors=checkpoint is not None,
        trust_remote_code=False, dtype=torch.float32, **settings,
    ).to(device)
    encoder.requires_grad_(False)
    encoder.eval()  # Turn off dropout: frozen features must be repeatable.
    return tokenizer, encoder


def pool_tokens(encoder, batch):
    """This path retains gradients; callers decide whether the encoder is frozen."""
    hidden = encoder(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return F.layer_norm(pooled, (pooled.shape[-1],))


@torch.no_grad()
def encode_options(encoder, tokenizer, encoded: dict, device: str, batch_size=4):
    """Average contextual token vectors, excluding padding; normalize each vector."""
    features = []
    for start in range(0, len(encoded["input_ids"]), batch_size):
        batch = tokenizer.pad(
            {key: value[start:start + batch_size] for key, value in encoded.items()},
            padding=True, return_tensors="pt",
        ).to(device)
        features.append(pool_tokens(encoder, batch).cpu())
    return torch.cat(features)


class OptionScorer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # A common bias would cancel in softmax, so there is no bias parameter.
        self.score = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.score.weight)  # Begin with equal probabilities.

    def forward(self, features, option_mask):
        scores = self.score(features).squeeze(-1)
        return scores.masked_fill(~option_mask, float("-inf"))


def padded_features(features: list[torch.Tensor]):
    """[examples, options, representation dimensions], with a mask for real options."""
    padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    counts = torch.tensor([len(item) for item in features])
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < counts.unsqueeze(1)
    return padded, mask


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_run(run: Path, device: str):
    manifest = json.loads((run / "manifest.json").read_text())
    checkpoint = None if manifest.get("encoder_frozen", True) else run / "encoder"
    tokenizer, encoder = load_encoder(device, manifest["model_id"], manifest["model_revision"], checkpoint)
    scorer = OptionScorer(encoder.config.hidden_size)
    scorer.load_state_dict(load_file(run / "scorer.safetensors"))
    return tokenizer, encoder, scorer, manifest


def main():
    parser = argparse.ArgumentParser(description="Apply a saved scoring layer to described options.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON with state, question, options; no target needed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    item = json.loads(args.input.read_text())
    device = choose_device(args.device)
    tokenizer, encoder, scorer, manifest = load_run(args.run, device)
    tokens = tokenize_options(tokenizer, item, manifest["max_tokens"])
    features, mask = padded_features([encode_options(encoder, tokenizer, tokens, device)])
    with torch.no_grad():
        probabilities = scorer(features, mask).softmax(dim=-1)[0].tolist()
    ids = [option["id"] for option in item["options"]]
    print(json.dumps({
        "choice": ids[max(range(len(ids)), key=probabilities.__getitem__)],
        "probabilities": dict(zip(ids, probabilities)),
        "note": "Experimental decision model; these probabilities have not been calibrated.",
    }, indent=2))


if __name__ == "__main__":
    main()
