"""Read-only non-game transfer measurement on a frozen, existing benchmark."""
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from scale_lab.common import MODELS, label_token_ids, read_rows, write_json, write_rows, file_hash
from scale_lab.model import load_model, evaluate
from general_lab.train import macro_metrics


def run(data, adapter, output):
    output.mkdir(parents=True, exist_ok=False)
    spec = MODELS['qwen35-9b']
    tok = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], token=False)
    model = load_model(spec, 'cuda', adapter, training=False)
    rows = read_rows(data / 'transfer.jsonl')
    predictions = evaluate(model, rows, label_token_ids(tok), tok.pad_token_id or tok.eos_token_id, 'cuda', 8, 64)
    write_rows(output / 'predictions.jsonl', predictions)
    write_json(output / 'metrics.json', dict(adapter_sha256=file_hash(adapter / 'adapter_model.safetensors'),
        data_sha256=file_hash(data / 'transfer.jsonl'), **macro_metrics(predictions)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    run(args.data, args.adapter, args.output)
