"""Bind each validation prediction file to immutable adapter bytes before resuming."""
from pathlib import Path
import shutil

from general_lab import train
from scale_lab.common import file_hash, write_json

original_write_rows = train.write_rows


def checkpointed_rows(path, rows):
    original_write_rows(path, rows)
    path = Path(path)
    if path.name.startswith('validation-step-'):
        step = int(path.stem.rsplit('-', 1)[1])
        folder = path.parent / 'validation-checkpoints' / str(step)
        shutil.copytree(path.parent / 'latest', folder)
        write_json(path.parent / f'validation-checkpoint-{step}.json',
                   {'step': step, 'adapter_sha256': file_hash(folder / 'adapter_model.safetensors'),
                    'predictions_sha256': file_hash(path)})


if __name__ == '__main__':
    train.write_rows = checkpointed_rows
    train.main()
