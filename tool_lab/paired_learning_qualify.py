"""Bounded CPU target-interface qualification; no model weights or optimizer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json
from tool_lab.paired_curriculum import require
from tool_lab.paired_learning import model_row, require_use, target_loss


def qualify(source, output):
    start = time.monotonic(); torch.set_num_threads(1); output.mkdir(parents=True, exist_ok=False)
    names = ['tool_lab/paired_learning.py', 'tool_lab/paired_learning_qualify.py', 'tests/test_paired_learning.py',
             'docs/paired-learning-v1-protocol.md', 'tool_lab/paired_admission.py', 'tool_lab/decision_learning_v2.py']
    write_json(output/'preparation-freeze.json', dict(sources={n: file_hash(ROOT/n) for n in names},
        admission_summary_sha256=file_hash(source/'summary.json'),
        admission_usage_sha256=file_hash(source/'usage-private.json'), maximum_presentations=14313,
        model_weights_loaded=False, optimizer_created=False))
    try:
        admission = json.loads((source/'summary.json').read_text())
        require(admission['status'] == 'qualified_paired_training_admission', 'Unqualified training admission')
        for n, sha in admission['files'].items(): require(file_hash(source/n) == sha, 'Admitted artifact changed')
        rows = read_rows(source/'train-presentations-private.jsonl')
        usage = json.loads((source/'usage-private.json').read_text())
        require(len(rows) == 14313 and len({r['id'] for r in rows}) == len(usage['row_sha256']) == 14313, 'Admitted coverage differs')
        require(file_hash(source/'pre-admission-freeze.json') == usage['pre_admission_freeze_sha256'], 'Usage binding changed')
        total_loss = 0.; batches = 0; forward_types = Counter()
        for start_index in range(0, len(rows), 64):
            selected = rows[start_index:start_index+64]; require_use(selected, usage)
            for row in selected:
                public = model_row(row)
                require(set(public) == {'id', 'task', 'input_ids', 'option_ids', 'target_indices'} and
                        not public['target_indices'] and digest(public['input_ids']) == row['token_sha256'],
                        'Target boundary or exact tokens changed')
                forward_types[public['task']] += 1
            counts = [len(r['option_ids']) for r in selected]
            scores = torch.zeros(len(selected), max(counts), dtype=torch.float64, requires_grad=True)
            loss = target_loss(scores, [r['supervision'] for r in selected], counts)
            loss.backward()
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(scores.grad).all()), 'Nonfinite target loss or derivative')
            require(bool((scores.grad.sum(-1).abs() < 1e-12).all()), 'Categorical derivative does not conserve mass')
            total_loss += float(loss.detach())*len(selected); batches += 1
        summary = dict(status='qualified_paired_supervised_consumer_cpu', presentations_validated=len(rows),
            canonical_questions=len({r['canonical_id'] for r in rows}), bounded_score_tensor_batches=batches,
            forward_contract_counts=dict(forward_types), supervision_counts=dict(Counter(r['supervision']['semantics'] for r in rows)),
            uniform_score_loss=total_loss/len(rows), foundation_weights_loaded=False, foundation_model_calls=0,
            optimizer_updates=0, actual_learning_presentations_consumed=0,
            seconds=time.monotonic()-start,
            scope='Admission/target interface and CPU derivatives only; no GPU, policy performance or full pilot qualification.',
            required_next='Freeze a useful bounded pilot recipe and integrate this consumer with replay and the qualified runtime.')
        summary['files'] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'summary.json', summary)
        return summary
    except BaseException as exc:
        write_json(output/'REJECTED.json', dict(error=type(exc).__name__, detail=str(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    print(json.dumps(qualify(p.parse_args().source, p.parse_args().output), indent=2))
