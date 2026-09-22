"""Bind CPU mechanics checks and exercise real candidate forecast contracts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from scale_lab.common import ROOT, file_hash, read_rows, write_json
from tool_lab.decision_learning_v2 import forecast_loss


def qualify(output, tests):
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    status = json.loads((tests/'status.json').read_text())
    if (status['status'] != 'completed' or status['exit_code'] or 'Ran 7 tests' not in (tests/'stderr.log').read_text() or
            not (tests/'stderr.log').read_text().endswith('OK\n')):
        raise ValueError('Consumer mechanics tests failed')
    names = ['tool_lab/decision_learning_v2.py', 'tool_lab/decision_learning_qualify.py',
        'tests/test_decision_learning_v2.py', 'docs/decision-learning-v2-protocol.md',
        'tool_lab/revisioned_live.py', 'tool_lab/live_contracts.py', 'tool_lab/live_mixed.py',
        'tool_lab/guarded_update.py', 'general_lab/outcome_train.py', 'general_lab/rl.py',
        'scale_lab/common.py', 'scale_lab/model.py']
    sources = {n: file_hash(ROOT/n) for n in names}
    inputs = {}; rows = []
    for stage, filename in [('revisioned-admission-v1', 'train-staged-private.jsonl'),
                            ('revisioned-live-v1', 'forecasts-private.jsonl')]:
        folder = ROOT/'output'/stage; frozen = json.loads((folder/'freeze.json').read_text())
        for name, sha in frozen['sources'].items():
            if file_hash(ROOT/name) != sha: raise ValueError('Changed qualified data source')
        if file_hash(folder/filename) != frozen['files'][filename]:
            raise ValueError('Changed qualified candidate forecasts')
        subset = [r for r in read_rows(folder/filename) if 'soft_target' in r]
        inputs[stage] = dict(freeze_sha256=file_hash(folder/'freeze.json'), forecast_rows=len(subset),
            data_sha256=file_hash(folder/filename))
        rows.extend(subset)
    if len(rows) != 3216:
        raise ValueError('Candidate forecast coverage differs')
    write_json(output/'pre-qualification-freeze.json', dict(sources=sources, inputs=inputs,
        tests={p.name: file_hash(p) for p in tests.iterdir() if p.is_file()},
        foundation_model_calls=0, foundation_optimizer_updates=0))
    class SyntheticLogits:
        def __call__(self, chunk):
            width = max(len(r['option_ids']) for r in chunk)
            logits = torch.zeros((len(chunk), width), requires_grad=True)
            return logits, None, None
    policy = SyntheticLogits(); losses = []
    for start in range(0, len(rows), 32):
        loss = forecast_loss(policy, rows[start:start+32])
        if not torch.isfinite(loss): raise ValueError('Invalid candidate loss')
        loss.backward(); losses.append(float(loss.detach()))
    summary = dict(status='qualified_cpu_mechanics_and_candidate_contracts', tiny_cpu_tests_passed=7,
        candidate_distribution_rows_checked=len(rows), forecast_contracts=dict(Counter(r['forecast_contract'] for r in rows)),
        qualified_live_action_types=['shell_action', 'retail_live_action', 'revisioned_live_action'],
        allowed_objective_arms=['outcome', 'reward', 'hybrid'],
        synthetic_loss_range=[min(losses), max(losses)], foundation_model_calls=0, foundation_optimizer_updates=0,
        foundation_training_presentations=0, new_forecast_mixture_admission=False, cuda_policy_qualified=False,
        model_performance_measured=False, seconds=time.monotonic()-started,
        limitations='Tiny network mechanics and synthetic-logit input checks; not 9B training, GPU qualification, a new training recipe, or a transfer result.')
    write_json(output/'summary.json', summary)
    write_json(output/'freeze.json', dict(status=summary['status'], sources=sources, inputs=inputs,
        files={p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != 'freeze.json'}))
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'tests'): p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); qualify(args.output, args.tests)
