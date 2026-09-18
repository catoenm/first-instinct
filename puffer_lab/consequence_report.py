"""Recompute the consequence pilot's selected results from recovered evidence.

This reporting module is deliberately outside the prospective training freeze.
It performs no inference or checkpoint selection.
"""

import argparse
import json
import math
from pathlib import Path
import tempfile

from scale_lab.common import ROOT, file_hash, read_rows, write_json
from .consequence_train import CONFIG, summarize


def matched_predictions(rows, path):
    records = read_rows(path)
    if len(rows) != len(records) or any(r['id'] != p['id'] for r, p in zip(rows, records)):
        raise ValueError('Prediction ordering/identity differs')
    result = []
    for row, record in zip(rows, records):
        values = record['probabilities']
        if set(values) != set(row['option_ids']):
            raise ValueError('Prediction options differ')
        p = [values[k] for k in row['option_ids']]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p)-1) > 1e-5:
            raise ValueError('Invalid predicted distribution')
        result.append(p)
    return result


def replay_diagnostics(folder, cases):
    from .native import compile_core, library, NativeEpisode
    from .text_render import render, semantic_choice
    from .text_baseline import summarize as summarize_policy
    transitions, episodes = read_rows(folder/'transitions.jsonl'), read_rows(folder/'episodes.jsonl')
    expected = [(case, variant) for case in cases['cases'] for variant in cases['variants']]
    if len(episodes) != len(expected):
        raise ValueError('Incomplete episode set')
    cursor = 0
    with tempfile.TemporaryDirectory() as tmp:
        lib = library(compile_core(Path(tmp)/'core.so'))
        for episode, (case, variant) in zip(episodes, expected):
            if any(episode[k] != case[k] for k in case) or episode['variant'] != variant:
                raise ValueError('Case identity differs')
            history, total = [], 0.
            with NativeEpisode(lib, case['world'], case['profile']) as env:
                while not env.state()['done']:
                    record = transitions[cursor]
                    cursor += 1
                    if record['case'] != case['case'] or record['variant'] != variant:
                        raise ValueError('Transition mapping differs')
                    item = render(env.public(), history, variant)
                    if record['input'] != item:
                        raise ValueError('Public prompt cannot be reconstructed')
                    p = record['probabilities']
                    if set(p) != {o['id'] for o in item['options']} or any(not math.isfinite(v) or not 0<=v<=1 for v in p.values()) or abs(sum(p.values())-1)>1e-5:
                        raise ValueError('Invalid policy probabilities')
                    action = semantic_choice(p, item)
                    if record['action'] != action:
                        raise ValueError('Executed action differs from greedy choice')
                    reward = env.step(action)
                    total += reward
                    if record['state'] != env.state() or record['public'] != env.public() or abs(reward-record['reward'])>1e-7:
                        raise ValueError('Environment replay differs')
                    history.append(dict(action=action, result=env.public()['last_result']))
                if episode['actions'] != history or abs(episode['return']-total)>1e-6 or episode['success'] != int(env.state()['outcome']==1):
                    raise ValueError('Episode outcome differs')
    if cursor != len(transitions):
        raise ValueError('Unexpected extra transitions')
    policy = summarize_policy(episodes)
    from .forecast_probe import selected, question
    forecasts = read_rows(folder/'forecasts.jsonl')
    expected_forecasts = [(target, order) for target in selected() for order in (False, True)]
    if len(forecasts) != len(expected_forecasts):
        raise ValueError('Incomplete forecast set')
    for record, (target, order) in zip(forecasts, expected_forecasts):
        q = target['success_probability'][0]/target['success_probability'][1]
        if record['input'] != question(target, order) or record['q'] != q or record['reversed'] != order:
            raise ValueError('Forecast target/prompt differs')
        p = record['probabilities']
        if set(p) != {'success','failure'} or any(not math.isfinite(v) or not 0<=v<=1 for v in p.values()) or abs(sum(p.values())-1)>1e-5:
            raise ValueError('Invalid forecast probabilities')
    forecast = {}
    for order in (False, True):
        for subset in ('all', 'uncertain', 'deterministic'):
            group = [r for r in forecasts if r['reversed']==order and
                     (subset=='all' or (0<r['q']<1)==(subset=='uncertain'))]
            forecast[f'{order}/{subset}'] = dict(n=len(group), mse=sum((r['probabilities']['success']-r['q'])**2 for r in group)/len(group))
    return dict(policy=policy, forecast=forecast, replayed_transitions=cursor)


def report(root):
    data, run = root/'data', root/'run'
    receipt = json.loads((run/'run.json').read_text())
    frozen = json.loads((data/'freeze.json').read_text())
    if receipt['status'] != 'complete' or receipt['config'] != CONFIG:
        raise ValueError('Training or final evaluation incomplete')
    if file_hash(data/'freeze.json') != receipt['freeze_sha256'] or file_hash(data/'freeze.json') != file_hash(ROOT/'results/consequence-training-v1/freeze.json'):
        raise ValueError('Prospective freeze differs')
    for name, expected in frozen['files'].items():
        if file_hash(data/name) != expected:
            raise ValueError('Frozen data changed: '+name)
    first, last = receipt['initial_trainable'], receipt['first_updated_trainable']
    if first['parameters'] != last['parameters'] or first['sha256']==last['sha256']:
        raise ValueError('No evidence of an adapter update')
    step = receipt['best_step']
    result = dict(status='verified_complete', selected_step=step, optimizer_steps=receipt['steps'],
                  first_optimizer_changed_internal_adapters=True,
                  selected_changed_internal_adapters=receipt['selected_trainable']['sha256'] != first['sha256'],
                  trainable_parameters=first['parameters'], training_seconds=receipt['seconds'],
                  scope='Supervised one-mechanism development pilot; not reinforcement learning or independent task transfer.',
                  data_counts=frozen['counts'])
    for split, soft, stem in [('validation', True, 'consequence'), ('retention', False, 'retention')]:
        rows = read_rows(data/(split+'.jsonl'))
        before = summarize(rows, matched_predictions(rows,run/('baseline-'+stem+'.jsonl')), soft)
        after_path = run/(f'{stem}-step-{step}.jsonl' if step else 'baseline-'+stem+'.jsonl')
        after = summarize(rows, matched_predictions(rows,after_path), soft)
        result[stem] = dict(baseline=before, selected=after)
    cases = json.loads((data/'cases.json').read_text())
    result['baseline_diagnostics'] = replay_diagnostics(run/'baseline-diagnostics',cases)
    result['selected_diagnostics'] = replay_diagnostics(run/'selected-diagnostics',cases)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = report(args.root)
    write_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('consequence','retention')},indent=2))
