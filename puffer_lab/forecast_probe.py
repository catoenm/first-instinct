"""Forecast specified executed events, with exact targets kept outside prompts."""

import argparse
import json
import math
from pathlib import Path
import time

from general_lab.robustness_local import LocalPredictor
from scale_lab.common import encode, file_hash, messages
from .contract import ACTIONS, digest
from .text_baseline import ROOT, append, binding, read, require, rows, tokenizer, write
from .text_render import DESCRIPTIONS, render

TARGETS = ROOT / 'results/puffer-reservation-v1/qualification/conditional-targets.jsonl'
EXTRA_SOURCES = ('puffer_lab/forecast_probe.py', 'test_reservation_forecast.py',
                 'docs/reservation-forecast-v1-protocol.md')
CONTINUATION = (
    'After the specified first action, repeatedly apply the first matching rule below until finish or no turns remain: '
    '(1) If the request header and both allocations exist, finish. '
    '(2) Otherwise, if any request header or allocation exists, undo this request. '
    '(3) Otherwise, if the account is known absent, create it. '
    '(4) Otherwise, if A is known zero, replenish A. '
    '(5) Otherwise, if B is known zero, replenish B. '
    '(6) Otherwise, if either stock quantity is unknown, inspect stock. '
    '(7) Otherwise, if account presence is unknown, inspect account. '
    '(8) Otherwise, reserve atomically. '
    'Known here means explicitly observed or revealed by an action; this fixed controller does not substitute deductions from the prior. '
    'The first action and every continuation action consume a turn. No action follows a terminal state.'
)


def selected():
    result = [r for r in rows(TARGETS) if len(r['public_context']['history']) == 1 or
              0 < r['success_probability'][0] < r['success_probability'][1]]
    require(len(result) == 39, 'Unexpected qualified forecast corpus')
    return result


def question(target, reversed_order=False):
    # Explicit allowlist; probabilities and branch outcomes are not renderer arguments.
    context = target['public_context']
    observation = context['history'][-1]['observation']
    history = [{'action': h['action'], 'result': h['observation']['last_result']} for h in context['history'][1:]]
    public = render(observation, history, 'original')
    first = DESCRIPTIONS[ACTIONS.index(target['action'])]
    options = [{'id': 'success', 'description': 'The episode ends with a complete valid reservation.'},
               {'id': 'failure', 'description': 'The episode ends without a complete valid reservation.'}]
    if reversed_order:
        options.reverse()
    return {'state': public['state'] + '\nFixed continuation: ' + CONTINUATION,
            'question': 'First action: ' + first + ' Then follow the fixed continuation exactly. '
                        'Which final event occurs, given the hidden world and the supplied evidence?',
            'options': options}


def prepare(output, adapter_run):
    output = Path(output)
    require(not output.exists(), 'Use a fresh forecast output directory')
    provenance = binding(adapter_run)
    tok = tokenizer()
    public, gold = [], []
    for target in selected():
        identity = digest([target['context_sha256'], target['action']])
        gold.append({'id': identity, 'success_probability': target['success_probability'],
                     'context_sha256': target['context_sha256'], 'action': target['action']})
        for order in ('original', 'reversed'):
            item = question(target, order == 'reversed')
            ids = encode(tok, item, 1536)
            public.append({'index': len(public), 'id': identity, 'order': order,
                           'input': item, 'input_sha256': digest(item), 'messages': messages(item), 'input_ids': ids})
    require(binding(adapter_run) == provenance, 'Provenance changed during preparation')
    frozen = {'schema': 'reservation-forecast-probe-v1', 'created_at_unix': time.time(),
              'binding': provenance, 'extra_source_sha256': {p: file_hash(ROOT / p) for p in EXTRA_SOURCES},
              'source_targets_sha256': file_hash(TARGETS), 'public_sha256': digest(public), 'gold_sha256': digest(gold),
              'maximum_model_attempts': 78, 'max_wall_seconds': 3600, 'retries': 0, 'selection_role': 'none',
              'min_tokens': min(len(r['input_ids']) for r in public), 'max_tokens': max(len(r['input_ids']) for r in public)}
    frozen['content_sha256'] = digest(frozen)
    output.mkdir(parents=True)
    write(output / 'questions.json', public)
    write(output / 'gold.json', gold)
    write(output / 'freeze.json', frozen)
    return frozen


def verify(output, adapter_run=None):
    output = Path(output)
    frozen = read(output / 'freeze.json')
    require(frozen['content_sha256'] == digest({k: v for k, v in frozen.items() if k != 'content_sha256'}), 'Freeze differs')
    require(frozen['schema'] == 'reservation-forecast-probe-v1' and frozen['maximum_model_attempts'] == 78 and
            frozen['max_wall_seconds'] == 3600 and frozen['retries'] == 0, 'Protocol bounds differ')
    for name, expected in {**frozen['binding']['sources_sha256'], **frozen['extra_source_sha256']}.items():
        require(file_hash(ROOT / name) == expected, f'Frozen source differs: {name}')
    require(file_hash(TARGETS) == frozen['source_targets_sha256'], 'Qualified targets changed')
    public, gold = read(output / 'questions.json'), read(output / 'gold.json')
    require(digest(public) == frozen['public_sha256'] and digest(gold) == frozen['gold_sha256'], 'Frozen data differs')
    require(len(public) == 78 and len(gold) == 39, 'Question count differs')
    if adapter_run is not None:
        require(binding(adapter_run) == frozen['binding'], 'Model, tokenizer or runtime differs')
    return frozen, public, gold


def execute(output, adapter_run):
    output = Path(output)
    frozen, public, _ = verify(output, adapter_run)
    require(not (output / 'run.json').exists(), 'Do not overwrite or silently resume a probe')
    started, attempts = time.monotonic(), 0
    receipt = {'status': 'running', 'training': False, 'model_updated': False, 'started_at_unix': time.time(),
               'freeze_sha256': file_hash(output / 'freeze.json')}
    write(output / 'run.json', receipt)
    try:
        predictor = LocalPredictor(8766, frozen['binding']['expected_metadata'], output)
        require(predictor.limits['max_tokens'] == 1536, 'Serving token limit changed')
        for row in public:
            require(attempts < 78 and time.monotonic() - started < 3600, 'Forecast probe budget exhausted')
            append(output / 'attempts.jsonl', {'index': attempts, 'input_sha256': row['input_sha256']})
            attempts += 1
            probability = predictor([row['input']])[0]
            require(set(probability) == {'success', 'failure'} and
                    all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1 for p in probability.values()) and
                    abs(sum(probability.values()) - 1) < 1e-5, 'Invalid binary prediction')
            print(json.dumps({'completed': attempts, 'of': 78}), flush=True)
        predictor.verify_end()
        verify(output, adapter_run)
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='failed', error={'type': type(error).__name__, 'detail': str(error)})
        raise
    finally:
        receipt.update(attempts=attempts, elapsed_seconds=time.monotonic() - started, completed_at_unix=time.time())
        write(output / 'run.json', receipt)


def loss(p, q):
    p = min(1 - 1e-12, max(1e-12, p))
    return -q * math.log(p) - (1 - q) * math.log(1 - p)


def analyze(output):
    output = Path(output)
    frozen, public, gold = verify(output)
    run, responses, attempts = read(output / 'run.json'), rows(output / 'responses.jsonl'), rows(output / 'attempts.jsonl')
    require(run['status'] == 'complete' and run['attempts'] == len(responses) == len(attempts) == 78, 'Incomplete probe')
    target_by_id = {r['id']: r for r in gold}
    original = {digest([r['context_sha256'], r['action']]): r for r in selected()}
    predictions = []
    for row, response, attempt in zip(public, responses, attempts):
        require(row['index'] == response['index'] == attempt['index'] and row['input_sha256'] == attempt['input_sha256'],
                'Prediction mapping differs')
        target = original[row['id']]
        require(question(target, row['order'] == 'reversed') == row['input'] and messages(row['input']) == row['messages'],
                'Public prompt cannot be reconstructed')
        require(target_by_id[row['id']]['success_probability'] == target['success_probability'], 'Gold probability differs')
        p = response['probabilities']['success']
        q = target['success_probability'][0] / target['success_probability'][1]
        require(math.isfinite(p) and 0 <= p <= 1 and set(response['probabilities']) == {'success', 'failure'} and
                abs(sum(response['probabilities'].values()) - 1) < 1e-5, 'Invalid response probabilities')
        predictions.append({'id': row['id'], 'order': row['order'], 'p': p, 'q': q, 'uncertain': 0 < q < 1})
    summary = {}
    for order in ('original', 'reversed'):
        for subset in ('all', 'uncertain', 'deterministic'):
            group = [r for r in predictions if r['order'] == order and
                     (subset == 'all' or r['uncertain'] == (subset == 'uncertain'))]
            summary[order + '/' + subset] = {
                'n': len(group), 'probability_mse': sum((r['p'] - r['q']) ** 2 for r in group) / len(group),
                'mean_absolute_error': sum(abs(r['p'] - r['q']) for r in group) / len(group),
                'expected_log_loss': sum(loss(r['p'], r['q']) for r in group) / len(group),
                'excess_log_loss': sum(loss(r['p'], r['q']) - loss(r['q'], r['q']) for r in group) / len(group),
                'uniform_probability_mse': sum((.5 - r['q']) ** 2 for r in group) / len(group)}
    by_id = {identity: {r['order']: r['p'] for r in predictions if r['id'] == identity} for identity in target_by_id}
    deltas = [abs(r['original'] - r['reversed']) for r in by_id.values()]
    return {'status': 'verified_complete', 'distinct_targets': 39, 'model_attempts': 78, 'summary': summary,
            'order_mean_absolute_difference': sum(deltas) / len(deltas), 'order_max_absolute_difference': max(deltas),
            'predictions': predictions, 'training': False, 'model_updated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'execute', 'analyze'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter-run', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        frozen = prepare(args.output, args.adapter_run)
        print(json.dumps({k: frozen[k] for k in ('min_tokens', 'max_tokens', 'maximum_model_attempts')}))
    elif args.command == 'execute':
        execute(args.output, args.adapter_run)
    else:
        report = analyze(args.output)
        write(args.output / 'audit.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))
