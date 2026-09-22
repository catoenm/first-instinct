"""Exact one-command consequence targets derived from executed SQLite branches."""

import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from scale_lab.common import file_hash
from .contract import ACTIONS, digest
from .text_baseline import ROOT, append, read, require, rows, write
from .text_render import DESCRIPTIONS, REWORDED, VARIANTS, render

BRANCHES = ROOT / 'results/puffer-reservation-v1/qualification/branches.jsonl'
TARGETS = ROOT / 'results/puffer-reservation-v1/qualification/conditional-targets.jsonl'
SOURCES = ('puffer_lab/dynamics_data.py', 'tests/test_reservation_dynamics.py',
           'docs/reservation-dynamics-data-v1.md', 'puffer_lab/text_render.py', 'puffer_lab/contract.py')
EVENTS = {
    'request': ('What is the status of request 17 immediately after this command?',
                [('empty', 'No header or allocation exists for this request.'),
                 ('partial', 'Some request records exist but the full reservation is incomplete.'),
                 ('complete', 'The request has its header and exactly one allocated A and one allocated B.')]),
    'account': ('Does account 7 exist immediately after this command?',
                [('yes', 'Account 7 exists.'), ('no', 'Account 7 is absent.')]),
    'stock_a': ('Is A out of stock immediately after this command?',
                [('yes', 'A has zero available units.'), ('no', 'A has at least one available unit.')]),
    'stock_b': ('Is B out of stock immediately after this command?',
                [('yes', 'B has zero available units.'), ('no', 'B has at least one available unit.')]),
}


def truth(event, state):
    if event == 'request':
        if state['header'] and state['alloc_a'] == state['alloc_b'] == 1:
            return 'complete'
        return 'partial' if state['header'] or state['alloc_a'] or state['alloc_b'] else 'empty'
    if event == 'account':
        return 'yes' if state['account'] else 'no'
    if event in ('stock_a', 'stock_b'):
        return 'yes' if state[event[-1]] == 0 else 'no'
    raise ValueError('Unknown consequence event')


def event_input(context, action, event, variant):
    before = context['history'][-1]['observation']
    history = [{'action': h['action'], 'result': h['observation']['last_result']} for h in context['history'][1:]]
    public = render(before, history, variant)
    descriptions = REWORDED if variant == 'reworded' else DESCRIPTIONS
    question, choices = EVENTS[event]
    item = {'state': public['state'],
            'question': 'Execute just this one command: ' + descriptions[ACTIONS.index(action)] + ' '
                        + question + ' Do not execute any later repair or continuation.',
            'options': [{'id': identity, 'description': description} for identity, description in choices]}
    if variant == 'reversed':
        item['options'].reverse()
    return item


def generate():
    primary = [r for r in rows(BRANCHES) if r['replay'] == 0]
    groups = defaultdict(list)
    for index, branch in enumerate(primary):
        if branch['prior_weight']:
            groups[(branch['context_sha256'], branch['first'])].append((index, branch))
    require(len(groups) == 288, 'Unexpected qualified context/action count')
    result = []
    for (context_hash, action), support in sorted(groups.items()):
        require(len({r['world'] for _, r in support}) == len(support), 'Repeated hidden world would distort probabilities')
        context = support[0][1]['context']
        require(all(r['context'] == context for _, r in support), 'Conditioning histories differ')
        denominator = sum(r['prior_weight'] for _, r in support)
        for event, (_, choices) in EVENTS.items():
            counts = {identity: 0 for identity, _ in choices}
            for _, branch in support:
                step = branch['steps'][len(branch['prefix'])]
                require(step['action'] == action, 'First-action receipt mapping differs')
                counts[truth(event, step['state'])] += branch['prior_weight']
            fractions = {identity: Fraction(n, denominator) for identity, n in counts.items()}
            require(sum(fractions.values()) == 1, 'Target probability mass differs')
            group = digest(['reservation-history-v1', context_hash])
            event_id = digest([context_hash, action, event])
            for variant in VARIANTS:
                item = event_input(context, action, event, variant)
                result.append({'id': digest([event_id, variant]), 'event_id': event_id, 'group_id': group,
                               'role': 'development_training_pool', 'event': event, 'variant': variant,
                               'input': item, 'target': {'probabilities': {k: float(v) for k, v in fractions.items()},
                                                       'exact_fractions': {k: [v.numerator, v.denominator] for k, v in fractions.items()}},
                               'provenance': {'context_sha256': context_hash, 'action': action,
                                              'primary_branch_indices': [index for index, _ in support],
                                              'label_origin': 'actual_SQLite_state_after_specified_command'}})
    return result


def statistics(records):
    canonical = [r for r in records if r['variant'] == 'original']
    return {'rows': len(records), 'canonical_event_questions': len(canonical),
            'context_action_groups': len({(r['provenance']['context_sha256'], r['provenance']['action']) for r in canonical}),
            'public_histories': len({r['group_id'] for r in canonical}),
            'uncertain_canonical_events': sum(sum(p > 0 for p in r['target']['probabilities'].values()) > 1 for r in canonical),
            'probability_fractions': sorted({tuple(v) for r in canonical for v in r['target']['exact_fractions'].values()}),
            'by_event': {event: {'canonical_questions': sum(r['event'] == event for r in canonical),
                                 'uncertain_questions': sum(r['event'] == event and
                                                           sum(p > 0 for p in r['target']['probabilities'].values()) > 1
                                                           for r in canonical)} for event in EVENTS},
            'independent_mechanisms': 1, 'untouched_test_rows': 0, 'new_SQLite_attempts': 0, 'model_calls': 0}


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = {'schema': 'reservation-dynamics-development-v1',
              'source_sha256': {p: file_hash(ROOT / p) for p in SOURCES},
              'branches_sha256': file_hash(BRANCHES), 'qualified_targets_sha256': file_hash(TARGETS),
              'max_rows': 3500, 'max_bytes': 25_000_000, 'training': False}
    write(output / 'freeze.json', frozen)
    records = generate()
    require(len(records) == 3456 and len(records) <= frozen['max_rows'], 'Data row budget differs')
    for row in records:
        append(output / 'rows.jsonl', row)
    require((output / 'rows.jsonl').stat().st_size <= frozen['max_bytes'], 'Data size budget exceeded')
    report = statistics(records)
    report['data_sha256'] = file_hash(output / 'rows.jsonl')
    write(output / 'summary.json', report)
    return report


def verify(output):
    output = Path(output)
    frozen = read(output / 'freeze.json')
    for name, expected in frozen['source_sha256'].items():
        require(file_hash(ROOT / name) == expected, f'Frozen data source changed: {name}')
    require(file_hash(BRANCHES) == frozen['branches_sha256'] and file_hash(TARGETS) == frozen['qualified_targets_sha256'],
            'Qualified execution source changed')
    actual = rows(output / 'rows.jsonl')
    require(actual == generate(), 'Published consequence targets cannot be reconstructed')
    summary = statistics(actual)
    # JSON turns tuples into lists.
    summary['probability_fractions'] = [list(v) for v in summary['probability_fractions']]
    summary['data_sha256'] = file_hash(output / 'rows.jsonl')
    require(summary == read(output / 'summary.json'), 'Data summary differs')
    return {'status': 'verified', **summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(__import__('json').dumps(build(args.output) if args.command == 'build' else verify(args.output), indent=2))
