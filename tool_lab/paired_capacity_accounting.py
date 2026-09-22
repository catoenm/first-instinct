"""Account for copied paired-learning receipts without touching a live model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scale_lab.common import file_hash, read_rows, write_json
from tool_lab.live_pilot_accounting import summarize
from tool_lab.paired_curriculum import require


def learning_counts(events, paired, replay, schedule):
    """Bind completed backwards to exact scheduled positions and canonical questions."""
    ledger = summarize(events)
    paired = {r['id']: r for r in paired}; replay = {r['id']: r for r in replay}
    plans = {p['update']: p for p in schedule}; offsets = Counter(); completed = defaultdict(list)
    terminal = {e['update']: e['phase'] for e in events
                if e['phase'] in ('accepted', 'rejected', 'interrupted_rejected')}
    fields = dict(teacher='teacher_ids', outcome='forecast_ids', replay='replay_ids')
    for event in events:
        if event['phase'] not in ('started_backward', 'completed_backward'): continue
        component, update = event['component'], event['update']; key = (update, component)
        require(component in fields and update in plans, 'Unknown objective or update')
        ids = event['ids']; source = replay if component == 'replay' else paired
        require(all(i in source for i in ids), 'Unadmitted presentation')
        expected = plans[update][fields[component]][offsets[key]:offsets[key]+len(ids)]
        require(ids == expected, 'Consumed presentation differs from frozen schedule')
        canonical = [] if component == 'replay' else [source[i]['canonical_id'] for i in ids]
        require(event['canonical_ids'] == canonical, 'Canonical identity differs from admitted row')
        if event['phase'] == 'completed_backward':
            offsets[key] += len(ids); completed[component].extend((update, source[i]) for i in ids)
    for update, phase in terminal.items():
        if phase == 'accepted':
            require(all(offsets[(update, component)] == len(plans[update][field])
                        for component, field in fields.items()), 'Accepted update did not consume its full recipe')
    coverage = {}
    for component, entries in completed.items():
        def count(items):
            ids = [r['id'] for _, r in items]
            canonical = [r.get('canonical_id', r['id']) for _, r in items]
            return dict(completed_presentations=len(ids), distinct_position_ids=len(set(ids)),
                        distinct_canonical_questions=len(set(canonical)), repeated_position_presentations=len(ids)-len(set(ids)))
        coverage[component] = dict(all_completed_backwards=count(entries),
            accepted_transactions=count([x for x in entries if terminal.get(x[0]) == 'accepted']),
            by_family={family: count([x for x in entries if x[1].get('family', 'general') == family])
                       for family in sorted({r.get('family', 'general') for _, r in entries})})
    return dict(transactions=ledger, coverage=coverage,
        note='Completed backwards include later rollbacks. Distinct positions and canonical questions are not independent worlds.')


def report(run, data):
    receipt = json.loads((run/'run.json').read_text()); frozen = json.loads((data/'freeze.json').read_text())
    require(receipt['data_freeze_sha256'] == file_hash(data/'freeze.json') and
            receipt['parent_adapter_sha256'] == frozen['parent_adapter_sha256'], 'Run/data lineage differs')
    for name in ('paired.jsonl', 'replay.jsonl', 'schedule.jsonl'):
        require(file_hash(data/name) == frozen['files'][name], 'Accounting source changed')
    path = run/'learning-ledger.jsonl'; events = read_rows(path) if path.exists() else []
    counts = learning_counts(events, read_rows(data/'paired.jsonl'), read_rows(data/'replay.jsonl'), read_rows(data/'schedule.jsonl'))
    ledger = counts['transactions']
    agrees = receipt['accepted_steps'] == ledger['accepted_transactions'] and receipt['physical_optimizer_attempts'] == ledger['physical_optimizer_attempts']
    if receipt['status'] in ('complete', 'failed', 'bounded_incomplete'):
        require(agrees, 'Closed run counters differ from ledger')
        require(not ledger['attempted_updates_without_closure'], 'Closed run has unclosed optimizer attempt')
    return dict(status=receipt['status'], receipt_counters_match_ledger=agrees,
        selected_update=receipt['selected_update'], parent_adapter_sha256=receipt['parent_adapter_sha256'],
        data_freeze_sha256=receipt['data_freeze_sha256'], actual_learning=counts,
        scope='Consumption only. Qualification backwards and evaluation executions are excluded. No metric or checkpoint audit.',
        release_eligible=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'output'): p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); require(not args.output.exists(), 'Preserve prior accounting snapshots')
    write_json(args.output, report(args.run, args.data))
