"""Execute every forecast branch and qualify controls before freezing a pilot."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json, write_rows
from tool_lab.evidence_env import (Episode, Executor, HORIZON, VERSION, commands,
                                  forecast_input, make_cases, reference, terminal_success)


def write_cases(path, cases):
    # The entity mapping's declared iteration order is part of menu construction.
    # Generic sorted-key serialization changes it and thus permutes option labels.
    with path.open('w') as stream:
        for case in cases:stream.write(json.dumps(case,allow_nan=False)+'\n')


def case_rows(case, executor):
    rows, receipts, controls = [], [], []
    for policy in ('reference', 'blind_left', 'always_inspect'):
        ep = Episode(case, executor)
        while not ep.done:
            if policy == 'blind_left':
                action = 'repair_0' if not ep.depth else 'finish'
            elif policy == 'always_inspect' and ep.depth == 0:
                action = 'summary'
            else:
                action = reference(ep.input())
            ep.step(action)
        receipt = ep.receipt(); receipt['policy'] = policy
        controls.append(receipt)
        if policy == 'reference':
            expected = 0. if case['regime'] == 'costly_hidden' else (1. - case['write_cost']
                       - (case['read_cost'] if case['initial_evidence'] != 'current' else 0.)
                       - (case['unlock_cost'] if case['initial_lock'] else 0.))
            if abs(receipt['reward'] - expected) > 1e-9:
                raise ValueError('Reference reward disagrees with independent cost accounting')
    for phase, prefix in [('initial', []), ('inspected', ['summary']), ('unlocked', ['summary', 'unlock'])]:
        for action in ('repair_0', 'repair_1'):
            ep = Episode(case, executor)
            for name in prefix:
                ep.step(name)
            item = forecast_input(ep.input(), action)
            ep.step(action)
            # The forecast asks to stop now, even after a lock failure.
            success = terminal_success(case, executor.initial, executor.current)
            identity = digest([case['id'], phase, action])
            receipt = dict(id=identity, case_id=case['id'], phase=phase, action=action,
                           input_sha256=digest(item), success=success, trajectory=ep.receipt())
            receipts.append(receipt)
            rows.append(dict(id=identity, group_id=case['group_id'], case_id=case['id'], family=case['family'],
                             regime=case['regime'], phase=phase, task=case['family'] + '/forecast', input=item,
                             target={'option_id': 'yes' if success else 'no'}, outcome=int(success),
                             execution_sha256=digest(receipt)))
    return rows, receipts, controls


def build(output, train_bundles=4, evaluation_bundles=2):
    output.mkdir(parents=True, exist_ok=False)
    cases = {s: make_cases(s, train_bundles if s == 'train' else evaluation_bundles)
             for s in ('train', 'validation', 'test')}
    sets = [{c['group_id'] for c in cases[s]} for s in cases]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Split leakage')
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in
               [Path(__file__), ROOT/'tool_lab/evidence_env.py', ROOT/'tool_lab/evidence_shell.py',
                ROOT/'tool_lab/contextual_shell.py', ROOT/'tool_lab/shell_supervision.py']}
    write_json(output/'pre-execution-freeze.json', dict(schema=VERSION, sources=sources,
               cases={s: len(v) for s,v in cases.items()}, controls=['reference','blind_left','always_inspect'],
               forecast_phases=['initial','inspected','unlocked'], maximum_branches=sum(map(len,cases.values()))*6))
    counts, by_policy, all_rows = {}, defaultdict(list), {}
    jobs = [(s,c) for s in cases for c in cases[s]]
    def chunk_job(chunk):
        engine = Executor()
        results = []
        try:
            for split, case in chunk:
                rows, receipts, controls = case_rows(case, engine)
                results.append((split, rows, receipts, controls))
            return results, engine.count
        finally:
            engine.close()
    streams = {s: [(output/(s+suffix)).open('w') for suffix in ('-forecasts.jsonl','-executions.jsonl','-controls.jsonl')] for s in cases}
    executed = completed = 0
    try:
        # Bounded workers each own one persistent Docker process. Fixed chunks
        # preserve deterministic output order independent of completion order.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for results, commands_count in pool.map(chunk_job, [jobs[i::4] for i in range(4)]):
                executed += commands_count
                for split, rows, receipts, controls in results:
                    for stream, values in zip(streams[split], (rows,receipts,controls)):
                        for value in values:stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
                        stream.flush()
                    all_rows.setdefault(split,[]).extend(rows)
                    for r in controls:by_policy[(split,r['policy'])].append(r)
                    completed += 1
                print(json.dumps({'completed_cases':completed,'executed_commands':executed}),flush=True)
    finally:
        for items in streams.values():
            for stream in items:stream.close()
    uncertain = {}
    for split, rows in all_rows.items():
        groups = defaultdict(list)
        for r in rows:
            if r['regime'] in ('hidden','costly_hidden','stale') and r['phase']=='initial':
                groups[(r['group_id'],r['regime'],digest(r['input']))].append(r['outcome'])
        if any(sorted(v) != [0,1] for v in groups.values()):
            raise ValueError('Paired hidden-world uncertainty was lost')
        uncertain[split] = len(groups)
        write_cases(output/(split+'-cases.jsonl'),cases[split])
        counts[split] = {'cases':len(cases[split]),'forecasts':len(rows)}
    controls = {f'{split}/{name}': dict(episodes=len(v),successes=sum(r['success'] for r in v),
                    reward=sum(r['reward'] for r in v)/len(v)) for (split,name),v in by_policy.items()}
    manifest = dict(schema=VERSION,status='qualified',counts=counts,executed_commands=executed,
                    paired_uncertain_input_groups=uncertain,controls=controls,sources=sources,
                    files={p.name:file_hash(p) for p in output.glob('*.jsonl')},
                    scope='Authored executable tasks and actual branch outcomes; fixed command catalog. No model results yet.')
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


def finalize_existing(output):
    """Recover complete executions after the original overly strict group gate.

    Immediate success ignores inspection price, so hidden/costly_hidden can
    share forecast inputs. Qualify paired outcomes within each regime while
    retaining the original executed generator and all original labels.
    """
    if (output/'manifest.json').exists():raise ValueError('Already finalized')
    pre=json.loads((output/'pre-execution-freeze.json').read_text())
    old=output/'generation-builder.py'
    if file_hash(old)!=pre['sources']['tool_lab/evidence_data.py']:raise ValueError('Missing executed generator')
    for name,sha in pre['sources'].items():
        if name!='tool_lab/evidence_data.py' and file_hash(ROOT/name)!=sha:raise ValueError('Environment source changed')
    counts={};controls={};uncertain={};executed=0
    for split,total in pre['cases'].items():
        cases=make_cases(split,total//96)
        rows=read_rows(output/(split+'-forecasts.jsonl'))
        receipts=read_rows(output/(split+'-executions.jsonl'))
        traces=read_rows(output/(split+'-controls.jsonl'))
        expected={digest([c['id'],phase,a]) for c in cases for phase in ('initial','inspected','unlocked') for a in ('repair_0','repair_1')}
        if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected or {r['id'] for r in receipts}!=expected or len(receipts)!=len(expected):
            raise ValueError('Incomplete branch coverage')
        desired={(c['id'],p) for c in cases for p in ('reference','blind_left','always_inspect')}
        if len(traces)!=len(desired) or {(r['id'],r['policy']) for r in traces}!=desired:raise ValueError('Incomplete controls')
        groups=defaultdict(list)
        for r in rows:
            if r['regime'] in ('hidden','costly_hidden','stale') and r['phase']=='initial':
                groups[(r['group_id'],r['regime'],digest(r['input']))].append(r['outcome'])
        if any(sorted(v)!=[0,1] for v in groups.values()):raise ValueError('Unbalanced paired uncertainty')
        uncertain[split]=len(groups)
        for p in ('reference','blind_left','always_inspect'):
            values=[r for r in traces if r['policy']==p]
            controls[split+'/'+p]=dict(episodes=len(values),successes=sum(r['success'] for r in values),reward=sum(r['reward'] for r in values)/len(values))
        for t in traces+[r['trajectory'] for r in receipts]:
            executed+=len(t['prefix'])+sum(e['observation'] is not None for e in t['events'])
        write_cases(output/(split+'-cases.jsonl'),cases)
        counts[split]=dict(cases=len(cases),forecasts=len(rows))
    manifest=dict(schema=VERSION,status='qualified',counts=counts,executed_commands=executed,
                  paired_uncertain_input_groups=uncertain,controls=controls,
                  sources={name:file_hash(ROOT/name) for name in pre['sources']},
                  generation_sources=pre['sources'],generation_builder_snapshot=old.name,
                  aggregate_gate_correction='Group paired outcomes within bundle and regime; equal immediate-command questions may recur across inspection-cost regimes and independent bundles. No executions or labels changed.',
                  files={p.name:file_hash(p) for p in output.iterdir() if p.suffix in ('.jsonl','.py')},
                  scope='Authored executed tasks; recovered completed generation after aggregate-only grouping correction. No model training.')
    write_json(output/'manifest.json',manifest);print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-bundles',type=int,default=4)
    p.add_argument('--evaluation-bundles',type=int,default=2)
    p.add_argument('--finalize-existing',action='store_true')
    a=p.parse_args()
    if a.finalize_existing:finalize_existing(a.output)
    else:build(a.output,a.train_bundles,a.evaluation_bundles)
