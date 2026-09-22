"""Verify a public repair contract against existing receipts and exact labels."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab import repair_contract as contract
from tool_lab.paired_curriculum import rotate, require

PARENT = ROOT/'output/paired-curriculum-v1-qualified'
EXECUTIONS = ROOT/'output/decision-curriculum-v3-qualified'
PARENT_FREEZE = '98aeda2f4ed345d2becb9911d2679d907e9b406499b86b340778740127a8e2b5'


def selected_rows(path):
    with path.open() as stream:
        return [r for line in stream if (r := json.loads(line))['family'] in contract.CONTRACTS]


def measurements(family, before, visible):
    """Read saved snapshots solely for the auditor; never append these to model input."""
    if family == 'config':
        base, overlay, queues = [json.loads(before[n]['text']) for n in
                                ('base.json', 'overlay.json', 'queues.json')]
        return {'measurements': [dict(entity=name, metric=queues[name]/
            overlay.get(name, {}).get('workers', base[name]['workers']))
            for name in visible['entities'].values()]}
    tables = before['billing.db']['tables']
    values = []
    for invoice in sorted(tables['invoices']):
        if invoice[1] != 'draft':
            continue
        total = sum(line[2]*line[3] for line in tables['lines'] if line[1] == invoice[0])
        values.append(dict(entity=invoice[0], stored_subtotal=invoice[3], true_subtotal=total,
                           metric=abs(invoice[3]-total)))
    return {'measurements': values}


def verify_target_effect(trace, action):
    """Check the commanded target's actual effect, even for a goal-incorrect write."""
    before, after = trace['before'], trace['after']
    s = contract.state(trace['input'])
    path = 'overlay.json' if s['family'] == 'config' else 'billing.db'
    mutable = {'write.lock', 'offline', 'prepared'}
    require(set(after)-set(before) <= mutable, 'Unexpected new file')
    for name in set(before)-mutable-{path}:
        require(after.get(name) == before[name], 'Protected file changed')
    if action is None:
        require(after[path] == before[path], 'Nonwriting branch changed task data')
        return
    chosen = s['entities'][action[-1]]
    if s['family'] == 'config':
        expected = json.loads(before[path]['text'])
        base = json.loads(before['base.json']['text'])
        count = expected.get(chosen, {}).get('workers', base[chosen]['workers'])
        expected.setdefault(chosen, {})['workers'] = count+1
        require(json.loads(after[path]['text']) == expected, 'Configuration repair effect differs')
    else:
        old, new = before[path], after[path]
        require(old['schema'] == new['schema'] and new['integrity'] == 'ok', 'Database schema or integrity changed')
        expected = deepcopy(old['tables'])
        require(sum(r[0] == chosen for r in expected['invoices']) == 1, 'Missing selected invoice')
        fees = 'total=subtotal+shipping-discount.' in s['task']
        require(fees or 'total=subtotal.' in s['task'], 'Unknown total rule')
        for r in expected['invoices']:
            if r[0] == chosen:
                r[3] = sum(line[2]*line[3] for line in expected['lines'] if line[1] == chosen)
                r[4] = r[3]+(r[5]-r[6] if fees else 0)
        require(new['tables'] == expected, 'Database repair effect differs')


def audit_trace(case, trace):
    from tool_lab.decision_collect import reconstruct
    commands = reconstruct(case, trace)
    counters = Counter()
    write = None
    for index, event in enumerate(trace['events']):
        predicted = contract.immediate(event['input'], event['action'])
        observation = event['observation']
        code = None if observation is None else observation['returncode']
        require(predicted['returncode'] == code and predicted['cost'] == event['cost'] and
                predicted['terminal'] == event['terminal'], 'Public control mechanics disagree with receipt')
        counters[event['action']+':'+str(code)] += 1
        if predicted['irreversible_write']:
            require(write is None and index == len(trace['events'])-1, 'Nonterminal or repeated write')
            write = event['action']
        if event['action'] == 'summary' and code == 0:
            expected = measurements(trace['family'], trace['before'], contract.state(event['input']))
            require(json.loads(observation['stdout']) == expected, 'Summary measurements differ')
        if index+1 < len(trace['events']):
            next_state = contract.state(trace['events'][index+1]['input'])
            flags = {k: next_state[k] for k in predicted['after_flags']}
            require(next_state['decisions_remaining'] == predicted['remaining'], 'Turn accounting differs')
        else:
            flags = dict(write_lock='write.lock' in trace['after'], prepared='prepared' in trace['after'],
                         inspection_offline='offline' in trace['after'])
        require(flags == predicted['after_flags'], 'Control-state effect differs')
    for name in ('write.lock', 'offline'):
        if name in trace['after']:
            require(trace['after'][name] == trace['before'][name], 'Control marker unexpectedly changed')
    if 'prepared' in trace['after']:
        require(trace['after']['prepared'].get('text') == 'ready', 'Preparation effect differs')
    verify_target_effect(trace, write)
    return commands, counters


def qualify(output):
    from transformers import AutoTokenizer
    from tool_lab.decision_collect import make_questions
    from tool_lab.decision_sources import check_general_pools
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    sources = ['tool_lab/repair_contract.py', 'tool_lab/repair_contract_qualify.py',
               'tests/test_repair_contract.py', 'docs/repair-contract-v1-protocol.md',
               'tool_lab/report_contract.py', 'tool_lab/paired_curriculum.py', 'scale_lab/common.py']
    inputs = [PARENT/n for n in ('preparation-freeze.json', 'summary.json', 'canonical-private.jsonl', 'presentations-private.jsonl')]
    inputs += [EXECUTIONS/n for n in ('pre-execution-freeze.json', 'qualification.json', 'cases.jsonl', 'executions.jsonl', 'questions.jsonl')]
    write_json(output/'preparation-freeze.json', dict(version=contract.VERSION,
        sources={n: file_hash(ROOT/n) for n in sources},
        inputs={str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
        parent_freeze_sha256=PARENT_FREEZE, contract_sha256={k: digest(v) for k, v in contract.CONTRACTS.items()},
        canonical_cap=704, presentation_cap=2244, output_bytes_cap=100_000_000,
        training_admitted=False))
    try:
        require(file_hash(PARENT/'preparation-freeze.json') == PARENT_FREEZE, 'Wrong candidate package')
        parent = json.loads((PARENT/'summary.json').read_text())
        require(parent['status'] == 'qualified_candidate_pairing_and_position_index', 'Unqualified candidate parent')
        for name, sha in parent['files'].items():
            require(file_hash(PARENT/name) == sha, 'Candidate artifact changed')
        frozen = json.loads((PARENT/'preparation-freeze.json').read_text())
        for name, sha in {**frozen['sources'], **frozen['inputs']}.items():
            require(file_hash(ROOT/name) == sha, 'Candidate source binding changed')
        qualified = json.loads((EXECUTIONS/'qualification.json').read_text())
        executor_freeze = json.loads((EXECUTIONS/'pre-execution-freeze.json').read_text())
        require(file_hash(EXECUTIONS/'pre-execution-freeze.json') == qualified['pre_execution_freeze_sha256'], 'Execution freeze changed')
        for name, sha in executor_freeze['sources'].items():
            require(file_hash(ROOT/name) == sha, 'Frozen executor changed')
        for name in ('cases.jsonl', 'executions.jsonl', 'questions.jsonl'):
            require(file_hash(EXECUTIONS/name) == qualified['files'][name], 'Execution artifact changed')
        cases = {r['id']: r for r in selected_rows(EXECUTIONS/'cases.jsonl')}
        traces = {r['id']: r for r in selected_rows(EXECUTIONS/'executions.jsonl')}
        branches = Counter(); commands = Counter(); transitions = Counter(); mechanics = Counter()
        for t in traces.values():
            n, counts = audit_trace(cases[t['case_id']], t)
            family = t['family']; branches[family] += 1; commands[family] += n
            transitions[family] += sum(counts.values()); mechanics.update({family+'/'+k: v for k, v in counts.items()})
        rebuilt, _ = make_questions(list(cases.values()), list(traces.values()))
        original_questions = {r['id']: r for r in selected_rows(EXECUTIONS/'questions.jsonl')}
        require({r['id']: r for r in rebuilt} == original_questions, 'Original question reconstruction differs')
        registry_path = ROOT/'output/decision-source-registry-v1-qualified/train_candidate-forecasts.jsonl'
        registry = {r['id']: r for r in selected_rows(registry_path)}
        model = MODELS['qwen35-9b']
        tok = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], local_files_only=True, trust_remote_code=False)
        old_tokens = {r['token_sha256'] for r in read_rows(PARENT/'presentations-private.jsonl')}
        rows = selected_rows(PARENT/'canonical-private.jsonl')
        prepared = []; presentations = []; token_set = set(); paired = 0
        for row in rows:
            if row['source'] == 'source_registry':
                source = registry[row['source_question_id']]; vectors = []
                require(digest(source) == row['source_row_sha256'] and source['input'] == row['input'], 'Forecast source changed')
                for member in source['source_members']:
                    t = traces[member['question_id']]
                    require(member['source'] == 'shell' and t['forecast_input'] == row['input'] and
                            member['lineage']['receipt_ids'] == [t['id']] and
                            member['lineage']['receipt_sha256'] == [digest(t)], 'Forecast receipt binding differs')
                    vector = [float(o['id'] == t['outcome']) for o in row['input']['options']]
                    require(vector == member['vector'], 'Forecast executed label differs')
                    vectors.append(vector)
                    require(contract.augment(t['forecast_input']) == contract.augment(row['input']), 'Hidden world changes overlay')
                q = [sum(v[i] for v in vectors)/len(vectors) for i in range(len(vectors[0]))]
                require(q == row['supervision']['probabilities'] and len(vectors) == source['source_question_count'], 'Forecast distribution differs')
                paired += len(vectors) > 1
            else:
                require(row['source'] == 'shell', 'Unexpected teacher source')
                source = original_questions[row['source_question_id']]
                require(digest(source) == row['source_row_sha256'] and source['input'] == row['input'], 'Teacher source changed')
                accepted = source['target'].get('option_ids') or [source['target']['option_id']]
                ids = [o['id'] for o in row['input']['options']]
                require(sorted(ids.index(i) for i in accepted) == row['supervision']['indices'], 'Teacher target differs')
            item = contract.augment(row['input'])
            require(contract.original(item) == row['input'], 'Original input changed')
            prepared.append(dict(canonical_id=row['id'], family=row['family'], parent_row_sha256=digest(row),
                original_input_sha256=digest(row['input']), augmented_input=item, supervision=row['supervision']))
            for shift in range(len(item['options'])):
                rotated, label, order = rotate(row, shift)
                augmented = contract.augment(rotated)
                require(contract.original(augmented) == rotated, 'Rotation content changed')
                ids = encode(tok, augmented, 4096); key = digest(ids)
                require(key not in token_set and key not in old_tokens, 'Overlay token collision')
                token_set.add(key)
                presentations.append(dict(id=digest([contract.VERSION, row['id'], order]), canonical_id=row['id'],
                    parent_presentation_id=digest([row['id'], order]), order=order, supervision=label,
                    input_ids=ids, token_sha256=key))
        require(len(prepared) == 704 and len(presentations) == 2244, 'Bounded overlay coverage differs')
        general = check_general_pools([dict(token_sha256=k) for k in token_set])
        write_rows(output/'questions-private.jsonl', prepared)
        write_rows(output/'presentations-private.jsonl', presentations)
        require(read_rows(output/'questions-private.jsonl') == prepared and
                read_rows(output/'presentations-private.jsonl') == presentations, 'Serialization changed overlay')
        report = dict(version=contract.VERSION, status='qualified_public_repair_contract_overlay',
            training_admitted=False, canonical_questions=704, forecast_questions=616, teacher_questions=88,
            cyclic_presentations=2244, maximum_tokens=max(len(r['input_ids']) for r in presentations),
            uncertain_forecasts=sum(r['supervision']['semantics'] == 'outcome_distribution' and
                sum(p > 0 for p in r['supervision']['probabilities']) > 1 for r in rows),
            existing_branches_audited=dict(branches), existing_commands_reconstructed=dict(commands),
            existing_actor_transitions_audited=dict(transitions), transition_counts=dict(mechanics),
            compatible_world_groups_with_identical_overlay=paired, labels_unchanged=True,
            original_inputs_exactly_preserved=True, contract_constant_within_family=True,
            general_overlap_checks=general, truncations=0, new_underlying_tasks=0, new_executed_branches=0,
            new_model_calls=0, optimizer_updates=0, training_presentations_consumed=0,
            seconds=time.monotonic()-started,
            remaining_gates=['current full-mixture split admission', 'loss consumer and prospective learning recipe'])
        report['files'] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'summary.json', report)
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= 100_000_000, 'Output size cap')
        return report
    except BaseException as exc:
        write_json(output/'REJECTED.json', dict(error=type(exc).__name__, detail=str(exc), training_admitted=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(json.dumps(qualify(p.parse_args().output), indent=2))
