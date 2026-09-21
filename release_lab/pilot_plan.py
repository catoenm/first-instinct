"""Prospective, bounded continuation of the already admitted release mixture."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from scale_lab.common import ROOT, MODELS, digest, file_hash, read_rows, write_json, write_rows

PARENT = '882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'
INITIAL = '17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27'
CONFIG = dict(seed=20260922, max_steps=160, eval_every=40, patience=2,
              learning_rate=1e-5, weight_decay=.01, grad_clip=1.,
              micro_batch=2, eval_batch=4, max_tokens=4096, max_seconds=10800,
              per_step={'general':32, 'tools':28, 'verified':4},
              general_development_per_task=32, max_verified_visits=2,
              tool_accuracy_gain=.03, retention_accuracy_drop=.01,
              retention_log_loss_increase=.02, slice_accuracy_drop=.03,
              probability_tolerance=1e-6, restart_after_step=1)


def pool(row):
    names = {r['pool'] for r in row['source_refs']}
    if names == {'general_train'}:
        return 'general'
    if names == {'tools_train'}:
        return 'tools'
    if names <= {'forecasts_train_candidate', 'appworld_new_forecast', 'appworld_new_decision', 'retail'}:
        return 'verified'
    raise ValueError('Unexpected training provenance: '+str(names))


def schedule(rows, config=CONFIG):
    """No target-based sampling; exact order is frozen before model scores."""
    rng = random.Random(config['seed'])
    pools = defaultdict(list)
    for row in rows:
        if row['role'] != 'train':
            raise ValueError('Only training roles can enter the schedule')
        pools[pool(row)].append(row)
    streams = {}
    for name in ('general', 'tools'):
        indices = sorted(r['id'] for r in pools[name])
        rng.shuffle(indices)
        required = config['per_step'][name] * config['max_steps']
        if len(indices) < required:
            raise ValueError('Insufficient unique questions: '+name)
        streams[name] = indices[:required]
    families = defaultdict(list)
    for r in pools['verified']:
        families[r['family']].append(r['id'])
    queues = {}
    for family in sorted(families):
        queues[family] = []
        for _ in range(config['max_verified_visits']):
            visit = sorted(families[family]); rng.shuffle(visit)
            queues[family].extend(visit)
    verified = []
    while len(verified) < config['per_step']['verified'] * config['max_steps']:
        active = sorted(k for k,v in queues.items() if v)
        if not active:
            raise ValueError('Verified repetition ceiling exhausted')
        rng.shuffle(active)
        for k in active:
            verified.append(queues[k].pop(0))
    streams['verified'] = verified
    steps = []
    for i in range(config['max_steps']):
        ids = [ident for name,n in config['per_step'].items() for ident in streams[name][i*n:(i+1)*n]]
        # Length grouping within an update only; the update's weights do not change.
        steps.append(ids)
    return steps


def slices(task):
    result = []
    if task.startswith(('task638_', 'task573_', 'task575_')):
        result.append('dialogue_intent')
    if task.startswith(('verified_access_control_', 'verified_ordered_rules_')):
        result.append('supplied_rules')
    if task.startswith(('verified_evidence_consistency_', 'verified_propositional_logic_')):
        result.append('evidence_judgments')
    if task.startswith('verified_incident_priority_'):
        result.append('incident_priority')
    return result


def prepare(out):
    source = ROOT/'output/release-mixture-v1'
    admission = json.loads((source/'data-admission.json').read_text())
    assembly = json.loads((source/'assembly.json').read_text())
    if admission['status'] != 'qualified_data_pack_not_training_runtime' or admission['assembly_sha256'] != file_hash(source/'assembly.json'):
        raise ValueError('Missing admitted source pack')
    for name in ('train-private.jsonl','development-private.jsonl'):
        if file_hash(source/name) != assembly['files'][name]:
            raise ValueError('Changed admitted source: '+name)
    all_train = read_rows(source/'train-private.jsonl')
    steps = schedule(all_train)
    selected = {ident for step in steps for ident in step}
    train = [dict(r, learning_pool=pool(r)) for r in all_train if r['id'] in selected]
    development = read_rows(source/'development-private.jsonl')
    tool_meta = {r['id']:r['servers'] for r in read_rows(ROOT/'output/release-tool-data-v1/development-private.jsonl')}
    tool_manifest = json.loads((ROOT/'output/release-tool-data-v1/manifest-private.json').read_text())
    tool_path = ROOT/'output/release-tool-data-v1/development-private.jsonl'
    if file_hash(tool_path) != tool_manifest['outputs'][tool_path.name]:
        raise ValueError('Changed tool grouping source')
    general = defaultdict(list); evaluation = []
    for r in development:
        ref = r['source_refs'][0]; name = ref['pool']
        if name == 'general_validation':
            general[r['task']].append(r)
        else:
            if name == 'tools_development':
                suite,groups = 'tools',tool_meta[ref['row_id']]
            elif r['target_contract'] == 'categorical_distribution':
                suite,groups = 'outcomes',[name]
            elif name == 'appworld_development_decision':
                suite,groups = 'application_decisions',[r['task']]
            else:
                raise ValueError('Unexpected development pool')
            evaluation.append(dict(r, suite=suite, metric_groups=groups, slices=[]))
    for task,rows in sorted(general.items()):
        rows = sorted(rows,key=lambda r:digest([CONFIG['seed'],'development',r['id']]))
        for r in rows[:CONFIG['general_development_per_task']]:
            evaluation.append(dict(r,suite='general',metric_groups=[task],slices=slices(task)))
    out.mkdir(parents=True,exist_ok=False)
    write_rows(out/'train.jsonl',train); write_rows(out/'development.jsonl',evaluation)
    write_json(out/'schedule.json',steps)
    index = {r['id']:r for r in train}
    presentations = Counter(x for step in steps for x in step)
    census = dict(prepared_release_questions=admission['by_role']['train']['unique_token_questions'],
        scheduled_presentations=sum(presentations.values()), scheduled_unique_questions=len(presentations),
        scheduled_tokens=sum(len(index[x]['input_ids'])*n for x,n in presentations.items()),
        by_pool={p:dict(unique=sum(r['learning_pool']==p for r in train),
                       presentations=sum(n for x,n in presentations.items() if index[x]['learning_pool']==p)) for p in CONFIG['per_step']},
        by_verified_family=dict(Counter(index[x]['family'] for step in steps for x in step if index[x]['learning_pool']=='verified')),
        evaluation_by_suite=dict(Counter(r['suite'] for r in evaluation)),
        evaluation_slices=dict(Counter(s for r in evaluation for s in r['slices'])),
        training_presentations=0,optimizer_steps=0)
    write_json(out/'census.json',census)
    token_config = json.loads((ROOT/'output/appworld-supervised-v2-data/freeze.json').read_text())
    sources = ['release_lab/__init__.py','release_lab/pilot_plan.py','release_lab/pilot.py',
               'release_lab/pilot_metrics.py','release_lab/pilot_state.py','release_lab/objectives.py',
               'scale_lab/__init__.py','scale_lab/common.py','scale_lab/model.py',
               'tests/test_release_pilot.py','docs/release-pilot-v1-protocol.md',
               'requirements-scale-cuda.txt','requirements-monitor.txt']
    freeze = dict(version='release-pilot-v1',config=CONFIG,model=MODELS['qwen35-9b'],
        parent_adapter_sha256=PARENT,initial_trainable_sha256=INITIAL,
        label_token_ids=token_config['label_token_ids'],pad_id=token_config['pad_id'],
        source_assembly_sha256=file_hash(source/'assembly.json'),
        source_admission_sha256=file_hash(source/'data-admission.json'),
        files={p.name:file_hash(p) for p in sorted(out.iterdir())},
        sources={n:file_hash(ROOT/n) for n in sources},census=census,
        status='cpu_qualified_gpu_runtime_checks_required')
    write_json(out/'freeze.json',freeze)
    return census


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().output),indent=2))
