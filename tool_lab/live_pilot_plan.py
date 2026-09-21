"""Prospective bounded mixed-tool comparison; no release or hardware allocation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.expanded_curriculum import TRAIN_FAMILIES
from tool_lab.retail_matched import balanced_block

VERSION = 'live-tools-pilot-v1'
SEEDS = (20260924, 20260925)
ARMS = ('outcome', 'reward', 'hybrid')
PARENT = '882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'
INITIAL = '17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27'
RECIPE = dict(max_updates=24, batch_size=2, learning_rate=2e-7, value_learning_rate=1e-4,
    value_weight=.5, entropy_weight=.01, replay_weight=1., forecast_weight=.25,
    clip=.2, max_kl=.02, max_individual_kl=.10, max_tokens=4096,
    eval_every=8, patience=2, min_updates=16, exploration_floor=.2,
    shell_episodes_per_family=2, retail_episodes=6, forecasts_per_family=4,
    retail_forecasts=8, replay_rows=64)


def safety(metrics, baseline):
    return (metrics['retention']['macro_accuracy'] >= baseline['retention']['macro_accuracy']-.01 and
        metrics['retention']['macro_log_loss'] <= baseline['retention']['macro_log_loss']+.02 and
        metrics['reward'] >= baseline['reward']-.02 and
        metrics['forecast']['macro']['expected_brier'] <= baseline['forecast']['macro']['expected_brier']+.02)


def improvement(metrics, baseline):
    return (safety(metrics, baseline) and metrics['reward'] >= baseline['reward']+.03 and
        metrics['forecast']['macro']['expected_brier'] <= baseline['forecast']['macro']['expected_brier']-.02)


def schedule(cases, forecasts, replay, seed):
    if seed not in SEEDS or {c['family'] for c in cases} != set(TRAIN_FAMILIES):
        raise ValueError('Wrong paired seed or mechanism ownership')
    by_case, by_forecast = {}, {}
    for family in TRAIN_FAMILIES:
        regimes = defaultdict(list)
        for c in cases:
            if c['family'] == family:
                if c['split'] not in ('train', 'train_candidate'):
                    raise ValueError('Held-out world in training')
                regimes[c['regime']].append(c['id'])
        for regime, values in sorted(regimes.items()):
            values.sort();random.Random(f'{seed}:case:{family}:{regime}').shuffle(values)
        names = sorted(regimes);random.Random(f'{seed}:regime:{family}').shuffle(names)
        by_case[family] = [regimes[r][i] for i in range(max(map(len, regimes.values()))) for r in names if i < len(regimes[r])]
    for row in forecasts:
        if row['role'] != 'train' or row['family'] not in set(TRAIN_FAMILIES) | {'retail_workflows'}:
            raise ValueError('Wrong forecast ownership')
        by_forecast.setdefault(row['family'], []).append(row['id'])
    for family, ids in by_forecast.items():
        ids.sort();random.Random(f'{seed}:forecast:{family}').shuffle(ids)
    if set(by_forecast) != set(TRAIN_FAMILIES) | {'retail_workflows'}:
        raise ValueError('Missing forecast mechanism')
    block = balanced_block(seed)
    if len(block) != RECIPE['max_updates']*RECIPE['retail_episodes']:
        raise ValueError('Retail world/cost block must be consumed exactly once in a completed reward arm')
    result = []
    for i in range(RECIPE['max_updates']):
        case_ids, forecast_ids = [], []
        for family, ids in by_case.items():
            n = RECIPE['shell_episodes_per_family']
            chosen = [ids[(i*n+j)%len(ids)] for j in range(n)]
            if len(set(chosen)) != n:raise ValueError('Duplicate world in one live collector call')
            case_ids.extend(chosen)
        for family, ids in sorted(by_forecast.items()):
            n = RECIPE['retail_forecasts'] if family == 'retail_workflows' else RECIPE['forecasts_per_family']
            forecast_ids.extend(ids[(i*n+j)%len(ids)] for j in range(n))
        result.append(dict(update=i+1, case_ids=case_ids, retail=block[6*i:6*i+6], forecast_ids=forecast_ids,
            replay_ids=random.Random(f'{seed}:replay:{i}').sample(sorted(r['id'] for r in replay), RECIPE['replay_rows'])))
    return result


def prepared_retail(tokenizer):
    rows, lineage = [], {}
    for name, audit_name, status, hash_key in [
            ('retail-history-v1','independent-audit.json','independently_verified','questions_sha256'),
            ('retail-branches-v1','audit.json','independently_verified','candidate_questions_sha256')]:
        folder = ROOT/'output'/name
        audit = json.loads((folder/audit_name).read_text())
        if audit['status'] != status or file_hash(folder/'questions-private.jsonl') != audit[hash_key]:
            raise ValueError('Retail independent audit or question identity differs')
        lineage[str((folder/audit_name).relative_to(ROOT))] = file_hash(folder/audit_name)
        lineage[str((folder/'questions-private.jsonl').relative_to(ROOT))] = audit[hash_key]
        for r in read_rows(folder/'questions-private.jsonl'):
            if r['group_id'] != 'retail_workflows' or r['role'] not in ('train','training_candidate'):
                raise ValueError('Retail source ownership differs')
            ids = [o['id'] for o in r['input']['options']]
            target = r.get('soft_target') or [r['target'][n] for n in ids]
            rows.append(dict(id=name+':'+r['id'], input=r['input'], input_ids=encode(tokenizer,r['input'],4096),
                option_ids=ids,target_indices=[],soft_target=target,task='executed_consequence',
                forecast_contract='command_then_stop',family='retail_workflows',metric_group='retail_workflows',
                group_id='retail_workflows',role='train',source_row_sha256=digest(r),source_stage=name))
    return rows, lineage


def prepare(output):
    from transformers import AutoTokenizer
    previous = ROOT/'output/expanded-decisions-v1-data'
    frozen = json.loads((previous/'freeze.json').read_text())
    for name, expected in frozen['files'].items():
        if file_hash(previous/name) != expected:raise ValueError('Changed earlier data: '+name)
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'], revision=MODELS['qwen35-9b']['revision'], local_files_only=True)
    retail, lineage = prepared_retail(tokenizer)
    old = read_rows(previous/'train-forecasts.jsonl')
    forecasts = []
    for r in old:
        if r['role'] != 'train_candidate' or r['family'] not in TRAIN_FAMILIES:
            raise ValueError('Old forecast not training eligible')
        if encode(tokenizer, r['input'], 4096) != r['input_ids']:
            raise ValueError('Original rendered forecast changed')
        forecasts.append({**r,'role':'train','forecast_contract':
            'command_then_stop' if r['family']=='reservation' or 'Stop immediately after' in r['input']['question'] else 'displayed_fixed_continuation'})
    # Exact-token deduplication must preserve distributions and whole ownership.
    unique = {};duplicates = []
    for r in forecasts+retail:
        key = digest(r['input_ids'])
        if key in unique:
            prior = unique[key]
            if (r['soft_target'],r['option_ids'],r['family']) != (prior['soft_target'],prior['option_ids'],prior['family']):
                raise ValueError('Conflicting labels or ownership for identical visible question')
            duplicates.append(r['id'])
        else:unique[key] = r
    forecasts = list(unique.values())
    # Only token identities are read here, never model scores from reserved data.
    scans = [ROOT/'output/history-pilot-v1-data/development.jsonl']
    scans += list((ROOT/'output/release-evaluation-v1-data').glob('*jsonl'))
    for path in scans:
        for r in read_rows(path):
            if 'input_ids' in r and digest(r['input_ids']) in unique:
                raise ValueError('Training forecast overlaps development/reserved input: '+str(path))
    output.mkdir(parents=True, exist_ok=False)
    for name in ('train-cases.jsonl','validation-cases.jsonl','replay.jsonl','retention.jsonl',
                 'shell-parity-cases.jsonl','shell-parity-references.jsonl','application-parity-cases.jsonl',
                 'application-parity-references.jsonl','expanded-parity-jobs.jsonl','expanded-parity-references.jsonl'):
        shutil.copyfile(previous/name, output/name)
    write_rows(output/'train-forecasts.jsonl',forecasts)
    dev = read_rows(previous/'validation-forecasts.jsonl')
    write_rows(output/'validation-forecasts.jsonl',dev)
    probes = read_rows(previous/'guard-probes.jsonl')
    for row in probes:
        if row['task'] != 'shell_action' or row['target_indices'] != [0]:raise ValueError('Legacy guard placeholder differs')
        row['target_indices'] = []
    write_rows(output/'guard-probes.jsonl',probes)
    cases = read_rows(output/'train-cases.jsonl'); replay = read_rows(output/'replay.jsonl')
    for seed in SEEDS:write_rows(output/f'schedule-{seed}.jsonl',schedule(cases,forecasts,replay,seed))
    admission = dict(candidate_retail_questions=len(retail),admitted_retail_questions=sum(r['family']=='retail_workflows' for r in forecasts),
        duplicate_ids=duplicates,exact_development_reserved_overlap=0,source_lineage=lineage,
        ownership='All retail workflows are one connected training group; no internal retail holdout.',
        new_model_calls=0,new_optimizer_steps=0,new_executions=0)
    write_json(output/'admission.json',admission)
    sources = [p for folder in ('tool_lab','general_lab','scale_lab','puffer_lab','release_lab') for p in (ROOT/folder).glob('*.py')]
    sources += list((ROOT/'puffer_lab').glob('*.h'))
    sources += [ROOT/n for n in ('tool_lab/calendar_assets/America_New_York.tzif','tool_lab/calendar_assets/provenance.json',
        'requirements-scale-cuda.txt','requirements-monitor.txt','test_general_rl.py','test_retail_live.py',
        'tests/test_retail_actor.py','tests/test_live_contracts.py','tests/test_live_mixed.py','tests/test_live_pilot.py',
        'docs/live-tools-pilot-v1-protocol.md')]
    census = dict(training_case_variants=len(cases),training_shell_mechanisms=len(TRAIN_FAMILIES),retail_goals=3,
        retail_initial_physical_states=10,retail_goal_world_pairs=20,retail_unique_world_goal_cost_cells=120,
        retail_prepared_reset_presentations_per_reward_arm=144,forecast_unique_questions=len(forecasts),
        forecast_by_mechanism=dict(Counter(r['family'] for r in forecasts)),replay_pool=len(replay),
        scheduled_forecast_presentations_per_supervised_arm=24*28,scheduled_replay_presentations_per_arm=24*64,
        scheduled_live_training_episodes_per_reward_arm=24*16,actual_training_presentations=0,actual_optimizer_updates=0)
    result = dict(version=VERSION,recipe=RECIPE,seeds=list(SEEDS),arms=list(ARMS),model=MODELS['qwen35-9b'],
        parent_adapter_sha256=PARENT,initial_trainable_sha256=INITIAL,prior_data_freeze_sha256=file_hash(previous/'freeze.json'),
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},census=census,
        status='prepared_diagnostic_pending_runtime_qualification',release_eligible=False,
        selection='Development report mechanism only; safety stop or two checks without joint +.03 return/-.02 Brier. No reserved scores.',
        transfer='Previously exposed calendar/payment are development only. Fresh independent transfer remains required before promotion.')
    write_json(output/'freeze.json', result)
    return census


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args().output),indent=2))
