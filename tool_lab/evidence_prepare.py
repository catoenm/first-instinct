"""Audit executed outcomes, tokenize without truncation, freeze the pilot."""
import argparse
import base64
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

from scale_lab.common import MODELS, ROOT, digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.contextual_shell_audit import public_expected
from tool_lab.evidence_env import Episode, HORIZON, VERSION, forecast_input, terminal_success
from tool_lab.evidence_train import RECIPE, SEEDS


def audit_trajectory(case, trace):
    ep=Episode.__new__(Episode);ep.case=case;ep.depth=0;ep.history=[]
    for p in trace['prefix']:
        if p['action'] in ('cached','summary'):
            ep.history.append(dict(action='summary',validity=case['initial_evidence'],**p['observation']))
        else:
            ep.history.append(dict(action=p['action'],validity='current',**p['observation']))
    evidence=trace['private_verifier_evidence']
    expected={k:hashlib.sha256(base64.b64decode(v)).hexdigest() for k,v in case['files'].items()}
    if expected!={k:v['sha256'] for k,v in evidence['before'].items()}:raise ValueError('Wrong initial world')
    success=terminal_success(case,evidence['before'],evidence['after'])
    for i,event in enumerate(trace['events']):
        if event['input']!=ep.input() or event['input_sha256']!=digest(ep.input()):raise ValueError('Public input reconstruction differs')
        action,obs=event['action'],event['observation'];reward=0.
        if action!='finish':
            if obs['returncode'] not in (0,75):raise ValueError('Invalid command outcome')
            ep.history.append(dict(action=action,validity='current',**obs))
            reward-=case['read_cost'] if action=='summary' else case['unlock_cost'] if action=='unlock' else case['write_cost']
        wrote=action.startswith('repair_') and obs['returncode']==0
        if wrote:reward+=1. if success else -1.
        ep.depth+=1
        terminal=wrote or action=='finish' or ep.depth==HORIZON
        if abs(reward-event['reward'])>1e-12 or terminal!=event['terminal'] or (terminal and i!=len(trace['events'])-1):
            raise ValueError('Reward or termination reconstruction differs')
    if abs(sum(e['reward'] for e in trace['events'])-trace['reward'])>1e-12:
        raise ValueError('Trajectory return mismatch')
    if trace['done'] and success!=trace['success']:raise ValueError('Terminal verifier mismatch')
    return len(trace['events'])


def audit(raw):
    manifest=json.loads((raw/'manifest.json').read_text())
    if manifest['status']!='qualified':raise ValueError('Data did not qualify')
    for name,sha in manifest['files'].items():
        if file_hash(raw/name)!=sha:raise ValueError('Changed raw evidence')
    for name,sha in manifest['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Changed execution source')
    for name,sha in manifest.get('generation_sources',{}).items():
        path=raw/manifest['generation_builder_snapshot'] if name=='tool_lab/evidence_data.py' else ROOT/name
        if file_hash(path)!=sha:raise ValueError('Changed original generation source')
    checked=decisions=controls=0
    for split in ('train','validation','test'):
        cases={c['id']:c for c in read_rows(raw/(split+'-cases.jsonl'))}
        for c in cases.values():
            if c['base']['expected']!=public_expected(c['base']):raise ValueError('Independent semantic oracle differs')
        for trace in read_rows(raw/(split+'-controls.jsonl')):
            decisions+=audit_trajectory(cases[trace['id']],trace);controls+=1
        rows={r['id']:r for r in read_rows(raw/(split+'-forecasts.jsonl'))}
        for r in read_rows(raw/(split+'-executions.jsonl')):
            row=rows.pop(r['id']);case=cases[r['case_id']];trace=r['trajectory']
            event=trace['events'][-1];evidence=trace['private_verifier_evidence']
            expected=terminal_success(case,evidence['before'],evidence['after'])
            decisions+=audit_trajectory(case,trace)
            if (row['execution_sha256']!=digest(r) or row['outcome']!=expected or r['success']!=expected
                or row['input']!=forecast_input(event['input'],r['action']) or r['input_sha256']!=digest(row['input'])):
                raise ValueError('Forecast does not match an executed branch')
            checked+=1
        if rows:raise ValueError('Missing executions')
    return dict(status='passed',independent_forecast_verifications=checked,control_episodes=controls,
                public_decisions_reconstructed=decisions,
                raw_manifest_sha256=file_hash(raw/'manifest.json'))


def prepare_data(raw,output,adapter):
    from transformers import AutoTokenizer
    from general_lab.rl import prepare
    from general_lab.train import validation_subset
    verified=audit(raw)
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],token=False,local_files_only=True)
    maximum=0
    for split in ('train','validation','test'):
        rows=[]
        for r in read_rows(raw/(split+'-forecasts.jsonl')):
            row=prepare(tokenizer,r['input'],r['id'],RECIPE['max_tokens'],r['target']['option_id'])
            row.update({k:r[k] for k in ('group_id','case_id','family','regime','phase','task','outcome','execution_sha256')})
            maximum=max(maximum,len(row['input_ids']));rows.append(row)
        write_rows(output/(split+'-forecasts.jsonl'),rows)
        shutil.copyfile(raw/(split+'-cases.jsonl'),output/(split+'-cases.jsonl'))
        # A conservative long-history bound using actual measurement outputs.
        bycase={}
        for r in read_rows(raw/(split+'-executions.jsonl')):
            if r['phase']=='unlocked':bycase[r['case_id']]=r
        for c in read_rows(raw/(split+'-cases.jsonl')):
            e=Episode.__new__(Episode);e.case=c;e.depth=HORIZON-1
            state=json.loads(bycase[c['id']]['trajectory']['events'][-1]['input']['state'])
            obs=next(h for h in state['observations'] if h['action']=='summary' and h['validity']=='current')
            e.history=[obs]*4
            maximum=max(maximum,len(encode(tokenizer,e.input(),RECIPE['max_tokens'])))
    general=ROOT/'output/general-qwen35-9b-v2'
    gm=json.loads((general/'manifest.json').read_text())
    for split in ('train','validation'):
        if file_hash(general/(split+'.jsonl'))!=gm['outputs'][split+'.jsonl']:raise ValueError('General source checksum differs')
    pool=read_rows(general/'train.jsonl');random.Random(1507).shuffle(pool)
    replay=pool[:4096];retention=validation_subset(read_rows(general/'validation.jsonl'),4,1507)
    transfer=read_rows(ROOT/'output/contextual-shell-training-v1-data/general-transfer.jsonl')
    if {r['id'] for r in replay}&{r['id'] for r in retention+transfer}:raise ValueError('General split overlap')
    write_rows(output/'replay.jsonl',replay);write_rows(output/'retention.jsonl',retention);write_rows(output/'transfer.jsonl',transfer)
    shutil.copyfile(raw/'manifest.json',output/'raw-manifest.json');write_json(output/'data-audit.json',verified)
    # Retain only a small cross-backend qualification set, not all training labels.
    cases=read_rows(raw/'train-cases.jsonl');selected={}
    for c in cases:selected.setdefault((c['family'],c['regime']),c)
    ids={c['id'] for c in selected.values()}
    from tool_lab.evidence_data import write_cases
    write_cases(output/'parity-cases.jsonl',selected.values())
    write_rows(output/'parity-executions.jsonl',[r for r in read_rows(raw/'train-executions.jsonl') if r['case_id'] in ids])
    sources=[p for folder in ('scale_lab','general_lab','tool_lab') for p in (ROOT/folder).glob('*.py')]
    sources += [ROOT/'requirements-scale-cuda.txt',ROOT/'tests/test_evidence_decisions.py',ROOT/'tests/test_general_rl.py',
                ROOT/'docs/evidence-decisions-v2-protocol.md']
    manifest=json.loads((raw/'manifest.json').read_text())
    frozen=dict(schema=VERSION,recipe=RECIPE,seeds=list(SEEDS),model=MODELS['qwen35-9b'],
        adapter={p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        reference_validation_reward=manifest['controls']['validation/reference']['reward'],
        raw_manifest_sha256=file_hash(raw/'manifest.json'),maximum_checked_tokens=maximum,
        counts={**manifest['counts'],'replay':len(replay),'retention':len(retention),'transfer':len(transfer)},
        selection='validation reward - 0.25*forecast Brier with general retention and per-metric gates; include update zero',
        budget=dict(pilot_usd=60,compute_ceiling_usd=48.6,storage_and_recovery_reserve_usd=11.4,
                    prior_compute_usd_excluding_storage=151.4214039661,cumulative_user_cap_usd=500))
    write_json(output/'freeze.json',frozen)
    print(json.dumps({k:frozen[k] for k in ('schema','maximum_checked_tokens','counts','budget')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','output','adapter'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare_data(a.raw,a.output,a.adapter)
