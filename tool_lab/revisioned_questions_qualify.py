"""Read-only source, target, public-rendering and tokenizer qualification."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from scale_lab.common import ROOT,MODELS,digest,encode,file_hash,read_rows,write_json,write_rows
from tool_lab.revisioned_questions import load_source,prepare,reverse_options,PARENT
from tool_lab.revisioned_questions_audit import audit


def negative_controls(source,rows):
    soft=next(i for i,r in enumerate(rows) if 'soft_target' in r and max(r['soft_target'])<1)
    decision=next(i for i,r in enumerate(rows) if 'target_indices' in r)
    results=[]
    for mutation in ('outcome','decision','prior','replica','private','question','menu','missing'):
        altered=deepcopy(rows);r=altered[soft]
        if mutation=='outcome':r['soft_target']=[1.,0.,0.]
        elif mutation=='decision':
            d=altered[decision];d['target_indices']=[(d['target_indices'][0]+1)%len(d['option_ids'])]
        elif mutation in ('prior','replica'):
            m=next(iter(r['provenance'].values()))[0]
            m['weight' if mutation=='prior' else 'replica']=.9 if mutation=='prior' else 1
        elif mutation=='private':
            state=json.loads(r['input']['state']);state['visible']['hidden_world']=True;r['input']['state']=json.dumps(state)
        elif mutation=='question':r['input']['question']='What happens after some unspecified later continuation?'
        elif mutation=='menu':r['input']['options'].reverse()
        else:altered.pop()
        try:audit(source,altered)
        except ValueError:results.append(mutation)
        else:raise ValueError('Corruption was accepted: '+mutation)
    return results


def run(args):
    from transformers import AutoTokenizer
    from release_lab.laya_compatibility import inspect,require_native_parity
    from release_lab.laya_input_qualify import formatter
    started=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    names=['tool_lab/revisioned_questions.py','tool_lab/revisioned_questions_audit.py',
           'tool_lab/revisioned_questions_qualify.py','tests/test_revisioned_questions.py',
           'docs/revisioned-questions-v1-protocol.md','scale_lab/common.py','release_lab/laya_compatibility.py',
           'release_lab/laya_input_qualify.py']
    plan=dict(parent_freeze_sha256=PARENT,sources={n:file_hash(ROOT/n) for n in names},
              candidate_questions=138,diagnostic_reversed_presentations=138,new_executions=0,
              model_calls=0,optimizer_presentations=0)
    write_json(args.output/'pre-qualification-freeze.json',plan)
    source=load_source(args.source);rows=prepare(source);reversed_rows=[reverse_options(r) for r in rows]
    original_audit=audit(source,rows);reversed_audit=audit(source,reversed_rows,reversed_menu=True)
    controls=negative_controls(source,rows)
    write_rows(args.output/'questions-private.jsonl',rows)
    write_rows(args.output/'reversed-private.jsonl',reversed_rows)
    # Serialization may not change targets, histories, menus or source identity.
    if audit(source,read_rows(args.output/'questions-private.jsonl'))!=original_audit:raise ValueError('Canonical read-back differs')
    if audit(source,read_rows(args.output/'reversed-private.jsonl'),reversed_menu=True)!=reversed_audit:raise ValueError('Reversed read-back differs')
    asset=ROOT/'.local/laya-baseline-review/assets/laya-typed-decisions'
    checkpoint=json.loads((ROOT/'output/laya-checkpoint-v1-plan.json').read_text())
    for name,record in checkpoint['files'].items():
        if name!='model.safetensors' and file_hash(asset/name)!=record['sha256']:raise ValueError('Tokenizer/config identity differs')
    cfg=json.loads((asset/'rl_agent_config.json').read_text())
    tconfig=json.loads((asset/'tokenizer/tokenizer_config.json').read_text());overrides={}
    if isinstance(tconfig.get('extra_special_tokens'),list):overrides['extra_special_tokens']={f'extra_{i}':s for i,s in enumerate(tconfig['extra_special_tokens'])}
    ltok=AutoTokenizer.from_pretrained(asset/'tokenizer',local_files_only=True,trust_remote_code=False,**overrides)
    qtok=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],local_files_only=True,trust_remote_code=False)
    native=formatter(ROOT/'.local/laya-baseline-review/source/laya/common.py')
    tokens=[];coverage=Counter();loss=Counter()
    for presentation,items in [('canonical',rows),('reversed',reversed_rows)]:
        for row in items:
            item=row['input'];ids=encode(qtok,item,4096);seen=inspect(item,ltok,cfg)
            q=seen['request']['questions']['decision']
            seq,markers=native(ltok,seen['request']['state'],dict(t=q['type'],ins=q['instructions'],crit=q['criteria']),cfg['max_len'],cfg['head_max_len'])
            require_native_parity(seen,seq,markers)
            coverage[presentation]+=bool(seen['full_information'])
            for key,value in seen['dropped_tokens'].items():loss[key]+=int(value>0)
            tokens.append(dict(id=row['id'],presentation=presentation,task=row['task'],qwen_input_ids=ids,
                               qwen_token_sha256=digest(ids),qwen_tokens=len(ids),laya_sequence_sha256=digest(seq),
                               laya_tokens=len(seq),laya_full_information=seen['full_information'],laya_dropped=seen['dropped_tokens']))
    write_rows(args.output/'tokens-private.jsonl',tokens)
    if len({r['qwen_token_sha256'] for r in tokens})!=276:raise ValueError('Duplicate actual token presentations')
    summary=dict(status='qualified_qwen_question_candidates',canonical_questions=len(rows),reversed_diagnostic_presentations=len(reversed_rows),
        questions_by_kind=dict(Counter(r['task'] for r in rows)),outcome_questions=sum('soft_target' in r for r in rows),
        acceptable_set_questions=sum('target_indices' in r for r in rows),uncertain_outcome_questions=original_audit['uncertain_outcome_questions'],
        existing_fixture_roots=1,existing_initial_databases=1,existing_post_schedule_states=2,existing_world_goal_tasks=4,
        existing_mechanisms=1,ownership_groups=len({r['group_id'] for r in rows}),initial_visible_histories=len({r['visible_history_sha256'] for r in rows}),
        existing_primary_branches_used=original_audit['unique_primary_source_branches'],source_replays_audited=96,replay_probability_mass=0,
        new_tasks=0,new_executions=0,new_model_calls=0,admitted_training_questions=0,optimizer_presentations=0,
        qwen_complete_presentations=len(tokens),qwen_max_tokens=max(r['qwen_tokens'] for r in tokens),
        laya_complete_presentations=dict(coverage),laya_loss_presentations=dict(loss),laya_max_tokens=max(r['laya_tokens'] for r in tokens),
        native_formatter_parities=len(tokens),negative_controls_rejected=controls,seconds=time.monotonic()-started,
        limits='One authored mechanism; fixed continuations/finite procedure comparisons, not unrestricted optimal actions. Candidates have not passed mixture admission or trained any model.')
    write_json(args.output/'summary.json',summary)
    write_json(args.output/'audit.json',dict(status='passed',canonical=original_audit,reversed=reversed_audit,negative_controls=controls))
    write_json(args.output/'freeze.json',dict(status=summary['status'],parent_freeze_sha256=PARENT,sources=plan['sources'],
        tokenizer_model=MODELS['qwen35-9b'],laya_checkpoint_metadata_sha256=file_hash(ROOT/'output/laya-checkpoint-v1-plan.json'),
        files={p.name:file_hash(p) for p in args.output.iterdir() if p.is_file() and p.name!='freeze.json'}))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
