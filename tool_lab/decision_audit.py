"""Read-only audit of next-stage receipts, paired questions and model readiness."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scale_lab.common import ROOT, digest, encode, file_hash, MODELS, read_rows, write_json
from tool_lab import decision_curriculum as env
from tool_lab.decision_collect import make_questions, qualify


def audit(folder,tokenizer=None):
    if (folder/'REJECTED.json').exists():raise ValueError('This collection was rejected before training')
    frozen=json.loads((folder/'pre-execution-freeze.json').read_text())
    saved=json.loads((folder/'qualification.json').read_text())
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Executed source changed: '+name)
    for name,sha in saved['files'].items():
        if file_hash(folder/name)!=sha:raise ValueError('Executed artifact changed: '+name)
    if file_hash(folder/'cases.jsonl')!=frozen['cases_sha256']:raise ValueError('Fixture freeze changed')
    cases=read_rows(folder/'cases.jsonl');traces=read_rows(folder/'executions.jsonl')
    actual=read_rows(folder/'questions.jsonl');stored_contexts=read_rows(folder/'contexts-private.jsonl')
    rows,contexts=make_questions(cases,traces)
    if rows!=actual or contexts!=stored_contexts:raise ValueError('Questions/targets do not reproduce')
    computed=qualify(cases,traces,rows,contexts)
    if any(saved[k]!=v for k,v in computed.items()):raise ValueError('Qualification accounting differs')
    bycase={c['id']:c for c in cases};byid={t['id']:t for t in traces}
    reference=[];cross_split={}
    for case in cases:
        branches=[t for t in traces if t['case_id']==case['id']]
        item=branches[0]['input'];action=env.continuation(item)
        t=next(t for t in branches if t['plan']=='evidence_then_commit' and t['action']==action)
        expected='unfinished' if case['regime']=='expensive' else 'completed'
        if t['outcome']!=expected:raise ValueError('Public reference continuation failed')
        reference.append(t)
        if case['group_id'] in cross_split and cross_split[case['group_id']]!=case['split']:
            raise ValueError('Related tasks cross splits')
        cross_split[case['group_id']]=case['split']
    # Every exported label has exact source receipt identities and hashes.
    for row in rows:
        if row['receipt_sha256']!=[digest(byid[i]) for i in row['receipt_ids']]:raise ValueError('Label lineage changed')
        if set(row['input'])!={'state','question','options'}:raise ValueError('Model input allowlist failed')
        state=json.loads(row['input']['state'])
        if any(k in state for k in ('outcome','expected','world','label','case_id','split','reward')):
            raise ValueError('Private metadata in public state')
    # Keep unseen-family ownership even if wording/goal/cost variants multiply.
    family_ownership={f:sorted({c['split'] for c in cases if c['family']==f}) for f in env.OWNERS}
    if family_ownership!={f:[s] for f,s in env.OWNERS.items()}:raise ValueError('Family holdout failed')
    metrics={}
    for family in env.OWNERS:
        values=[t for t in reference if t['family']==family]
        metrics[family]=dict(episodes=len(values),completed=sum(t['outcome']=='completed' for t in values),
            mean_reward=sum(t['reward'] for t in values)/len(values))
    # On fresh evidence quartets, neither goal nor observations alone can select
    # the right target. Report the best possible lookup-table shortcut accuracy.
    fresh={c['id'] for c in contexts if c['regime']=='fresh'}
    ablations={}
    for mode in ('menu_only','goal_without_observations','observations_without_goal'):
        groups=defaultdict(list)
        for row in rows:
            if row['kind']!='decision' or row['context_id'] not in fresh:continue
            item=row['input'];state=json.loads(item['state'])
            if mode=='menu_only':key=item['options']
            elif mode=='goal_without_observations':
                state['observations']=[];key=[state,item['options']]
            else:
                state['task']='[goal removed]';key=[state,item['options']]
            groups[digest(key)].append(set(row['target']['option_ids']))
        successes=sum(max(sum(a in targets for targets in values) for a in set.union(*values)) for values in groups.values())
        denominator=sum(map(len,groups.values()))
        ablations[mode]=dict(contexts=denominator,maximum_accuracy=successes/denominator)
        if successes/denominator>.5:raise ValueError('Fresh-state shortcut gate failed: '+mode)
    tokens=None
    if tokenizer is not None:
        lengths=[len(encode(tokenizer,r['input'],4096)) for r in rows]
        actor_inputs={e['input_sha256']:e['input'] for t in traces for e in t['events']}
        actor_lengths=[len(encode(tokenizer,item,4096)) for item in actor_inputs.values()]
        tokens=dict(questions=len(rows),maximum=max(lengths),over_1536=sum(n>1536 for n in lengths),
                    over_2048=sum(n>2048 for n in lengths),maximum_options=max(len(r['input']['options']) for r in rows),
                    by_kind={k:max(n for n,r in zip(lengths,rows) if r['kind']==k) for k in {r['kind'] for r in rows}},
                    unique_actor_histories=len(actor_inputs),maximum_actor_history=max(actor_lengths),
                    actor_histories_over_1536=sum(n>1536 for n in actor_lengths),
                    note='Readiness audit only; no truncation and no model inference. A new token limit requires its own launch freeze.')
    return dict(status='passed',collection_qualification_sha256=file_hash(folder/'qualification.json'),
                auditor_source_sha256=file_hash(Path(__file__)),reference_controls=metrics,
                fresh_context_shortcut_ceilings=ablations,
                audit_dependencies_sha256={n:file_hash(ROOT/n) for n in
                    ('scale_lab/common.py','tool_lab/contextual_shell_audit.py','tool_lab/decision_curriculum.py','tool_lab/decision_collect.py')},
                questions_reconstructed=len(rows),family_ownership=family_ownership,token_audit=tokens,
                prepared_by_split={s:dict(Counter(r['kind'] for r in rows if r['split']==s)) for s in set(env.OWNERS.values())},
                trained_questions=0,optimizer_steps=0,model_inference=False,
                limits='Four authored mechanisms and one fixture root per family in the initial qualification. '
                       'Historical contradictions have explicitly obsolete provenance; noisy current-source reliability is not yet covered.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenize',action='store_true');a=p.parse_args()
    tokenizer=None
    if a.tokenize:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],
                                                local_files_only=True,token=False)
    result=audit(a.folder,tokenizer);write_json(a.output,result);print(json.dumps(result,indent=2))
