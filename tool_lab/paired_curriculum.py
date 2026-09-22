"""Index audited paired questions without executing worlds or predicting labels."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, messages, read_rows, write_json, write_rows

VERSION = 'paired-curriculum-v1'
FAMILIES = {'config','sqlite','application_delivery','filesystem_scope','reservation','retail','revisioned_database'}
REGISTRY = ROOT/'output/decision-source-registry-v1-qualified'
SOURCES = {'shell': ROOT/'output/decision-curriculum-v3-qualified',
    'application': ROOT/'output/application-curriculum-v1-replay-qualified',
    'filesystem': ROOT/'output/filesystem-decisions-v1-qualified'}


def require(condition, message):
    if not condition: raise ValueError(message)


def target(ids, *, probabilities=None, acceptable=None, semantics):
    require(len(ids) >= 2 and len(set(ids)) == len(ids), 'Invalid option identities')
    require(semantics in ('outcome_distribution','acceptable_choice_set','decision_distribution'), 'Unknown supervision type')
    if semantics == 'acceptable_choice_set':
        require(probabilities is None and acceptable and set(acceptable) <= set(ids), 'Invalid acceptable-choice target')
        return dict(semantics=semantics, indices=sorted(ids.index(i) for i in set(acceptable)))
    require(acceptable is None and probabilities is not None and set(probabilities) == set(ids), 'Distribution vocabulary differs')
    values = [probabilities[i] for i in ids]
    require(all(type(v) in (float,int) and math.isfinite(v) and v >= 0 for v in values) and
        math.isclose(math.fsum(values),1.,rel_tol=0.,abs_tol=1e-10), 'Invalid probability mass')
    return dict(semantics=semantics, probabilities=values)


def rotate(row, shift):
    n = len(row['input']['options'])
    require(type(shift) is int and 0 <= shift < n, 'Invalid cyclic rotation')
    order = list(range(shift,n))+list(range(shift))
    item = deepcopy(row['input']); item['options'] = [item['options'][i] for i in order]
    label = deepcopy(row['supervision'])
    if label['semantics'] == 'acceptable_choice_set':
        label['indices'] = sorted(order.index(i) for i in label['indices'])
    else:
        label['probabilities'] = [label['probabilities'][i] for i in order]
    return item, label, order


def public_key(item, namespace):
    state = item['state']
    if namespace == 'retail': state = json.loads(state)['context']
    elif namespace.startswith('revisioned'):
        value = json.loads(state)
        state = value.get('visible',value)
        if isinstance(state,dict): state = {k:v for k,v in state.items() if k not in ('options','offered_commands')}
    return digest([namespace,state])


def canonical(source, family, row, path, supervision, *, task, namespace=None):
    require(family in FAMILIES, 'Nontraining mechanism cannot enter candidate index')
    source_role=row.get('role',row.get('split','train_candidate'))
    diagnostic=(source=='revisioned_optimal' and family=='revisioned_database' and source_role=='training_mechanism_diagnostic')
    require(source_role in ('train','train_candidate') or diagnostic, 'Source ownership changed')
    item = deepcopy(row['input']); messages(item)
    require(set(item) == {'state','question','options'}, 'Private fields in public input')
    groups = ([row['group_id']] if 'group_id' in row else
        sorted({r['lineage']['group_id'] for r in row['source_members']}))
    require(groups and all(isinstance(g,str) and g for g in groups), 'Missing source ownership groups')
    return dict(id=digest([VERSION,source,row['id']]), source=source, family=family,
        role=source_role, group_id=digest([VERSION,'whole_mechanism',family]), source_group_ids=groups,
        ownership_component=family,
        source_role=row.get('role',row.get('split')), source_question_id=row['id'],
        source_file=str(path.relative_to(ROOT)), source_row_sha256=digest(row),
        input=item, supervision=supervision, task=task,
        public_evidence_key=public_key(item,namespace or source), training_admitted=False)


def source_inputs():
    paths = [REGISTRY/'report.json',REGISTRY/'preparation-freeze.json',REGISTRY/'source-audits.json',
        REGISTRY/'train_candidate-forecasts.jsonl']
    paths += [p/name for p in SOURCES.values() for name in ('qualification.json','cases.jsonl','questions.jsonl','executions.jsonl')]
    paths += [ROOT/'output/release-retail-questions-v1'/name for name in ('freeze.json','admission.json','usage-private.json','train-private.jsonl')]
    for folder in ('revisioned-questions-v1','revisioned-oracle-v1'):
        paths += [ROOT/'output'/folder/name for name in ('freeze.json','questions-private.jsonl')]
    return paths


def verify_sources():
    report = json.loads((REGISTRY/'report.json').read_text())
    require(report['status'] == 'qualified_source_index', 'Rejected source registry')
    require(file_hash(REGISTRY/'report.json') == file_hash(ROOT/'results/decision-source-registry-v1/report.json'), 'Accepted public registry differs')
    for name,sha in report['files'].items(): require(file_hash(REGISTRY/name)==sha,'Registry artifact changed: '+name)
    frozen = json.loads((REGISTRY/'preparation-freeze.json').read_text())
    for name,sha in frozen['inputs'].items(): require(file_hash(ROOT/name)==sha,'Qualified source changed: '+name)
    # These later sources have their own accepted, immutable question freezes.
    for name in ('revisioned-questions-v1','revisioned-oracle-v1'):
        folder = ROOT/'output'/name; freeze=json.loads((folder/'freeze.json').read_text())
        for filename,sha in freeze['files'].items(): require(file_hash(folder/filename)==sha,'Later source artifact changed')
        for filename,sha in freeze['sources'].items(): require(file_hash(ROOT/filename)==sha,'Later qualified source changed')
    folder = ROOT/'output/release-retail-questions-v1'
    admission = json.loads((folder/'admission.json').read_text())
    usage = json.loads((folder/'usage-private.json').read_text())
    freeze = json.loads((folder/'freeze.json').read_text())
    require(admission['status']=='qualified_paired_questions' and usage['status']=='qualified_training_only' and usage['role']=='train','Retail admission not accepted')
    require(file_hash(folder/'freeze.json')==admission['freeze_sha256'] and file_hash(folder/'admission.json')==usage['admission_sha256'],'Retail admission binding changed')
    for filename,sha in admission['files'].items(): require(file_hash(folder/filename)==sha,'Admitted retail questions changed')
    require({r['id']:digest(r) for r in read_rows(folder/'train-private.jsonl')}==usage['row_sha256'],'Retail row usage changed')
    for filename,sha in freeze['paths'].items(): require(file_hash(Path(filename))==sha,'Retail qualified source changed')
    return dict(registry=report,retail_admission=admission,retail_usage=usage)


def build():
    rows=[]
    path=REGISTRY/'train_candidate-forecasts.jsonl'
    for r in read_rows(path):
        ids=[o['id'] for o in r['input']['options']]
        label=target(ids,probabilities=dict(zip(ids,r['soft_target'])),semantics='outcome_distribution')
        namespaces={m['source'] for m in r['source_members']}
        require(len(namespaces)==1,'Forecast crosses source namespaces')
        rows.append(canonical('source_registry',r['family'],r,path,label,task='explicit_question_outcome',namespace=next(iter(namespaces))))
    for source,folder in SOURCES.items():
        path=folder/'questions.jsonl'
        for r in read_rows(path):
            if r['family'] not in FAMILIES or r['kind']=='forecast': continue
            require(r['kind'] in ('decision','observation_value'),'Unexpected legacy teacher question')
            ids=[o['id'] for o in r['input']['options']]
            acceptable=r['target'].get('option_ids') or [r['target']['option_id']]
            label=target(ids,acceptable=acceptable,semantics='acceptable_choice_set')
            rows.append(canonical(source,r['family'],r,path,label,task=r['kind']+'_fixed_continuation'))
    path=ROOT/'output/release-retail-questions-v1/train-private.jsonl'
    for r in read_rows(path):
        ids=[o['id'] for o in r['input']['options']]
        outcome=r['task'] in ('immediate_success','continued_success')
        require(outcome or r['task'] in ('next_procedure','observation_value'),'Unknown retail task')
        label=target(ids,probabilities=r['target']['probabilities'],semantics='outcome_distribution' if outcome else 'decision_distribution')
        rows.append(canonical('retail','retail',r,path,label,task=r['task'],namespace='retail'))
    for source,folder in [('revisioned_fixed','revisioned-questions-v1'),('revisioned_optimal','revisioned-oracle-v1')]:
        path=ROOT/'output'/folder/'questions-private.jsonl'
        for r in read_rows(path):
            ids=[o['id'] for o in r['input']['options']]
            if 'soft_target' in r:
                label=target(ids,probabilities=dict(zip(ids,r['soft_target'])),semantics='outcome_distribution')
            else:
                label=target(ids,acceptable=[ids[i] for i in r['target_indices']],semantics='acceptable_choice_set')
            rows.append(canonical(source,'revisioned_database',r,path,label,task=r['task']))
    require(len(rows)<=7000 and len({r['id'] for r in rows})==len(rows),'Candidate question cap/identity collision')
    rendered={}
    for r in rows:
        key=digest(messages(r['input']))
        require(key not in rendered,'Canonical rendered collision needs source reconciliation')
        rendered[key]=r['id']
    return rows


def prepare(output):
    from transformers import AutoTokenizer
    output.mkdir(parents=True,exist_ok=False)
    paths=source_inputs()
    freeze=dict(version=VERSION,sources={str(p.relative_to(ROOT)):file_hash(p) for p in
        [Path(__file__),ROOT/'tests/test_paired_curriculum.py',ROOT/'docs/paired-curriculum-v1-protocol.md']},
        inputs={str(p.relative_to(ROOT)):file_hash(p) for p in paths},model=MODELS['qwen35-9b'],
        canonical_cap=7000,presentation_cap=30000,output_bytes_cap=200_000_000,training_admitted=False)
    write_json(output/'preparation-freeze.json',freeze)
    try:
        evidence=verify_sources();rows=build()
        model=freeze['model'];tok=AutoTokenizer.from_pretrained(model['id'],revision=model['revision'],local_files_only=True,trust_remote_code=False)
        from tool_lab.decision_sources import check_general_pools
        tokens=set();canonical_tokens=[];counts=Counter();maximum=0;presentations=0
        with (output/'presentations-private.jsonl').open('x') as stream:
            for row in rows:
                n=len(row['input']['options']);positions=defaultdict(set)
                for shift in range(n):
                    item,label,order=rotate(row,shift);ids=encode(tok,item,4096);sha=digest(ids)
                    require(sha not in tokens,'Augmented token collision requires reconciliation')
                    tokens.add(sha);maximum=max(maximum,len(ids));presentations+=1
                    require(presentations<=30000,'Presentation cap')
                    for pos,o in enumerate(item['options']): positions[o['id']].add(pos)
                    entry=dict(id=digest([row['id'],order]),canonical_id=row['id'],order=order,
                        supervision=label,input_ids=ids,token_sha256=sha)
                    stream.write(json.dumps(entry,allow_nan=False)+'\n')
                    if shift==0:
                        row['input_ids']=ids;row['token_sha256']=sha;canonical_tokens.append(dict(token_sha256=sha))
                require(all(v==set(range(n)) for v in positions.values()),'Incomplete semantic option position coverage')
                counts[row['family']]+=n
        general=check_general_pools([dict(token_sha256=k) for k in tokens])
        write_rows(output/'canonical-private.jsonl',rows)
        require(read_rows(output/'canonical-private.jsonl')==rows,'Canonical serialization changed')
        byid={r['id']:r for r in rows};read_count=0
        for p in read_rows(output/'presentations-private.jsonl'):
            r=byid[p['canonical_id']];item,label,order=rotate(r,p['order'][0])
            require(order==p['order'] and label==p['supervision'] and digest(p['input_ids'])==p['token_sha256'],'Serialized target/presentation changed')
            read_count+=1
        require(read_count==presentations,'Lost serialized presentation')
        families={}
        for family in sorted(FAMILIES):
            group=[r for r in rows if r['family']==family];contexts=defaultdict(set)
            for r in group:contexts[r['public_evidence_key']].add(r['supervision']['semantics'])
            families[family]=dict(canonical_questions=len(group),by_supervision=dict(Counter(r['supervision']['semantics'] for r in group)),
                by_task=dict(Counter(r['task'] for r in group)),source_group_ids=len({g for r in group for g in r['source_group_ids']}),
                public_evidence_groups=len(contexts),groups_with_both_decisions_and_forecasts=sum('outcome_distribution' in s and len(s)>1 for s in contexts.values()),
                uncertain_outcomes=sum(r['supervision']['semantics']=='outcome_distribution' and sum(p>0 for p in r['supervision']['probabilities'])>1 for r in group),
                cyclic_presentations=counts[family])
        report=dict(status='qualified_candidate_pairing_and_position_index',version=VERSION,training_admitted=False,
            canonical_questions=len(rows),cyclic_presentations=presentations,by_family=families,
            ownership_aliases={family:sorted({g for r in rows if r['family']==family for g in r['source_group_ids']}) for family in sorted(FAMILIES)},
            maximum_tokens=maximum,general_overlap_checks=general,truncations=0,
            new_underlying_tasks=0,new_executed_branches=0,new_model_calls=0,optimizer_updates=0,training_presentations_consumed=0,
            remaining_gates=['public tool-contract sufficiency review','current full-mixture split admission','loss consumer and prospective learning recipe'],
            scope='Existing audited evidence and exact question semantics; cyclic positions add presentations, not independent tasks. Reservation remains forecast-only.')
        report['files']={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'summary.json',report)
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file())<=200_000_000,'Output byte cap')
        return report
    except BaseException as exc:
        write_json(output/'REJECTED.json',dict(error=type(exc).__name__,detail=str(exc),training_admitted=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args().output),indent=2))
