"""Audit a successfully closed, recovered expanded pilot; never runs inference."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tarfile

from tool_lab.expanded_advancement import require_closed_recovery,advancement
from tool_lab.expanded_checkpoint_audit import audit as audit_checkpoints,sha
from tool_lab import expanded_evaluation_audit as evaluation
from tool_lab import expanded_selection_audit as selection
from tool_lab.mixed_accounting import read_rows
from tool_lab.mixed_results import general_metrics

EXPECTED_FREEZE='838d3c3ba19292ec80cd4618a641c74ada72b802a2243bce37cdf50b8e3daf90'


def read(p):return json.loads(p.read_text())


def verify_files(root,archive):
    receipt=read(root/'cloud-collection.json')
    if archive.stat().st_size!=receipt['archive_bytes'] or sha(archive)!=receipt['archive_sha256']:
        raise ValueError('Recovered archive differs from collection receipt')
    with tarfile.open(archive) as tf:
        members=tf.getmembers();names=[m.name for m in members]
        if len(set(names))!=len(names) or any(not m.isfile() for m in members):raise ValueError('Invalid archive members')
        for n in names:
            p=PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or str(p)!=n:raise ValueError('Unsafe archive path')
        member=tf.getmember('artifact-hashes.json')
        if member.size>2_000_000:raise ValueError('Unexpectedly large artifact manifest')
        contents=tf.extractfile(member).read()
    if contents!=(root/'artifact-hashes.json').read_bytes():raise ValueError('Local manifest differs from archived manifest')
    hashes=json.loads(contents)
    if set(names)!=set(hashes)|{'artifact-hashes.json'} or len(hashes)!=receipt['verified_files']:
        raise ValueError('Artifact coverage differs from collection receipt')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=set(names)|{'cloud-collection.json'}:raise ValueError('Unexpected or missing recovered artifact')
    for n,h in hashes.items():
        p=root/n
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not re.fullmatch('[0-9a-f]{64}',h) or sha(p)!=h:
            raise ValueError('Recovered artifact hash differs: '+n)
    return dict(status='passed',files=len(hashes),archive_sha256=receipt['archive_sha256'],archive_bytes=receipt['archive_bytes'])


def cached_development(cache,name,directory):
    audit_name=name+'-audit.json';selection_name=name+'-selection.json'
    manifest_name='snapshot-manifest.json' if name=='outcome-1507' else name+'-snapshot-manifest.json'
    hashes_name='hashes.json' if name=='outcome-1507' else name+'-hashes.json'
    if not (cache/audit_name).exists():return None
    hashes=read(cache/hashes_name)
    for n,h in hashes.items():
        if Path(n).name!=n or sha(cache/n)!=h:raise ValueError('Cached development report or receipt changed')
    snapshot=read(cache/manifest_name)
    allowed=re.compile(r'^(baseline-validation|update-[0-9]+-validation)-(metrics|forecasts|trajectories|retention-predictions)\.(json|jsonl)$')
    fixed={'run.json','learning-ledger.jsonl','training.jsonl','gradient-diagnostic.json','rollouts.jsonl'}
    expected={p.name for p in directory.iterdir() if p.is_file() and (allowed.fullmatch(p.name) or p.name in fixed)}
    if set(snapshot['files'])!=expected:raise ValueError('Cached development snapshot has incomplete or extra coverage')
    for n,h in snapshot['files'].items():
        if sha(directory/n)!=h:raise ValueError('Recovered development receipt differs from audited snapshot')
    result=read(cache/audit_name);selected=read(cache/selection_name)
    if result['status']!='passed' or selected['status']!='passed':
        raise ValueError('Cached development audit did not pass')
    receipt=read(directory/'run.json')
    if (result['arm']!=receipt['arm'] or result['seed']!=receipt['seed'] or
        result['selected_update']!=receipt['selected_update'] or selected['selected_update']!=receipt['selected_update']):
        raise ValueError('Cached audit belongs to a different arm or selected checkpoint')
    if (result['source_sha256']!=sha(Path(evaluation.__file__)) or
        selected['source_sha256']!=sha(Path(selection.__file__))):
        # Preserve the old, byte-verified report but recompute with the current
        # auditor. A source revision must never silently inherit its approval.
        return None
    return result,selected


def input_digest(row):
    payload=json.dumps([row['input_ids'],len(row['option_ids'])],separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest()


def consumption(run,data,arms):
    ids=defaultdict(set);tokens=defaultdict(set);counts=Counter();diagnostics=Counter();episodes=0;caseids=set()
    forecasts={r['id']:r for r in read_rows(data/'train-forecasts.jsonl')};replay={r['id']:r for r in read_rows(data/'replay.jsonl')}
    for name,a in arms.items():
        if name=='original-test':continue
        directory=run/name;traces=read_rows(directory/'rollouts.jsonl');episodes+=len(traces);caseids.update(t['case_id'] for t in traces)
        policy={}
        for t in traces:
            for e in t['actor_events']:
                r=e['encoded_input'];key=r['id'];value=input_digest(r)
                if key in policy and policy[key]!=value:raise ValueError('One policy question ID refers to different token inputs')
                policy[key]=value
        for event in read_rows(directory/'learning-ledger.jsonl'):
            phase=event['phase'];kind=event.get('component')
            if phase=='completed_diagnostic_backward':diagnostics[kind]+=len(event['ids'])
            if phase!='completed_backward':continue
            counts[kind]+=len(event['ids']);ids[kind].update(event['ids'])
            for key in event['ids']:
                tokens[kind].add(policy[key] if kind=='policy' else input_digest((forecasts if kind=='outcome' else replay)[key]))
    cases={c['id']:c for c in read_rows(data/'train-cases.jsonl')}
    return dict(live_training_episodes=episodes,underlying_training_tasks=evaluation.task_counts([cases[k] for k in caseids]),
        objectives={k:dict(completed_presentations=counts[k],distinct_question_ids=len(ids[k]),
            distinct_model_token_inputs=len(tokens[k]),repeated_question_presentations=counts[k]-len(ids[k])) for k in sorted(counts)},
        diagnostic_presentations=dict(diagnostics),
        accepted_updates=sum(a['development']['actual_learning']['accepted_transactions'] for n,a in arms.items() if n!='original-test'),
        rejected_updates=sum(a['development']['actual_learning']['rejected_transactions'] for n,a in arms.items() if n!='original-test'),
        note='Token-input identity uses exact token IDs and offered-choice count. Question identities can differ for identical visible inputs in different worlds. Counts pool all arms; shared inputs and repeated presentations are not new underlying tasks. Evaluation episodes and preexecuted forecast-label branches are separate.')


def audit(root,archive,original,cache,tokenizer):
    states=require_closed_recovery(root)
    recovery=verify_files(root,archive);data=root/'data';frozen=read(data/'freeze.json')
    if sha(data/'freeze.json')!=EXPECTED_FREEZE:raise ValueError('Different prospective experiment freeze')
    repository=Path(__file__).resolve().parents[1]
    for name,h in frozen['sources'].items():
        if sha(root/name)!=h or sha(repository/name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in frozen['files'].items():
        if sha(data/name)!=h:raise ValueError('Frozen data changed: '+name)
    checkpoints=audit_checkpoints(root,original);arms={}
    transfer_cases=read_rows(data/'transfer-cases.jsonl');forecast_truth=read_rows(data/'transfer-forecasts.jsonl')
    retention_truth=read_rows(data/'retention.jsonl');general_truth=read_rows(data/'transfer.jsonl')
    for name in sorted(states):
        directory=root/'run'/name;entry={}
        if name!='original-test':
            cached=cached_development(cache,name,directory)
            if cached:development,selected=cached;entry['reused_development_audit']=True
            else:
                development=evaluation.audit(data,directory,tokenizer)
                selected=selection.audit(read_rows(directory/'training.jsonl'),development['measurements'],states[name],frozen['recipe'])
                entry['reused_development_audit']=False
            entry.update(development=development,selection=selected,
                baseline_retention=development['measurements']['baseline-validation']['retention'])
        stated=read(directory/'selected-test-metrics.json')
        executed,trajectory=evaluation.trajectories(read_rows(directory/'selected-test-trajectories.jsonl'),transfer_cases,tokenizer,require_exact_coverage=True)
        forecast=evaluation.forecasts(read_rows(directory/'selected-test-forecasts.jsonl'),forecast_truth)
        retention=general_metrics(read_rows(directory/'selected-test-retention-predictions.jsonl'),retention_truth)
        general=general_metrics(read_rows(directory/'selected-general-transfer-predictions.jsonl'),general_truth)
        evaluation.check_report(stated,trajectory);evaluation.check_report(stated['forecast'],forecast)
        evaluation.check_report(stated['retention'],retention);evaluation.check_report(read(directory/'selected-general-transfer.json'),general)
        entry.update(transfer=dict(**trajectory,forecast=forecast,retention=retention),
            transfer_execution=executed,general_transfer=general,selected_update=states[name]['selected_update'])
        arms[name]=entry
    return dict(status='passed',freeze_sha256=EXPECTED_FREEZE,recovery=recovery,checkpoint_audit=checkpoints,
        arms=arms,advancement=advancement(arms,frozen['advancement']),consumption=consumption(root/'run',data,arms),
        prepared=frozen['counts'],new_model_calls=0,new_environment_executions=0,
        auditor_sources={p.name:sha(p) for p in [Path(__file__),Path(evaluation.__file__),Path(selection.__file__),repository/'tool_lab/expanded_checkpoint_audit.py',repository/'tool_lab/expanded_advancement.py']},
        limits='Closed-study receipt reconstruction. Finite authored priors and twelve related initial calendar worlds do not support broad real-world calibration or statistical significance claims. Inspect per-structure results and both seeds; do not turn the exposed calendar mechanism into another untouched test.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','archive','original','cache','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(a.root.resolve()) or a.output.exists():raise ValueError('Use a fresh report outside immutable recovered artifacts')
    require_closed_recovery(a.root)  # Check before tokenizer initialization or reading predictions.
    from transformers import AutoTokenizer
    from scale_lab.common import MODELS
    spec=MODELS['qwen35-9b'];tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],token=False,local_files_only=True)
    result=audit(a.root,a.archive,a.original,a.cache,tokenizer)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
