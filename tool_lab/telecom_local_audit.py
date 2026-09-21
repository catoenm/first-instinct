"""Verify frozen simulator receipts and controls without executing new worlds."""
import argparse
import json
from pathlib import Path

from tool_lab.telecom_local import CASES, VARIANTS, canonical, read, sha, verify_state, write


def audit(directory,failed):
    plan=read(directory/'freeze-private.json')
    for path,expected in plan['paths'].items():
        if sha(path)!=expected:raise ValueError('Frozen simulator source changed')
    attempts=[json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    earlier=[json.loads(line) for line in (failed/'attempts.jsonl').read_text().splitlines()]
    if len(earlier)!=1 or earlier[0]['returncode']==0 or len(attempts)!=15:
        raise ValueError('Execution accounting differs from the corrected protocol')
    if {(r['case'],r['variant']) for r in attempts}!={(c,v) for c in CASES for v in VARIANTS}:
        raise ValueError('A declared case/control is missing')
    result,hashes,calls={}, {}, 0
    for case in CASES:
        records={variant:read(directory/f'{case}-{variant}-private.json') for variant in VARIANTS}
        for variant,record in records.items():
            if record['case']!=case or record['variant']!=variant or record['outbound_attempts'] or record['new_model_calls']:
                raise ValueError('Execution scope changed')
            if record['calls']!=len(record['trace']) or record['calls']>20:
                raise ValueError('Call accounting differs')
            actual=verify_state(record['initial'],record['final'],case,plan['fixture'])
            if actual!=record['verdict']:raise ValueError('Stored-state verdict does not reproduce')
            if any(event['tool'].startswith(('_','assert_')) for event in record['trace']):
                raise ValueError('Private assertion/break function leaked into public calls')
            calls+=record['calls'];hashes[f'{case}-{variant}-private.json']=sha(directory/f'{case}-{variant}-private.json')
        repair,replay=records['repair'],records['replay']
        if canonical({k:v for k,v in repair.items() if k!='variant'})!=canonical({k:v for k,v in replay.items() if k!='variant'}):
            raise ValueError('Exact independent execution replay differs')
        passed=(repair['verdict']['success'] and replay['verdict']['success'] and
                not records['stop']['verdict']['success'] and not records['incomplete']['verdict']['success'] and
                records['collateral']['verdict']['goal_satisfied'] and not records['collateral']['verdict']['frame_preserved'] and
                not records['collateral']['verdict']['success'])
        result[case]=dict(qualified=passed,controls={name:r['verdict'] for name,r in records.items()},
                         executed_worlds=5,exact_replay=True)
    return dict(status='passed' if all(r['qualified'] for r in result.values()) else 'partially_qualified',
        source_commit=plan['commit'],mechanisms=3,qualified_mechanisms=sum(r['qualified'] for r in result.values()),
        authored_underlying_tasks=3,world_executions=15,prior_failed_adapter_attempts=1,total_attempted_worlds=16,
        completed_tool_calls=calls,known_failed_attempt_calls=3,independent_replays=3,
        no_op_controls=3,incomplete_repair_controls=3,collateral_controls=3,
        load_official_task_solutions=False,new_model_calls=0,new_training_questions=0,optimizer_steps=0,
        outbound_attempts=0,freeze_sha256=sha(directory/'freeze-private.json'),runtime_sha256=sha(directory/'runtime.txt'),
        receipt_hashes=hashes,results=result,
        limitation='Three authored fixtures over pinned real telecom tools in a restricted package namespace. Not an official benchmark run, general decision data, a training result or a policy-calibration demonstration.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','failed','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=audit(a.directory,a.failed);write(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipt_hashes','results')},indent=2))
