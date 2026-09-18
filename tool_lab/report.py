"""Join completed Harbor terminal rewards to exact selected-command trajectories."""

import argparse
import json
from pathlib import Path


def summarize(job):
    rows=[]
    for path in sorted(Path(job).glob('*/result.json')):
        result=json.loads(path.read_text())
        trajectory_path=path.parent/'agent/choices.json'
        trajectory=json.loads(trajectory_path.read_text()) if trajectory_path.exists() else None
        verifier=result.get('verifier_result') or {}
        rewards=verifier.get('rewards')
        success=rewards.get('reward') if isinstance(rewards,dict) else None
        error=result.get('exception_info')
        count=trajectory['executed_commands'] if trajectory else None
        usable=(not error and trajectory is not None and trajectory['status'] in ('finished_by_selector','step_budget_exhausted')
                and success in (0,1))
        row={'trial':path.parent.name,'task':result.get('task_name'),'harbor_exception':error,
             'verified_success':success,'commands':count,'status':'complete' if usable else 'unusable',
             'total_reward':success-.01*count if usable else None,'training_performed':False}
        if usable:
            remaining=success
            returns=[]
            for event in reversed(trajectory['events']):
                remaining+=event.get('immediate_reward',0.)
                returns.append({'step':event['step'],'return':remaining,
                                'selector_input_sha256':event['selector_input_sha256'],
                                'selected':event['selected'],'selected_log_probability':event['selected_log_probability']})
            row['returns']=list(reversed(returns))
            row['policy_mode']=trajectory['mode']
            row['training_note']='These saved inference traces are not automatically reusable PPO rollouts; training must collect with its current policy and exact input encoding.'
        rows.append(row)
    return {'schema':'harbor-command-report-v1','trials':rows,
            'scope':'Engineering qualification only. Task and candidate generators contribute to success; this is not a selector benchmark.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.job),indent=2,allow_nan=False)+'\n')
