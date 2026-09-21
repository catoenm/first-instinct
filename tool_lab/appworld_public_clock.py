"""Execution-verified public clock reads without loading task ground truth."""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

from tool_lab.appworld_qualification import read, sha, write
from tool_lab.appworld_trace import digest, snapshot, state_hashes

VERSION = 'appworld-public-clock-v2'


def prepare(source, output):
    plan = read(source/'freeze-private.json')
    audit = read(source/'audit.json')
    if not audit['all_four_programs_qualify']:
        raise ValueError('All reserved branches must qualify first')
    paths = {Path(p):h for p,h in plan['paths'].items()}
    paths.update({source/p: h for p,h in audit['receipt_hashes'].items()})
    for p in (source/'audit.json', source/'freeze-private.json', Path(__file__).resolve(),
              Path('docs/appworld-evidence-interface-v2-protocol.md').resolve()):
        paths[p] = sha(p)
    for p,h in paths.items():
        if sha(p)!=h: raise ValueError('Existing qualification changed')
    result = dict(version=VERSION, source=str(source.resolve()), root=plan['root'],
                  tasks=plan['tasks'], max_worlds=8, max_calls_per_world=96,
                  load_ground_truth=False, paths={str(p.resolve()):h for p,h in sorted(paths.items())})
    output.mkdir(parents=True,exist_ok=False)
    write(output/'freeze-private.json',result)
    return result


def execute(plan,index,replica,output):
    if plan['version']!=VERSION or not 0<=index<4 or replica not in (0,1) or plan['load_ground_truth']:
        raise ValueError('Undeclared clock execution')
    for path,h in plan['paths'].items():
        if sha(path)!=h: raise ValueError('Frozen source changed')
    task = plan['tasks'][index]
    if task['role']!='reserved_transfer_only' or 'venmo' not in task['apps']:
        raise ValueError('Unexpected task ownership')
    reference = read(Path(plan['source'])/f'{index}-capture.json')
    points = sorted(set([0]+reference['points']))
    if len(points)!=3: raise ValueError('Expected initial and two history points')
    name=f'{VERSION}/{index}-{replica}'
    path=output/f'{index}-{replica}-private.json'
    if path.exists() or (Path(plan['root'])/'experiments/outputs'/name).exists():
        raise ValueError('Never overwrite clock evidence')
    os.environ['APPWORLD_ROOT']=plan['root'];os.environ['PYTHON_DOTENV_DISABLED']='1'
    from appworld import AppWorld,load_task_ids
    if task['task_id'] not in load_task_ids('train'):raise ValueError('Upstream training inventory only')
    denied=[]
    def block(*args,**kwargs):
        denied.append(True);raise PermissionError('Outgoing sockets disabled')
    observations=[];responses=[]
    with ExitStack() as stack:
        for target,method in ((socket.socket,'connect'),(socket.socket,'connect_ex'),(socket,'create_connection')):
            stack.enter_context(patch.object(target,method,block))
        with AppWorld(task_id=task['task_id'],experiment_name=name,max_interactions=1,
                      max_api_calls_per_interaction=96,random_seed=20260920,timeout_seconds=90,
                      load_ground_truth=False,raise_on_failure=True) as world:
            initial=state_hashes(snapshot(world.models))
            if initial!=reference['initial']:raise ValueError('Initial world differs')
            for i in range(points[-1]+1):
                if i in points:
                    before=state_hashes(snapshot(world.models))
                    expected=initial if i==0 else read(Path(plan['source'])/f'{index}-{i}-stop.json')['final']
                    if before!=expected:raise ValueError('Clock prefix state differs from executed branch')
                    response=world.requester.request('phone','get_current_date_and_time')
                    after=state_hashes(snapshot(world.models))
                    if before!=after:raise ValueError('Clock read changed application records')
                    observations.append(dict(before_call=i,app='phone',api='get_current_date_and_time',
                                             arguments={},response=response,before=before,after=after))
                if i<points[-1]:
                    event=reference['trace'][i]
                    response=world.requester.request(event['app'],event['api'],raise_on_failure=False,
                                                     **copy.deepcopy(event['arguments']))
                    if response!=event['response']:raise ValueError('Prefix response differs')
                    responses.append(response)
            if denied or len(world.requester.requests)>96:raise ValueError('Execution bound exceeded')
            result=dict(version=VERSION,task_index=index,replica=replica,
                        task_id_sha256=digest(task['task_id']),initial=initial,observations=observations,
                        prefix_response_sha256=digest(responses),prefix_calls=points[-1],clock_calls=3,
                        total_api_calls=len(world.requester.requests),outbound_attempts=len(denied),
                        load_ground_truth=False)
    write(path,result)
    return {k:result[k] for k in ('task_index','replica','prefix_calls','clock_calls','total_api_calls')}


def audit(directory):
    plan=read(directory/'freeze-private.json')
    for path,h in plan['paths'].items():
        if sha(path)!=h:raise ValueError('Frozen input changed')
    calls=0;hashes={}
    for i in range(4):
        paths=[directory/f'{i}-{r}-private.json' for r in (0,1)]
        a,b=map(read,paths)
        if a['replica']!=0 or b['replica']!=1:raise ValueError('Independent replica identity differs')
        if {k:v for k,v in a.items() if k!='replica'}!={k:v for k,v in b.items() if k!='replica'}:
            raise ValueError('Independent public clock evidence differs')
        if len(a['observations'])!=3 or a['load_ground_truth'] or a['outbound_attempts']:
            raise ValueError('Clock execution contract differs')
        if any(o['before']!=o['after'] for o in a['observations']):raise ValueError('Clock mutated state')
        if len({digest(o['response']) for o in a['observations']})!=1:
            raise ValueError('Clock changes across observed prefix')
        calls+=a['total_api_calls']+b['total_api_calls']
        hashes.update({p.name:sha(p) for p in paths})
    return dict(status='passed',world_executions=8,underlying_task_instances=4,task_programs=4,
                public_clock_calls=24,total_api_calls=calls,independent_replays=4,
                load_ground_truth=False,new_model_calls=0,new_outcome_labels=0,new_accepted_questions=0,
                all_clock_reads_state_preserving=True,all_prefixes_match_original_branches=True,
                freeze_sha256=sha(directory/'freeze-private.json'),receipt_hashes=hashes)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','execute','audit'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--index',type=int);p.add_argument('--replica',type=int)
    a=p.parse_args()
    if a.mode=='prepare':r=prepare(a.source,a.output);r={k:r[k] for k in ('version','max_worlds','load_ground_truth')}
    elif a.mode=='execute':r=execute(read(a.output/'freeze-private.json'),a.index,a.replica,a.output)
    else:
        r=audit(a.output)
        with (a.output/'audit.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))
