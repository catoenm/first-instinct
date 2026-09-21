"""Correct plain-text tool serialization; preserve the failed first attempt."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import socket
import sys
import types
import uuid
from unittest.mock import patch
import tomllib

from tool_lab.telecom_local import (COMMIT, CASES, VARIANTS, canonical, digest, sha,
                                   read, write, snapshot, verify_state)

def prepare(upstream,output):
    import subprocess
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise ValueError('Different telecom source revision')
    if subprocess.check_output(['git','-C',str(upstream),'diff','--name-only'],text=True).strip():
        raise ValueError('Upstream simulator source changed')
    db=tomllib.loads((upstream/'data/tau2/domains/telecom/db.toml').read_text())
    lines={r['line_id']:r for r in db['lines']};plans={r['plan_id']:r for r in db['plans']}
    eligible=[]
    for customer in sorted(db['customers'],key=lambda r:r['customer_id']):
        if not customer['line_ids']:continue
        line=lines[customer['line_ids'][0]];plan=plans[line['plan_id']]
        if line['status']=='Active' and plan['data_limit_gb']>0 and plan['data_refueling_price_per_gb']>0:
            eligible.append(dict(phone=line['phone_number'],customer_id=customer['customer_id'],
                line_id=line['line_id'],price_per_gb=plan['data_refueling_price_per_gb']))
    if not eligible: raise ValueError('No declared fixture account')
    files=list((upstream/'src/tau2').rglob('*.py'))
    files.extend(p for p in (upstream/'data/tau2/domains/telecom').iterdir()
                 if p.suffix in ('.toml','.md'))
    files.extend([Path(__file__).resolve(),Path('docs/telecom-local-v1-correction.md').resolve(),upstream/'LICENSE',upstream/'pyproject.toml',Path('tool_lab/telecom_local.py').resolve(),Path('docs/telecom-local-v1-protocol.md').resolve()])
    freeze=dict(version='telecom-local-v1-retry1',commit=COMMIT,upstream=str(upstream.resolve()),fixture=eligible[0],
        cases=list(CASES),variants=list(VARIANTS),max_worlds=15,max_calls_per_world=20,
        prior_failed_attempts=1,combined_attempt_ceiling=16,seed=20260921,uuid_base=1000,new_model_calls=0,role='substrate_qualification_only',
        paths={str(p.resolve()):sha(p) for p in sorted(files)})
    output.mkdir(parents=True,exist_ok=False);write(output/'freeze-private.json',freeze)
    return dict(version=freeze['version'],max_worlds=15,cases=list(CASES),frozen_files=len(files))

def execute(plan,case,variant,output):
    if case not in CASES or variant not in VARIANTS: raise ValueError('Undeclared execution')
    for path,expected in plan['paths'].items():
        if sha(path)!=expected:raise ValueError('Frozen simulator/adapter source changed')
    destination=output/f'{case}-{variant}-private.json'
    if destination.exists():raise ValueError('Never overwrite an executed world')
    os.environ['PYTHON_DOTENV_DISABLED']='1';os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
    os.environ['TAU2_DATA_DIR']=str(Path(plan['upstream'])/'data')
    namespace=types.ModuleType('tau2');namespace.__path__=[str(Path(plan['upstream'])/'src/tau2')]
    if 'tau2' in sys.modules:raise ValueError('Fresh process required for isolated substrate loading')
    sys.modules['tau2']=namespace
    random.seed(plan['seed']);denied=[];uuid_counter=plan['uuid_base']
    def block(*args,**kwargs):denied.append(True);raise PermissionError('No outbound execution')
    def fixed_uuid():
        nonlocal uuid_counter
        uuid_counter+=1;return uuid.UUID(int=uuid_counter<<96)
    with ExitStack() as stack:
        for target,method in ((socket.socket,'connect'),(socket.socket,'connect_ex'),(socket,'create_connection')):
            stack.enter_context(patch.object(target,method,block))
        stack.enter_context(patch.object(uuid,'uuid4',fixed_uuid))
        from tau2.domains.telecom.environment import get_environment
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.domains.telecom.user_data_model import TelecomUserDB
        from loguru import logger
        logger.remove()
        root=Path(plan['upstream'])/'data/tau2/domains/telecom'
        db=TelecomDB.load(root/'db.toml');user=TelecomUserDB.load(root/'user_db.toml');fixture=plan['fixture']
        line=next(r for r in db.lines if r.line_id==fixture['line_id'])
        rate=next(r for r in db.plans if r.plan_id==line.plan_id)
        user.surroundings.phone_number=fixture['phone'];user.surroundings.is_abroad=case=='coupled_roaming'
        line.data_refueling_gb=0;line.data_used_gb=rate.data_limit_gb if case=='carrier_allowance' else 0
        line.roaming_enabled=False;user.device.roaming_enabled=False
        user.device.airplane_mode=case=='device_switches';user.device.data_enabled=case!='device_switches'
        env=get_environment(db=db,user_db=user,solo_mode=True)
        env.user_tools.simulate_network_search()  # fixture initialization only
        initial=snapshot(env);trace=[]
        def call(name,arguments=None,read_only=False):
            if len(trace)>=20:raise ValueError('Tool-call ceiling')
            before=snapshot(env)
            # Deliberately propagate every unexpected exception. The public
            # wrapper's catch-all must not turn infrastructure failure into truth.
            result=env.make_tool_call(name,requestor='assistant',**(arguments or {}));env.sync_tools()
            value=env.to_json_str(result)
            if read_only and canonical(before)!=canonical(snapshot(env)):
                raise ValueError('Read probe mutated state')
            trace.append(dict(tool=name,arguments=arguments or {},response=value))
            return value
        customer=json.loads(call('get_customer_by_phone',dict(phone_number=fixture['phone']),True))
        if customer['customer_id']!=fixture['customer_id'] or customer['line_ids'][0]!=fixture['line_id']:
            raise ValueError('Public identity lookup differs')
        ids=dict(customer_id=customer['customer_id'],line_id=customer['line_ids'][0])
        call('get_data_usage',ids,True);before_probe=call('run_speed_test',read_only=True)
        operations={'device_switches':[('toggle_airplane_mode',{}),('toggle_data',{})],
            'coupled_roaming':[('enable_roaming',ids),('toggle_roaming',{})],
            'carrier_allowance':[('refuel_data',dict(ids,gb_amount=1))]}[case]
        if variant=='stop':operations=[]
        elif variant=='incomplete':operations=[operations[-1]] if case!='carrier_allowance' else [('toggle_data',{})]
        for name,arguments in operations:call(name,arguments)
        if variant=='collateral':env.user_tools.db.device.battery_level-=1
        final_probe=call('run_speed_test',read_only=True);final=snapshot(env)
        result=verify_state(initial,final,case,fixture)
        probe_success=isinstance(final_probe,str) and final_probe.startswith('Speed Test Result:')
        if result['goal_satisfied']!=probe_success:raise ValueError('Stored predicate and service observation disagree')
        if denied:raise ValueError('Outbound attempt during simulator execution')
        imported={name:sha(module.__file__) for name,module in sys.modules.items() if name.startswith('tau2.') and getattr(module,'__file__',None)}
        if any(str(Path(module.__file__).resolve()) not in plan['paths'] for name,module in sys.modules.items()
               if name.startswith('tau2.') and getattr(module,'__file__',None)):
            raise ValueError('Unfrozen simulator module was imported')
        receipt=dict(case=case,variant=variant,initial=initial,final=final,trace=trace,verdict=result,
            initial_service_probe=before_probe,final_service_probe=final_probe,calls=len(trace),
            outbound_attempts=0,new_model_calls=0,imported_sources=imported)
        write(destination,receipt)
    return dict(case=case,variant=variant,calls=len(trace),**result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','execute'))
    p.add_argument('--upstream',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case');p.add_argument('--variant');a=p.parse_args()
    result=prepare(a.upstream,a.output) if a.mode=='prepare' else execute(read(a.output/'freeze-private.json'),a.case,a.variant,a.output)
    print(json.dumps(result))
