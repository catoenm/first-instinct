"""Bounded qualification of real telecom tools; no benchmark tasks or models."""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import tomllib
import types
import uuid
from unittest.mock import patch

COMMIT='b7ea9074c1cba482b30687fecdb5c8425fd6f619'
CASES=('device_switches','coupled_roaming','carrier_allowance')
VARIANTS=('repair','stop','incomplete','replay','collateral')


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


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
    files.extend([Path(__file__).resolve(),Path('docs/telecom-local-v1-protocol.md').resolve(),upstream/'LICENSE',upstream/'pyproject.toml'])
    freeze=dict(version='telecom-local-v1',commit=COMMIT,upstream=str(upstream.resolve()),fixture=eligible[0],
        cases=list(CASES),variants=list(VARIANTS),max_worlds=15,max_calls_per_world=20,
        seed=20260921,uuid_base=1000,new_model_calls=0,role='substrate_qualification_only',
        paths={str(p.resolve()):sha(p) for p in sorted(files)})
    output.mkdir(parents=True,exist_ok=False);write(output/'freeze-private.json',freeze)
    return dict(version=freeze['version'],max_worlds=15,cases=list(CASES),frozen_files=len(files))


def snapshot(env):
    return dict(agent=env.tools.db.model_dump(mode='json'),user=env.user_tools.db.model_dump(mode='json'))


def table(state,name,key): return {r[key]:r for r in state['agent'][name]}


def verify_state(initial,final,case,fixture):
    """Independent explicit goal plus unchanged-frame and exact-charge checks."""
    adjusted=copy.deepcopy(initial)
    a,b=adjusted['user'],final['user']
    changed_paths={
        'device_switches': [('device',k) for k in ('airplane_mode','data_enabled','network_connection_status',
                                                'network_technology_connected','network_signal_strength')],
        'coupled_roaming':[('device','roaming_enabled'),('surroundings','roaming_allowed')],
        'carrier_allowance':[('surroundings','mobile_data_usage_exceeded')]}[case]
    for parent,key in changed_paths:a[parent][key]=b[parent][key]
    lid,cid=fixture['line_id'],fixture['customer_id']
    before=table(initial,'lines','line_id')[lid];after=table(final,'lines','line_id')[lid]
    line=next(r for r in adjusted['agent']['lines'] if r['line_id']==lid)
    charge_ok=True
    if case=='coupled_roaming':line['roaming_enabled']=after['roaming_enabled']
    if case=='carrier_allowance':
        line['data_refueling_gb']=after['data_refueling_gb']
        previous=table(initial,'bills','bill_id');current=table(final,'bills','bill_id')
        changed=[key for key in current if key not in previous or canonical(current[key])!=canonical(previous[key])]
        added_gb=after['data_refueling_gb']-before['data_refueling_gb']
        expected=added_gb*fixture['price_per_gb']
        charge_ok=added_gb in (0,1) and len(changed)==int(added_gb==1) and set(previous)<=set(current)
        for key in changed:
            record=current[key];old=previous.get(key)
            old_items=[] if old is None else old['line_items']
            delta=record['line_items'][len(old_items):]
            charge_ok=charge_ok and record['customer_id']==cid and record['status']=='Draft' and \
                len(delta)==1 and canonical(record['line_items'][:len(old_items)])==canonical(old_items) and \
                delta[0]['amount']==expected and delta[0]['item_type']=='Charge' and \
                record['total_due']-(0 if old is None else old['total_due'])==expected
            if old is not None:
                allowed=copy.deepcopy(old);allowed['line_items']=record['line_items'];allowed['total_due']=record['total_due']
                charge_ok=charge_ok and canonical(allowed)==canonical(record)
            existing=next((j for j,r in enumerate(adjusted['agent']['bills']) if r['bill_id']==key),None)
            if existing is None:adjusted['agent']['bills'].append(record)
            else:adjusted['agent']['bills'][existing]=record
        old_customer=table(initial,'customers','customer_id')[cid];new_customer=table(final,'customers','customer_id')[cid]
        new_ids=[key for key in current if key not in previous]
        charge_ok=charge_ok and new_customer['bill_ids']==old_customer['bill_ids']+new_ids
        next(r for r in adjusted['agent']['customers'] if r['customer_id']==cid)['bill_ids']=new_customer['bill_ids']
    frame=canonical(adjusted)==canonical(final)
    device,surroundings=b['device'],b['surroundings']
    cellular=(not device['airplane_mode'] and device['data_enabled'] and
        device['network_connection_status']=='connected' and device['network_signal_strength']!='none' and
        surroundings['line_active'] and not surroundings['mobile_data_usage_exceeded'] and
        (not surroundings['is_abroad'] or device['roaming_enabled'] and surroundings['roaming_allowed']))
    goal=cellular and (case!='carrier_allowance' or after['data_refueling_gb']-before['data_refueling_gb']==1)
    return dict(goal_satisfied=bool(goal),frame_preserved=frame,charge_consistent=bool(charge_ok),
                success=bool(goal and frame and charge_ok))


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
            value=json.loads(env.to_json_str(result))
            if read_only and canonical(before)!=canonical(snapshot(env)):
                raise ValueError('Read probe mutated state')
            trace.append(dict(tool=name,arguments=arguments or {},response=value))
            return value
        customer=call('get_customer_by_phone',dict(phone_number=fixture['phone']),True)
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
