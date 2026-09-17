"""Create visible evidence separately from private verdicts and keep all receipts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .candidates import proposals
from .tasks import TASKS,cases

ROOT=Path(__file__).resolve().parents[1]
WORKER=Path(__file__).with_name('worker.py')
VIEWS=('initial','copy','new_check','higher_price')
QUESTION='Will this candidate pass the fixed private test suite for the stated contract? Estimate the probability of yes using only the code and revealed checks.'


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()
def file_sha(p):return sha(Path(p).read_bytes())
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_rows(p,rows):
    raw=''.join(canonical(r)+'\n' for r in rows).encode()
    Path(p).write_bytes(gzip.compress(raw,mtime=0) if str(p).endswith('.gz') else raw)
def read_rows(p):
    raw=gzip.decompress(Path(p).read_bytes()) if str(p).endswith('.gz') else Path(p).read_bytes()
    return [json.loads(x) for x in raw.splitlines()]


def run(source,inputs):
    started=time.perf_counter()
    try:
        p=subprocess.run([sys.executable,'-I',str(WORKER)],input=canonical({'code':source,'inputs':inputs}),
                         text=True,capture_output=True,timeout=3,check=True)
        result=json.loads(p.stdout)
    except subprocess.TimeoutExpired:result={'results':None,'error':'Timeout'}
    except (subprocess.CalledProcessError,json.JSONDecodeError):result={'results':None,'error':'WorkerFailure'}
    return result,time.perf_counter()-started


def check(t,source,inputs,repeat=True):
    result,seconds=run(source,inputs)
    other,elapsed=run(source,inputs) if repeat else (result,0.)
    stable=canonical(result)==canonical(other)
    expected=[t.reference(*json.loads(json.dumps(args))) for args in inputs]
    records=[]
    for index,(args,value) in enumerate(zip(inputs,expected)):
        actual=result['results'][index] if result['results'] is not None else {'value':None,'error':result['error']}
        passed=actual['error'] is None and canonical(actual['value'])==canonical(value)
        records.append({'input':args,'expected':value,'observed':actual['value'],'error':actual['error'],'passed':passed})
    return {'stable':stable,'passed':all(r['passed'] for r in records),'checks':records,
            'seconds':seconds+elapsed,'process_runs':2 if repeat else 1,'test_executions':len(inputs)*(2 if repeat else 1),
            'worker_error':result['error'],'result_sha256':sha(canonical(result))}


def seed_for(t,offset):return int(sha(t.name)[:8],16)+offset


def visible_inputs(t):return cases(t,seed_for(t,11000),5)
def private_inputs(t):
    visible=visible_inputs(t)
    # Fixed examples from the contract plus fresh unique inputs; no visible test is reused.
    initial=[a for a,_ in t.fixtures if canonical(a) not in {canonical(x) for x in visible}]
    return initial+cases(t,seed_for(t,22000),32-len(initial),exclude=visible+initial)


def render(case,view):
    checks=case['visible_checks'][:3 if view=='new_check' else 2]
    lines=[]
    for i,r in enumerate(checks):
        result=canonical(r['observed']) if r['error'] is None else 'raised '+r['error']
        lines.append(f'Check {i+1}: input {canonical(r["input"])}; expected {canonical(r["expected"])}; '+
                     ('PASS (returned expected value)' if r['passed'] else 'FAIL; observed '+result))
    if view=='copy':lines.append('The following is a copy of Check 1, not another execution: '+lines[0])
    price='0.15' if view=='higher_price' else '0.01'
    return ('Contract: '+case['description']+'\nCandidate:\n'+case['code']+
            'Revealed executions:\n'+'\n'.join(lines)+'\nThe private suite uses other inputs satisfying this contract. '
            'Its results have not been revealed. The verification price is '+price+
            ' cost units; it was assigned independently of correctness.\nQuestion: '+QUESTION)


def build_pool(folder,splits,limit=24,held_out=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    rows=[];receipts=[];rejected=[]
    for t in TASKS:
        if t.split not in splits:continue
        for proposal in proposals(t,seed_for(t,33000 if not held_out else 44000),limit,held_out):
            identifier=t.name+'-'+proposal['code_sha256'][:16]
            result=check(t,proposal['code'],visible_inputs(t))
            receipt={'id':identifier,**result};receipts.append(receipt)
            if not result['stable'] or result['worker_error']:
                rejected.append({'id':identifier,'reason':'unstable' if not result['stable'] else result['worker_error'],**proposal});continue
            rows.append({'id':identifier,'task':t.name,'family':t.family,'split':t.split,'description':t.description,
                         **proposal,'visible_checks':result['checks'],'source':'authored-pilot-v1','license':'MIT'})
    write_rows(folder/'cases.jsonl.gz',rows);write_rows(folder/'visible-receipts.jsonl.gz',receipts)
    write_rows(folder/'rejected.jsonl.gz',rejected)
    requests=[{'id':r['id']+'-'+v,'case_id':r['id'],'view':v,'state':render(r,v)} for r in rows for v in VIEWS]
    write_rows(folder/'requests.jsonl.gz',requests)
    write_json(folder/'manifest.json',{'candidate_cases':len(rows),'requests':len(requests),'tasks':sorted({r['task'] for r in rows}),
        'proposals':len(receipts),'rejected':len(rejected),'held_out_transformations':held_out,
        'visible_test_executions':sum(r['test_executions'] for r in receipts),
        'visible_seconds':sum(r['seconds'] for r in receipts),
        'files':{p.name:file_sha(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    return rows


class LabelOracle:
    """Only explicitly requested ids return labels. Each recipe has its own budget.

    Cache reuse saves physical executions across strategies. The receipt records
    both logical verification work and newly executed work; neither is hidden.
    """
    def __init__(self,rows,cache,budget,ledger):
        self._rows={r['id']:r for r in rows};self._tasks={t.name:t for t in TASKS}
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.budget=budget;self.ledger=Path(ledger);self.seen=set()

    def query(self,ids,reason):
        if len(ids)!=len(set(ids)) or any(i in self.seen for i in ids):raise ValueError('Repeated acquisition')
        if len(self.seen)+len(ids)>self.budget:raise ValueError('Verification budget exceeded')
        if any(i not in self._rows for i in ids):raise ValueError('Unknown candidate')
        labels=[]
        with self.ledger.open('a') as log:
            for identifier in ids:
                row=self._rows[identifier];t=self._tasks[row['task']]
                inputs=private_inputs(t)
                key=sha(canonical({'code':row['code'],'inputs':inputs,'contract':t.description,
                                  'worker':file_sha(WORKER),'tasks':file_sha(Path(__file__).with_name('tasks.py')),
                                  'verifier':file_sha(Path(__file__))}))
                path=self.cache/(key+'.json');cached=path.exists()
                if cached:result=json.loads(path.read_text())
                else:
                    result=check(t,row['code'],inputs);write_json(path,result)
                if not result['stable'] or result['worker_error']:raise ValueError('Private verification was unstable or invalid')
                log.write(canonical({'id':identifier,'query':len(self.seen)+1,'reason':reason,
                    'passed':result['passed'],'receipt_sha256':file_sha(path),'cache_key':key,
                    'cache_hit':cached,'logical_test_executions':result['test_executions'],
                    'standalone_measured_seconds':result['seconds'],'new_execution_seconds':0. if cached else result['seconds']})+'\n')
                labels.append(int(result['passed']));self.seen.add(identifier)
        return labels


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build_pool(a.output,{'train','validation'})


if __name__=='__main__':main()
