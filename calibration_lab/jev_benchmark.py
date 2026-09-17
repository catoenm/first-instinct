"""Run probability probes through Jev; preview by default, explicitly execute later.

HTTP schema: https://docs.typesafe.ai/api (checked 2026-09-17).
No answer file is read while constructing or sending requests.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

from .inspection_train import digest,write_json
from .probability_benchmark import read_jsonl

ENDPOINT='https://api.typesafe.ai/v1/systemone'


def payload(case,model='jev-latest'):
    return {'model':model,'state':case['state'],
            'questions':{'event':{'type':'noul','instructions':case['instructions']}}}


def payload_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def parse_probability(response):
    if not isinstance(response.get('model'),str):
        raise ValueError('Response must identify its model')
    answer=response['answers']['event']
    q=answer['noul']
    if answer.get('type')!='noul' or isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) or not 0<=q<=1:
        raise ValueError('Response must contain a finite Noul probability from zero to one')
    return float(q)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


def send(value,key):
    request=urllib.request.Request(ENDPOINT,data=json.dumps(value).encode(),method='POST',
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    # Never forward credentials to a redirected host.
    opener=urllib.request.build_opener(NoRedirect)
    with opener.open(request,timeout=45) as response:
        return json.load(response)


def summarize(benchmark,response_file):
    cases={r['id']:r for r in read_jsonl(benchmark/'requests.jsonl.gz')}
    answers={r['id']:r for r in read_jsonl(benchmark/'answers.jsonl.gz')}
    attempts=read_jsonl(response_file)
    completed={}
    for row in attempts:
        if row['status']!='ok':
            continue
        case=cases[row['id']]
        if row['payload_sha256']!=payload_digest(payload(case,row['requested_model'])):
            raise ValueError('Response was collected for a different request')
        completed[row['id']]=row
    errors=[r['probability']-answers[k]['expected_probability'] for k,r in completed.items()]
    grouped={}
    paraphrases={}
    for identifier,row in completed.items():
        case=cases[identifier]
        grouped.setdefault((case['group'],case['text_version']),{})[case['variant']]=(row['probability'],answers[identifier]['expected_probability'])
        paraphrases.setdefault((case['group'],case['variant']),{})[case['text_version']]=row['probability']
    pair_results={}
    for name,left,right in [('copy_change','copy_initial','copy_revealed'),
                             ('price_change','initial','expensive'),
                             ('unseen_source_change','initial','copy_initial')]:
        values=[abs(v[left][0]-v[right][0]) for v in grouped.values() if left in v and right in v]
        pair_results[name]={'complete_pairs':len(values),'mean':float(np.mean(values)) if values else None}
    wording=[abs(v[0]-v[1]) for v in paraphrases.values() if 0 in v and 1 in v]
    pair_results['wording_change']={'complete_pairs':len(wording),'mean':float(np.mean(wording)) if wording else None}
    updates=[];wrong=[]
    for v in grouped.values():
        if 'initial' not in v:
            continue
        for key in ('independent_zero','independent_one'):
            if key in v:
                actual=v[key][0]-v['initial'][0]
                expected=v[key][1]-v['initial'][1]
                updates.append(abs(actual-expected));wrong.append(actual*np.sign(expected)<-1e-5)
    pair_results['independent_update_error']={'complete_pairs':len(updates),'mean':float(np.mean(updates)) if updates else None}
    pair_results['wrong_direction_rate']={'complete_pairs':len(wrong),'mean':float(np.mean(wrong)) if wrong else None}
    by_family={}
    for family in sorted({c['family'] for c in cases.values()}):
        differences=[r['probability']-answers[k]['expected_probability'] for k,r in completed.items()
                     if cases[k]['family']==family]
        by_family[family]={'successful_requests':len(differences),
                          'probability_rmse':float(np.sqrt(np.mean(np.square(differences)))) if differences else None}
    return {'provider':'TypeSafe','benchmark_requests':len(cases),'successful_requests':len(completed),
        'failed_attempts':sum(r['status']!='ok' for r in attempts),'http_attempts':len(attempts),
        'probability_rmse':float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        'returned_models':sorted({r['response']['model'] for r in completed.values()}),
        'paired_metrics':pair_results,'by_family':by_family,
        'note':'Partial coverage is not a full benchmark result. This cannot identify a hidden training recipe.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,default=Path('results/forecast-audit-v1/evaluation/benchmark'))
    parser.add_argument('--output',type=Path,default=Path('output/jev-probability-probes.jsonl'))
    parser.add_argument('--model',default='jev-latest')
    parser.add_argument('--limit',type=int,default=24,help='Maximum new HTTP attempts in this invocation')
    parser.add_argument('--execute',action='store_true',help='Send requests using TYPESAFE_API_KEY')
    parser.add_argument('--summarize',action='store_true',help='Score previously collected responses without network calls')
    args=parser.parse_args()
    if args.limit<1:
        parser.error('The request limit must be positive')
    if args.summarize:
        print(json.dumps(summarize(args.benchmark,args.output),indent=2));return
    cases=read_jsonl(args.benchmark/'requests.jsonl.gz')
    # Keep pairs together and include both families before moving to the next group number.
    cases.sort(key=lambda r:(r['group'].rsplit('-',1)[-1],r['family'],r['variant'],r['text_version']))
    config={'dataset_sha256':digest(args.benchmark/'requests.jsonl.gz'),'model':args.model,'endpoint':ENDPOINT}
    metadata=args.output.with_suffix(args.output.suffix+'.meta.json')
    completed={}
    if args.output.exists():
        if not metadata.exists() or json.loads(metadata.read_text())!=config:
            parser.error('Existing output belongs to a different dataset or model')
        completed={r['id']:r for r in read_jsonl(args.output) if r['status']=='ok'}
        for case in cases:
            if case['id'] in completed and completed[case['id']]['payload_sha256']!=payload_digest(payload(case,args.model)):
                parser.error('An existing response has a different request digest')
    pending=[r for r in cases if r['id'] not in completed][:args.limit]
    if not args.execute:
        print(json.dumps({'mode':'preview_only','network_calls':0,'pending_requests_in_limit':len(pending),
            'completed_requests':len(completed),'total_benchmark_requests':len(cases),
            'request_characters':sum(len(json.dumps(payload(r,args.model))) for r in pending),
            'example_request':payload(pending[0],args.model) if pending else None},indent=2));return
    key=os.environ.get('TYPESAFE_API_KEY')
    if not key:
        parser.error('TYPESAFE_API_KEY is not configured; no requests were sent')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not metadata.exists():
        write_json(metadata,config)
    with args.output.open('a') as stream:
        for case in pending:
            value=payload(case,args.model)
            started=time.perf_counter()
            row={'id':case['id'],'requested_model':args.model,'payload_sha256':payload_digest(value),
                 'started_at':datetime.now(timezone.utc).isoformat()}
            try:
                response=send(value,key)
                q=parse_probability(response)
                row.update(status='ok',probability=q,response=response,seconds=time.perf_counter()-started)
            except (urllib.error.URLError,ValueError,KeyError,TypeError,TimeoutError) as exc:
                # Do not log request headers, credentials, or arbitrary server error bodies.
                row.update(status='error',error_type=type(exc).__name__,http_status=getattr(exc,'code',None),
                           seconds=time.perf_counter()-started)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                raise SystemExit('Request failed; the sanitized failure was saved. Resume after resolving access or service errors.') from None
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            print(f"{case['id']}: probability {q:.6f}",flush=True)
    print(json.dumps(summarize(args.benchmark,args.output),indent=2))


if __name__=='__main__':
    main()
