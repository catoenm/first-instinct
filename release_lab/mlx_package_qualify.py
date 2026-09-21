"""Fresh-process package reload and complete, bounded development regression."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import resource
import statistics
import time

from scale_lab.common import file_hash,read_rows,write_json,write_rows
from release_lab.mlx_package import load_package
from release_lab.mlx_scorer import probabilities
from release_lab.pilot_metrics import summarize


def run(package,reference,data,cuda_reference,output):
    import mlx.core as mx
    if output.exists():raise ValueError('Preserve earlier qualifications')
    output.mkdir(parents=True,exist_ok=False)
    source=json.loads((reference/'report.json').read_text())
    if source['status']!='smoke_passed':raise ValueError('Eight-bit smoke qualification required')
    rows=read_rows(data/'development.jsonl');by_id={r['id']:r for r in rows}
    if len(rows)!=6072 or len(by_id)!=6072 or any(r['role']!='development' for r in rows):
        raise ValueError('Development cohort differs')
    probes=read_rows(reference/'predictions-private.jsonl')
    files=[Path(__file__),Path('release_lab/mlx_package.py'),Path('release_lab/mlx_scorer.py'),
           Path('docs/mlx-package-v1-protocol.md'),Path('release_lab/pilot_metrics.py'),
           package/'package.json',reference/'report.json',reference/'predictions-private.jsonl',data/'development.jsonl']
    baseline={}
    for suite in sorted({r['suite'] for r in rows}):
        path=cuda_reference/f'0-{suite}-predictions.jsonl';files.append(path)
        baseline.update({r['id']:r['probabilities'] for r in read_rows(path)})
    if set(baseline)!=set(by_id):raise ValueError('CUDA reference differs')
    plan=dict(version='mlx-package-v1',package_sha256=file_hash(package/'package.json'),
              paths={str(p.resolve()):file_hash(p) for p in files},reload_probe_ids=[r['id'] for r in probes],
              full_development_questions=6072,max_seconds=7200,max_tokens=4096,
              gates=dict(accuracy_drop=.005,log_loss_increase=.02,slice_accuracy_drop=.01,
                         brier_increase=.005,outcome_log_loss_increase=.01,agreement=.99,
                         mean_max_delta=.01,worst_max_delta=.20,peak_memory_bytes=10*1024**3))
    write_json(output/'plan-private.json',plan);start=time.monotonic()
    status=dict(status='loading',completed_questions=0,started_at=time.time(),plan_sha256=file_hash(output/'plan-private.json'))
    def save(**values):
        status.update(values);write_json(output/'run.json',status)
        print(json.dumps(values),flush=True)
    try:
        save(status='loading')
        # load_package has no foundation-directory parameter and reads only its
        # manifest-listed base and adapter artifacts, plus installed model code.
        model,manifest=load_package(package)
        save(status='reload_check',load_seconds=time.monotonic()-start)
        mx.clear_cache();mx.reset_peak_memory();reloaded=[]
        for probe in probes:
            row=by_id[probe['id']];p=probabilities(model,row['input_ids'],len(row['option_ids']))
            q=probe['probabilities'];delta=max(abs(a-b) for a,b in zip(p,q))
            agrees=max(range(len(p)),key=p.__getitem__)==max(range(len(q)),key=q.__getitem__)
            reloaded.append(dict(id=row['id'],probabilities=p,max_delta=delta,choice_agrees=agrees))
        write_rows(output/'reload-predictions-private.jsonl',reloaded)
        reload_report=dict(questions=len(reloaded),max_delta=max(r['max_delta'] for r in reloaded),
                           choice_agreement=all(r['choice_agrees'] for r in reloaded),
                           process_peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        reload_report['passed']=reload_report['choice_agreement'] and reload_report['max_delta']<=1e-5
        write_json(output/'reload.json',reload_report)
        if not reload_report['passed']:raise ValueError('Package reload equivalence failed')
        # This extra call measures exact-limit memory only; it has no truth label
        # and does not enter development metrics or agreement counts.
        longest=max(rows,key=lambda r:len(r['input_ids']))
        memory_ids=longest['input_ids']+[longest['input_ids'][-1]]*(4096-len(longest['input_ids']))
        probabilities(model,memory_ids,len(longest['option_ids']))
        save(status='evaluating',synthetic_memory_probe_tokens=4096,reload=reload_report)
        predictions={};latencies=[]
        with (output/'development-predictions-private.jsonl').open('x') as stream:
            for index,row in enumerate(sorted(rows,key=lambda r:(len(r['input_ids']),r['id']))):
                if time.monotonic()-start>7200:raise TimeoutError('Local evaluation deadline')
                began=time.perf_counter();p=probabilities(model,row['input_ids'],len(row['option_ids']));elapsed=time.perf_counter()-began
                if any(not math.isfinite(x) or x<0 for x in p) or not math.isclose(sum(p),1,abs_tol=1e-5):
                    raise ValueError('Invalid probability distribution')
                result=dict(id=row['id'],suite=row['suite'],tokens=len(row['input_ids']),probabilities=p,seconds=elapsed)
                predictions[row['id']]=result;latencies.append(elapsed)
                stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
                if (index+1)%50==0:
                    save(completed_questions=index+1,elapsed_seconds=time.monotonic()-start,last_tokens=len(row['input_ids']))
        suites=defaultdict(list)
        for row in rows:suites[row['suite']].append(row)
        current={s:summarize(rs,[predictions[r['id']]['probabilities'] for r in rs]) for s,rs in suites.items()}
        original={s:summarize(rs,[baseline[r['id']] for r in rs]) for s,rs in suites.items()}
        differences=[];agreed=0
        for row in rows:
            p,q=predictions[row['id']]['probabilities'],baseline[row['id']]
            differences.append(max(abs(a-b) for a,b in zip(p,q)))
            agreed+=max(range(len(p)),key=p.__getitem__)==max(range(len(q)),key=q.__getitem__)
        checks={}
        for suite in ('general','tools'):
            a,b=current[suite]['macro'],original[suite]['macro']
            checks[suite]=a['accuracy']>=b['accuracy']-.005 and a['log_loss']<=b['log_loss']+.02
        checks['product_slices']=all(v['accuracy']>=original['general']['slices'][s]['accuracy']-.01 for s,v in current['general']['slices'].items())
        checks['application_decisions']=current['application_decisions']['macro']['accuracy']>=original['application_decisions']['macro']['accuracy']
        checks['outcomes']=all(v['brier']<=original['outcomes']['by_group'][g]['brier']+.005 and
                               v['log_loss']<=original['outcomes']['by_group'][g]['log_loss']+.01 for g,v in current['outcomes']['by_group'].items())
        agreement=agreed/len(rows);mean=statistics.mean(differences);worst=max(differences);peak=mx.get_peak_memory()
        checks['probability_drift']=agreement>=.99 and mean<=.01 and worst<=.20
        checks['inference_memory']=peak<=10*1024**3
        report=dict(status='regression_passed' if all(checks.values()) else 'regression_failed',checks=checks,
                    metrics=current,reference_metrics=original,questions=len(rows),reload_questions=len(probes),synthetic_memory_questions=1,
                    choice_agreement=agreement,mean_max_delta=mean,worst_max_delta=worst,peak_inference_bytes=peak,
                    process_peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    median_seconds=statistics.median(latencies),total_seconds=time.monotonic()-start,
                    package_sha256=plan['package_sha256'],plan_sha256=file_hash(output/'plan-private.json'),
                    new_training_presentations=0,optimizer_steps=0,public_serving_qualified=False,
                    limitation='Existing released adapter converted locally; no new training, public deployment or Mac mini benchmark. Serving request limits still require qualification.')
        write_json(output/'report.json',report)
        save(status=report['status'],completed_questions=len(rows),checks=checks,finished_at=time.time())
        return report
    except BaseException as error:
        save(status='failed',error_type=type(error).__name__,error=str(error),finished_at=time.time())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','reference','data','cuda-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();report=run(**vars(args));print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','reference_metrics')},indent=2))
