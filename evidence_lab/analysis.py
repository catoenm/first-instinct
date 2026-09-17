"""Summarize all collection seeds, including weak-verifier counterexamples."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .data import read_rows,write_json
from .study import RECIPES


def summarize(root):
    root=Path(root);evaluation=root/'evaluation'
    results=json.loads((evaluation/'results.json').read_text());rows=read_rows(evaluation/'cases.jsonl.gz')
    outcomes={r['id']:r['passed'] for r in read_rows(evaluation/'answers.jsonl.gz')}
    cache={r['id']:r['cache_key'] for r in read_rows(evaluation/'acquisitions.jsonl')}
    summary=[]
    for domain in ('new_task','new_family','new_transformation'):
        for reference in ('text','visible_checks','label_prior'):
            for recipe in RECIPES:
                selected=[r for r in results if (r['domain'],r['reference'],r['recipe'])==(domain,reference,recipe)]
                values=[r['initial']['brier'] for r in selected]
                summary.append({'domain':domain,'reference':reference,'recipe':recipe,'seeds':[r['collection_seed'] for r in selected],
                    'initial_brier_mean':float(np.mean(values)),'initial_brier_seeds':values,
                    'new_check_brier_mean':float(np.mean([r['new_check']['brier'] for r in selected])),
                    'copy_change_mean':float(np.mean([r['copy_change'] for r in selected])),
                    'price_change_mean':float(np.mean([r['price_change'] for r in selected])),
                    'automatic_pass_counts':[r['initial']['automatic_pass_count'] for r in selected],
                    'automatic_pass_errors':[r['initial']['automatic_pass_errors'] for r in selected]})
    quality=[];counterexamples=[]
    for domain in ('new_task','new_family','new_transformation'):
        group=[r for r in rows if r['domain']==domain]
        weak=[r for r in group if all(c['passed'] for c in r['visible_checks'][:2])]
        misses=[r for r in weak if not outcomes[r['id']]]
        non_authored=[r for r in group if r['mechanism']!='authored']
        quality.append({'domain':domain,'tasks':len({r['task'] for r in group}),'candidates':len(group),
            'private_suite_passes':sum(outcomes[r['id']] for r in group),'initial_checks_all_pass':len(weak),
            'initial_pass_private_fail':len(misses),'mutation_proposals':len(non_authored),
            'mutations_passing_private_suite':sum(outcomes[r['id']] for r in non_authored),
            'initial_fail_private_pass':sum(outcomes[r['id']] and any(not c['passed'] for c in r['visible_checks'][:2]) for r in group),
            'visible_exception_candidates':sum(any(c['error'] for c in r['visible_checks']) for r in group)})
        for row in misses:
            verdict=json.loads((root/'verifications'/(cache[row['id']]+'.json')).read_text())
            counterexamples.append({'id':row['id'],'task':row['task'],'domain':domain,'contract':row['description'],
                'code':row['code'],'visible_checks':row['visible_checks'][:2],
                'first_private_failure':next(c for c in verdict['checks'] if not c['passed']),
                'private_failure_count':sum(not c['passed'] for c in verdict['checks'])})
    collection=[];run=json.loads((root/'run.json').read_text())
    for seed in run['seeds']:
        for recipe in RECIPES:
            folder=root/f'{recipe}-s{seed}';manifest=json.loads((folder/'manifest.json').read_text())
            records=read_rows(folder/'acquisitions.jsonl')
            collection.append({'recipe':recipe,'seed':seed,'private_passes':sum(r['passed'] for r in records),
                'queries':len(records),'logical_tests':sum(r['logical_test_executions'] for r in records),
                'standalone_verification_seconds':manifest['measured_standalone_verification_seconds'],
                'new_execution_seconds':manifest['new_execution_seconds']})
    dev=read_rows(root/'development-pool'/'cases.jsonl.gz');train=[r for r in dev if r['split']=='train']
    overlaps=sorted({r['code_sha256'] for r in train}&{r['code_sha256'] for r in rows})
    per_task=[];by_id={r['id']:r for r in rows};predictions=read_rows(evaluation/'predictions.jsonl.gz')
    grouped={}
    for p in predictions:
        row=by_id[p['id']];key=(p['model'],p['reference'],row['domain'],row['task'])
        grouped.setdefault(key,[]).append((p['probabilities'][0]-outcomes[p['id']])**2)
    for (model,reference,domain,task),errors in sorted(grouped.items()):
        per_task.append({'model':model,'reference':reference,'domain':domain,'task':task,'cases':len(errors),
                         'initial_brier':float(np.mean(errors))})
    return {'summary':summary,'data_quality':quality,'collection':collection,'counterexamples':counterexamples,
            'task_metrics':per_task,'exact_training_final_code_overlaps':overlaps,
            'train_candidates_by_task':dict(sorted(Counter(r['task'] for r in train).items())),
            'notes':['Candidate-weighted scores; related candidates share a task.',
                     'Pass rates describe this synthetic proposal population, not production software.',
                     'No human audit, Jev calls, encoder fine-tuning or reinforcement learning in this pilot.']}


def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),sharey=True)
    colors={'text':'#176BBD','visible_checks':'#538441','label_prior':'#B86E23'}
    names={'text':'Frozen language encoder','visible_checks':'Visible-test reference','label_prior':'Label-frequency reference'}
    titles={'new_task':'New tasks · 5 tasks','new_family':'New family · 4 tasks','new_transformation':'New edits · 9 tasks'}
    for ax,(domain,title) in zip(axes,titles.items()):
        for reference,offset in zip(colors,(-.12,0,.12)):
            group=[r for r in result['summary'] if r['domain']==domain and r['reference']==reference]
            xs=np.arange(3)+offset;means=[r['initial_brier_mean'] for r in group]
            ax.plot(xs,means,color=colors[reference],lw=1.6,label=names[reference])
            for x,row in zip(xs,group):
                ax.scatter([x]*3,row['initial_brier_seeds'],s=22,alpha=.7,color=colors[reference],edgecolors='white',linewidths=.4)
        ax.set_title(title,loc='left',fontsize=11,pad=15)
        ax.set_xticks(range(3),['Random','Coverage','Adaptive']);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Mean squared probability error (Brier score) ↓')
    fig.suptitle('Does choosing the data help at 100 verified programs?',x=.075,ha='left',fontweight='bold',fontsize=15)
    axes[0].legend(loc='upper left',fontsize=8,frameon=False)
    fig.text(.075,.015,'Each dot is one collection seed. Same frozen encoder and label budget; authored Python tasks, not real repositories.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[.01,.06,1,.92]);Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('results/executable-evidence-v1'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--figure',type=Path);a=p.parse_args()
    result=summarize(a.run);write_json(a.output,result)
    if a.figure:plot(result,a.figure)
    print(json.dumps({'summary':result['summary'],'data_quality':result['data_quality']},indent=2))


if __name__=='__main__':main()
