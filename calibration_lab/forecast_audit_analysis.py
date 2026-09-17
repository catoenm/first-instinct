"""Aggregate all seeds and plot the prespecified forecast-practice comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from .inspection_train import write_json


def summarize(rows):
    grouped={}
    for row in rows:
        grouped.setdefault((row['recipe'],row['stage'],row['domain']),[]).append(row)
    result=[]
    for (recipe,stage,domain),values in sorted(grouped.items()):
        metrics={}
        for key,value in values[0].items():
            if isinstance(value,float):
                numbers=np.array([r[key] for r in values])
                metrics[key]={'mean':float(numbers.mean()),'min':float(numbers.min()),
                              'max':float(numbers.max()),'standard_deviation':float(numbers.std(ddof=1))}
        result.append({'recipe':recipe,'stage':stage,'domain':domain,
                       'seeds':[r['seed'] for r in values],'metrics':metrics})
    return result


def draw(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'terminal_only':'#bc624b','audit_early':'#bb984b','audit_continuous':'#247f9e','supervised':'#617f53'}
    labels={'terminal_only':'Chosen-path\nrewards only','audit_early':'Extra practice\nearly only',
            'audit_continuous':'Extra practice\nthroughout','supervised':'Event labels\nsame exercise states'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2))
    for index,recipe in enumerate(colors):
        selected=[r for r in rows if r['recipe']==recipe and r['domain']=='in_distribution']
        final=[r['audit_probability_rmse']*100 for r in selected if r['stage']=='final']
        axes[0].bar(index,np.mean(final),width=.65,color=colors[recipe],alpha=.85)
        axes[0].scatter(index+np.linspace(-.10,.10,len(final)),final,color='#233742',s=24,zorder=3)
        lines=[]
        for seed in sorted({r['seed'] for r in selected}):
            points=[next(r['duplicate_probability_change']*100 for r in selected if r['seed']==seed and r['stage']==stage)
                    for stage in ('quarter','final')]
            lines.append(points)
            axes[1].plot([0,1],points,color=colors[recipe],alpha=.25,linewidth=1)
        axes[1].plot([0,1],np.mean(lines,axis=0),color=colors[recipe],marker='o',linewidth=2.4,
                     label=labels[recipe].replace('\n',' '))
    axes[0].set_xticks(range(4),[labels[k] for k in colors],fontsize=9)
    axes[0].set_ylabel('Common-state probability error\n(percentage points; lower is better)')
    axes[0].set_title('Does continued practice improve forecasts?',loc='left',fontsize=12,pad=16)
    axes[1].set_xticks([0,1],['After first quarter','End of training'])
    axes[1].set_ylabel('Forecast change after an exact copy\n(percentage points; ideal is zero)')
    axes[1].set_title('Copies should add no information',loc='left',fontsize=12,pad=16)
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    peak=max(r['duplicate_probability_change']*100 for r in rows if r['domain']=='in_distribution')
    axes[1].set_ylim(0,peak*1.5)
    for ax in axes:
        ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.text(.045,.025,'Original test conditions. Bars and thick lines: three-seed means. Dots and thin lines: individual seeds.\n'
             'Early and continued practice use exactly the same exercise blocks. Workflow rewards are reported separately.',fontsize=9,color='#59626a')
    fig.tight_layout(rect=[0,.10,1,1])
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        path=output/f'forecast-practice.{suffix}'
        fig.savefig(path,dpi=180,bbox_inches='tight')
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/forecast-audit-v1'))
    parser.add_argument('--figures',type=Path)
    args=parser.parse_args()
    rows=json.loads((args.run/'evaluation'/'results.json').read_text())
    write_json(args.run/'summary.json',summarize(rows))
    if args.figures:draw(rows,args.figures)


if __name__=='__main__':main()
