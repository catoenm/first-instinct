"""Summarize every seed and draw the learned-inspection comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from .inspection_train import write_json


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['recipe'],row['objective'],row['domain']),[]).append(row)
    summary = []
    for (recipe,objective,domain),items in sorted(groups.items()):
        metrics = {}
        for key,value in items[0].items():
            if isinstance(value,float):
                values = np.array([item[key] for item in items])
                metrics[key] = {'mean':float(values.mean()),'min':float(values.min()),
                                'max':float(values.max()),'standard_deviation':float(values.std(ddof=1))}
        summary.append({'recipe':recipe,'objective':objective,'domain':domain,
                        'seeds':[item['seed'] for item in items],'metrics':metrics})
    return summary


def figure(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = {'accuracy':'#bf6048','forecast_no_exploration':'#cf9a46',
              'forecast':'#286b8f','supervised':'#657e55'}
    labels = {'accuracy':'Correctness\nreward','forecast_no_exploration':'Forecast reward\nno early exploration',
              'forecast':'Forecast reward\nearly exploration','supervised':'Outcome labels\n+ fixed planner'}
    fig,axes = plt.subplots(1,2,figsize=(12.8,4.9),gridspec_kw={'width_ratios':[1,1.15]})
    for i,recipe in enumerate(colors):
        selected = [r for r in rows if r['recipe']==recipe and r['domain']=='in_distribution'
                    and r['objective']==('accuracy' if recipe=='accuracy' else 'forecast')]
        values = 100*np.array([r['audit_probability_rmse'] for r in selected])
        axes[0].bar(i,values.mean(),color=colors[recipe],width=.66,alpha=.85)
        axes[0].scatter(np.full(len(values),i)+np.linspace(-.10,.10,len(values)),values,
                        s=22,color='#172d3c',zorder=3)
    axes[0].set_xticks(range(4),[labels[k] for k in colors],fontsize=8)
    axes[0].set_ylabel('Probability error, percentage points\n(lower is better)')
    axes[0].set_title('Same audit states, different learning signals',loc='left',fontsize=12,pad=16)
    axes[0].set_ylim(bottom=0)
    domains = ['in_distribution','expensive_inspection','more_duplicates','reversed_new_source']
    names = ['Original','Higher prices','More copies','Reversed source']
    for offset,recipe in enumerate(('forecast_no_exploration','forecast','supervised')):
        values = [[100*r['oracle_regret'] for r in rows if r['recipe']==recipe and
                   r['objective']=='forecast' and r['domain']==domain] for domain in domains]
        x = np.arange(4)+(offset-1)*.25
        axes[1].bar(x,[np.mean(v) for v in values],width=.23,color=colors[recipe],
                    label=labels[recipe].replace('\n',' '),alpha=.85)
        for pos,v in zip(x,values):
            axes[1].scatter(np.full(len(v),pos)+np.linspace(-.035,.035,len(v)),v,s=14,color='#172d3c',zorder=3)
    axes[1].set_xticks(range(4),names,fontsize=9)
    axes[1].set_ylabel('Lost reward versus exact planner × 100\n(lower is better)')
    axes[1].set_title('Can the forecast policy use information well?',loc='left',fontsize=12,pad=16)
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    axes[1].set_ylim(0,1.4*max(100*r['oracle_regret'] for r in rows if r['objective']=='forecast'))
    for ax in axes:
        ax.grid(axis='y',alpha=.16)
        ax.set_axisbelow(True)
    fig.text(.05,.015,'Bars: three-seed mean. Dots: individual seeds. Numeric simulation; no text encoder.\n'
             'The correctness policy’s action probabilities are audited as event forecasts; correctness was its training objective.',
             fontsize=8.5,color='#56616c')
    fig.tight_layout(rect=[0,.10,1,1])
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        path = output/f'learned-inspection.{suffix}'
        fig.savefig(path,dpi=180,bbox_inches='tight')
        if suffix=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/learned-inspection-v1'))
    parser.add_argument('--figures',type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run/'evaluation'/'results.json').read_text())
    write_json(args.run/'summary.json',summarize(rows))
    if args.figures:
        figure(rows,args.figures)


if __name__=='__main__':
    main()
