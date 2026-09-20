"""Render the verified, closed expanded study without loading model predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS={'outcome':('Forecast supervision','#3972a6'),
         'reward':('Reward learning','#ba7027'),
         'hybrid':('Combined','#7d589d')}


def render(audit, destination):
    if audit.get('status')!='passed' or audit['advancement']['status']!='complete':
        raise ValueError('Plots require a completed independent final audit')
    destination.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.4,5.6))
    points=[]
    for method,(label,color) in METHODS.items():
        rows=audit['advancement']['methods'][method]['seeds']
        if {r['seed'] for r in rows}!={1507,1609}:raise ValueError('Both paired seeds are required')
        for row in rows:
            x=row['macro_calendar_return_gain'];y=row['macro_expected_brier_reduction']
            points.append((x,y))
            ax.scatter(x,y,s=100,marker='o' if row['seed']==1507 else '^',color=color,
                       edgecolor='white',linewidth=.8,label=label if row['seed']==1507 else None,zorder=4)
    xlo=min(-.02,min(x for x,y in points)-.015);xhi=max(.06,max(x for x,y in points)+.025)
    ylo=min(-.015,min(y for x,y in points)-.015);yhi=max(.05,max(y for x,y in points)+.025)
    ax.set(xlim=(xlo,xhi),ylim=(ylo,yhi),xlabel='Decision return gain over the original model →',
           ylabel='Forecast probability error reduction →')
    ax.axvline(0,color='#ced1d4',lw=1);ax.axhline(0,color='#ced1d4',lw=1)
    ax.axvline(.03,color='#596268',ls='--',lw=1);ax.axhline(.02,color='#596268',ls='--',lw=1)
    ax.fill_between([.03,xhi],.02,yhi,color='#64a372',alpha=.10,zorder=0)
    ax.text(.035,yhi-.008,'Joint improvement region',ha='left',va='top',fontsize=9,color='#44754d')
    ax.set_title('Do better forecasts lead to better decisions?',loc='left',fontweight='bold',pad=15)
    ax.text(.025,.95,'Circle: seed 1507\nTriangle: seed 1609',transform=ax.transAxes,
            ha='left',va='top',fontsize=9,color='#5b6166')
    ax.legend(loc='best',frameon=False,fontsize=9)
    ax.grid(alpha=.12)
    fig.text(.12,.035,'Reserved calendar mechanism · three task structures · 80 episodes / 576 forecast inputs per model\n'
             'Both seeds must pass both thresholds and preserve general capabilities. Descriptive pilot; no error bars.',
             fontsize=8,color='#5b6166')
    fig.subplots_adjust(bottom=.21,top=.90,left=.12,right=.96)
    fig.savefig(destination/'joint-transfer.png',dpi=180)
    fig.savefig(destination/'joint-transfer.pdf')
    plt.close(fig)

    original=audit['arms']['original-test']['transfer']
    names=['original-test']+[f'{m}-{s}' for m in METHODS for s in (1507,1609)]
    labels=['Original']+[f'{METHODS[m][0]}\n{s}' for m in METHODS for s in (1507,1609)]
    colors=['#5b6166']+[METHODS[m][1] for m in METHODS for s in (1507,1609)]
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.1))
    for ax,key,title in zip(axes,('ambiguous','deterministic'),('Uncertain outcomes','Deterministic outcomes')):
        values=[audit['arms'][n]['transfer']['forecast'][key]['expected_brier'] for n in names]
        floor=original['forecast'][key]['irreducible_brier']
        ax.bar(range(len(names)),values,color=colors,width=.65)
        ax.axhline(floor,color='#405146',ls='--',lw=1,label='Verified uncertainty floor')
        ax.axhline(2/3,color='#736c62',ls=':',lw=1,label='Uniform three-way forecast')
        ax.set(xticks=range(len(names)),xticklabels=labels,ylabel='Expected Brier error (lower is better)',
               title=f"{title} · {original['forecast'][key]['n']} inputs")
        ax.tick_params(axis='x',labelsize=7,labelrotation=35)
        ax.set_ylim(0,max(values+[floor]) * 1.35)
        for i,v in enumerate(values):ax.text(i,v+.009,f'{v:.3f}',ha='center',va='bottom',fontsize=7)
        ax.legend(frameon=False,fontsize=8,loc='upper right')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Probability quality on the unfamiliar workflow',x=.07,ha='left',fontweight='bold')
    fig.text(.07,.025,'Each panel weights unique visible inputs equally. The joint gate instead averages the three task structures equally.\n'
             'The uncertainty floor comes from executed compatible worlds; it is a diagnostic reference, not a model prediction.',
             fontsize=8,color='#5b6166')
    fig.subplots_adjust(left=.07,right=.98,bottom=.29,top=.87,wspace=.28)
    fig.savefig(destination/'forecast-uncertainty.png',dpi=180)
    fig.savefig(destination/'forecast-uncertainty.pdf')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(json.loads(a.audit.read_text()),a.output)
