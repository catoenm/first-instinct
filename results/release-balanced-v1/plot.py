"""Render the audited development learning curves; no private examples needed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT/'independent-audit.json').read_text())
steps = [0,80,160]
evaluations = [audit['evaluations'][str(n)] for n in steps]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'svg.hashsalt':'first-instinct-release-balanced-v1',
                     'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,3,figsize=(14,5))
fig.subplots_adjust(left=.065,right=.985,top=.70,bottom=.22,wspace=.32)
fig.text(.065,.94,'9B continuation: better forecasts, no release promotion',fontsize=19,weight='bold',color='#192c36')
fig.text(.065,.865,'160 supervised updates · 10,240 training presentations · 6,072 development questions',fontsize=11,color='#4d5b63')
for ax in axes:
    ax.set_xticks(steps)
    ax.set_xlabel('Continuation updates')
    ax.grid(axis='y',color='#e2e7e9',linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#34444e')

ax = axes[0]
ys = [e['outcomes']['macro']['brier'] for e in evaluations]
ax.plot(steps,ys,'o-',color='#147864',linewidth=2.3,markersize=6)
ax.set_title('Outcome forecast error',loc='left',weight='bold',pad=14)
ax.set_ylabel('Expected Brier score · lower is better')
ax.set_ylim(0,.4);ax.set_yticks([0,.1,.2,.3,.4])
ax.annotate(f'{ys[0]:.3f}',(0,ys[0]),xytext=(7,8),textcoords='offset points',color='#147864')
ax.annotate(f'{ys[-1]:.3f} (−57.2%)',(160,ys[-1]),xytext=(-8,-22),textcoords='offset points',ha='right',weight='bold',color='#147864')
ax.text(.02,.07,'Equal weight across 3 outcome groups',transform=ax.transAxes,fontsize=8.5,color='#59676f')

for ax,suite,title in zip(axes[1:],['general','tools'],['General decision accuracy','Tool-selection accuracy']):
    ys = [100*e[suite]['macro']['accuracy'] for e in evaluations]
    ax.plot(steps,ys,'o-',color='#245b85',linewidth=2.3,markersize=6)
    ax.set_title(title,loc='left',weight='bold',pad=14)
    ax.set_ylim(84,91);ax.set_yticks([84,86,88,90]);ax.set_ylabel('Macro accuracy (%)')
    floor = ys[0] + (-1 if suite == 'general' else 3)
    ax.axhline(floor,color='#a15c45',linestyle='--',linewidth=1.3)
    label = 'Retention floor' if suite == 'general' else 'Required improvement'
    ax.text(.02,.10 if suite == 'general' else .90,f'{label}: {floor:.2f}%',transform=ax.transAxes,fontsize=9,color='#924b36')
    ax.annotate(f'{ys[-1]:.2f}%',(160,ys[-1]),xytext=(-5,12),textcoords='offset points',ha='right',weight='bold',color='#245b85')
    ax.annotate(f'{ys[0]:.2f}%',(0,ys[0]),xytext=(5,12),textcoords='offset points',color='#245b85')

fig.text(.065,.08,'Development results from one adaptive run; reserved transfer was not evaluated.',fontsize=9,color='#59676f')
fig.text(.065,.035,'Promotion also requires log-loss, product-slice and per-outcome-group checks. The original checkpoint remains selected.',fontsize=9,color='#59676f')
fig.savefig(ROOT/'learning-curves.svg',metadata={'Date':None})
fig.savefig(ROOT/'learning-curves.png',dpi=140,metadata={'Software':'First Instinct / Matplotlib'})
plt.close(fig)
