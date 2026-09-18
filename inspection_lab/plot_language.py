"""Render the saved language-model measurements; requires matplotlib 3.11.2."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False,
                         'axes.edgecolor': '#d0d7d2', 'text.color': '#22312a', 'axes.labelcolor': '#52645a',
                         'xtick.color': '#52645a', 'ytick.color': '#22312a'})
    labels = ['Original foundation', 'Evidence-frequency reference', 'Adapted model', 'Adapted + temperature']
    colors = ['#939f99', '#c7a877', '#7bb69c', '#245b47']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for ax, split, title in zip(axes, ('test', 'challenge'), ('Held-out source groups', 'Held-out strings and ciphers')):
        methods = report['results'][split]
        values = [methods['base']['brier'], report['empirical_evidence_references'][split]['common_states']['brier'],
                  methods['trained']['brier'], methods['trained_temperature_scaled']['brier']]
        ax.barh(np.arange(4), values, color=colors, height=.62)
        ax.set_yticks(np.arange(4), labels)
        ax.invert_yaxis(); ax.set_xlim(0, .245)
        ax.set_title(title, loc='left', pad=18, fontweight='bold')
        ax.set_xlabel('Brier score · lower is better', labelpad=12)
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0, pad=10)
        for i, value in enumerate(values):
            ax.text(value + .004, i, f'{value:.4f}', va='center', fontsize=9)
    fig.suptitle('Forecasts improve on verified software outcomes', x=.02, ha='left', fontsize=16, fontweight='bold')
    fig.text(.02, .02, '128 candidates × 7 correlated evidence views per split. One adapter seed. Temperature fitted on six separate validation groups.', fontsize=9, color='#66756c')
    fig.tight_layout(rect=(0, .07, 1, .92), w_pad=2)
    for extension in ('png', 'svg'):
        fig.savefig(args.out / f'forecast-quality.{extension}', dpi=180, facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 5.2), sharex=True, sharey=True)
    for ax, split, title in zip(axes, ('test', 'challenge'), ('Held-out source groups', 'Held-out strings and ciphers')):
        ax.plot([0, 1], [0, 1], '--', color='#bdc6c0', lw=1, label='Equal forecast and frequency')
        for key, label, color in (('base', 'Original foundation', '#939f99'),
                                  ('trained_temperature_scaled', 'Adapted + temperature', '#245b47')):
            bins = report['results'][split][key]['bins']
            x = [b['forecast'] for b in bins]; y = [b['frequency'] for b in bins]
            ax.plot(x, y, color=color, alpha=.7, lw=1)
            ax.scatter(x, y, s=[14 + b['n']*.22 for b in bins], color=color, label=label, zorder=3)
        ax.set_title(title, loc='left', pad=14, fontweight='bold')
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
        ax.set_xlabel('Mean forecast in probability bin')
        ax.grid(alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel('Observed pass frequency')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Better forecasts still leave calibration gaps', x=.03, ha='left', fontsize=16, fontweight='bold')
    fig.text(.03, .025, 'Marker area reflects bin count. Views within a candidate are correlated; these are descriptive bins, not independent trials.', fontsize=9, color='#66756c')
    fig.tight_layout(rect=(0, .07, 1, .92), w_pad=2)
    for extension in ('png', 'svg'):
        fig.savefig(args.out / f'calibration-bins.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
