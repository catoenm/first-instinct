"""Recompute the explicitly post-hoc foundation/SFT comparison; no inference.

Use the recorded dependency versions required by the two provenance validators.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from general_lab.robustness_foundation import analyze_control
from general_lab.robustness_report import analyze


def reports():
    folder = ROOT / 'results/general-robustness-v1'
    supervised, supervised_receipt = analyze(folder / 'supervised', folder / 'corpus.json', folder / 'freeze.json')
    foundation, foundation_receipt = analyze_control(Path(__file__).resolve().parent)
    key = lambda row: (row['root_id'], row['world'], row['variant'])
    assert [key(x) for x in foundation['predictions']] == [key(x) for x in supervised['predictions']]
    assert [x['target'] for x in foundation['predictions']] == [x['target'] for x in supervised['predictions']]
    models = {'foundation': foundation, 'supervised': supervised}
    summary = {'scope': 'Post-hoc descriptive control proposed after seeing supervised results. Same frozen 48 authored roots/456 correlated questions; no checkpoint selection, tuning, broad generalization or Jev comparison.',
        'models': {name: {'deterministic_macro_accuracy': report['summary']['metrics']['accuracy'],
                         'forecast_probability_rmse': math.sqrt(report['summary']['metrics']['forecast_distribution_mse']),
                         'metrics': report['summary']['metrics'], 'families': report['families'],
                         'evidence_change': report['summary']['evidence_change'],
                         'invariance': report['summary']['invariance']} for name, report in models.items()},
        'provenance': {'foundation': foundation_receipt['input_sha256'],
                       'supervised': supervised_receipt['input_sha256']}}
    return models, summary


def plot(models, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    colors = {'foundation': '#969b9a', 'supervised': '#285c55'}
    labels = {'foundation': 'Untouched foundation', 'supervised': 'Supervised adapter'}
    families = ['routing', 'entailment', 'ordered_urgency']
    for index, (name, report) in enumerate(models.items()):
        positions = [i + (index - .5) * .32 for i in range(3)]
        values = [report['families'][family]['metrics']['accuracy'] * 100 for family in families]
        axes[0].bar(positions, values, width=.28, color=colors[name], label=labels[name])
        for x, value in zip(positions, values):
            axes[0].text(x, value + 1, f'{value:.1f}', ha='center', fontsize=9)
        rows = [row for row in report['predictions'] if row['family'] == 'finite_forecast']
        axes[1].scatter([r['target']['yes'] for r in rows], [r['probabilities']['yes'] for r in rows],
                        color=colors[name], label=labels[name], marker='x' if index == 0 else 'o', s=23, alpha=.6)
    axes[0].set_xticks(range(3), ['Routing', 'Entailment', 'Urgency'])
    axes[0].set_ylim(0, 119)
    axes[0].set_ylabel('Accuracy, mean across 12 roots per family (%)')
    axes[0].set_title('Deterministic decisions', loc='left', weight='bold')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
    axes[1].plot([0, 1], [0, 1], '--', color='#777777', linewidth=1)
    axes[1].set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel='Exact probability of orange', ylabel='Model probability of orange')
    axes[1].set_title('Probability forecasts', loc='left', weight='bold')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('First Instinct · post-hoc foundation control', weight='bold', fontsize=13)
    fig.supxlabel('Four small authored templates; correlated variants. This is not a broad capability benchmark.', fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    models, summary = reports()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(models, args.output / 'comparison.png')
    print(json.dumps({name: {k: value[k] for k in ('deterministic_macro_accuracy', 'forecast_probability_rmse')}
                      for name, value in summary['models'].items()}, indent=2))
