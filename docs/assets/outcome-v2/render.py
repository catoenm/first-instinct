"""Render one descriptive figure from the recovered outcome-v2 report; no inference."""

import argparse
import hashlib
import json
from pathlib import Path


REPORT_SHA256 = 'd3cb611be8519263243d8d3093f90b6eee949d3e31810cd653daddf79694787b'
RUNS = ('outcome-s77', 'outcome-s83', 'reward-s77', 'reward-s83', 'hybrid-s77', 'hybrid-s83')


def data(report_path):
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('This figure is bound to the reviewed recovered report.')
    report = json.loads(raw)
    if report['status'] != 'partial' or set(report['runs']) != set(RUNS):
        raise ValueError('Unexpected experiment coverage.')
    rows = []
    for name in RUNS:
        run = report['runs'][name]
        for role in ('best', 'latest'):
            checkpoint = run['checkpoints'][role]
            row = {'run': name, 'role': role, 'run_status': run['status'],
                   'status': checkpoint['status'],
                   'update': run['selected_update' if role == 'best' else 'updates']}
            if checkpoint['status'] == 'complete':
                row.update(reward={key: value['reward'] for key, value in checkpoint['policies'].items()},
                           exact_excess_brier={kind: values['exact_brier_excess'] for kind, values
                                              in checkpoint['audit']['macro']['marginals'].items()})
            else:
                if name != 'hybrid-s77' or checkpoint['status'] != 'missing':
                    raise ValueError('Unexpected missing/invalid checkpoint.')
            rows.append(row)
    return {'source_report_sha256': REPORT_SHA256, 'rows': rows,
            'scope': 'Descriptive means; all 12 roles. Identical best/latest weights are aliases, not replicates. '
                     'Realized policy reward and independent fixed-continuation forecast errors have different meanings. '
                     'Audit cost means include known singleton distributions.'}


def render(report_path, output):
    values = data(report_path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#c8ccd0',
                         'axes.labelcolor': '#25323a', 'text.color': '#25323a',
                         'xtick.color': '#47545c', 'ytick.color': '#25323a'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 8.2), sharey=True,
                                     gridspec_kw={'width_ratios': [1.2, 1]})
    blue, orange = '#166a94', '#bb6218'
    for index, row in enumerate(values['rows']):
        for axis in (left, right):
            if index // 2 % 2 == 0:
                axis.axhspan(index - .5, index + .5, color='#f3f5f6', zorder=0)
        if row['status'] != 'complete':
            left.text(.03, index, 'test missing', va='center', color='#777777')
            right.text(.03, index, 'test missing', va='center', color='#777777')
            continue
        left.scatter(row['reward']['controller'], index - .12, marker='o', s=40, color=blue, zorder=3)
        left.scatter(row['reward']['native_actor'], index + .12, marker='s', s=36, color=orange, zorder=3)
        right.scatter(row['exact_excess_brier']['outcome'], index - .12, marker='o', s=40, color=blue, zorder=3)
        right.scatter(row['exact_excess_brier']['cost'], index + .12, marker='s', s=36, color=orange, zorder=3)
    baseline = next(row['reward'] for row in values['rows'] if row['status'] == 'complete')
    left.axvline(baseline['fixed_continuation'], color='#767676', ls='--', lw=1.2)
    left.axvline(baseline['exact_controller'], color='#32383d', ls=':', lw=1.6)
    left.axvline(0, color='#c8ccd0', lw=.8)
    labels = [f"{row['run']} / {row['role']} ({row['update']})" for row in values['rows']]
    left.set_yticks(range(len(labels)), labels)
    left.set_ylim(len(labels) - .5, -.5)
    left.set_xlim(-.17, .47)
    right.set_xlim(0, .46)
    for axis in (left, right):
        axis.tick_params(axis='y', length=0, pad=10)
        axis.grid(axis='x', color='#e4e8eb', lw=.7)
        axis.set_axisbelow(True)
    left.set_title('Executed policy reward', loc='left', fontweight='bold', pad=14)
    right.set_title('Forecast distribution error', loc='left', fontweight='bold', pad=14)
    left.set_xlabel('Mean realized reward (higher is better)', labelpad=10)
    right.set_xlabel('Summed squared probability error (lower is better)', labelpad=10)
    dots = [Line2D([], [], marker='o', ls='', color=blue, markersize=6),
            Line2D([], [], marker='s', ls='', color=orange, markersize=6)]
    left.legend(dots + [Line2D([], [], color='#767676', ls='--'), Line2D([], [], color='#32383d', ls=':')],
                ['Forecast controller', 'Native actor', 'Fixed continuation', 'Exact one-step reference'],
                loc='lower left', bbox_to_anchor=(-.02, 1.04), ncol=2, frameon=False,
                fontsize=9, columnspacing=1.1, handlelength=1.6)
    right.legend(dots, ['Terminal outcome', 'Future cost'], loc='lower left',
                 bbox_to_anchor=(-.02, 1.04), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Outcome-v2: executed decisions and independent forecasts', x=.27, y=.974,
                 ha='left', fontsize=15, fontweight='bold')
    fig.text(.27, .936, 'All 12 checkpoint roles · selected update or latest update in parentheses', fontsize=10)
    fig.text(.27, .035, 'Equal environment means; 128 roots per environment in each evaluation stream.\n'
             'Repeated best/latest values can share weights. Hybrid seed 77 stopped before final tests.',
             fontsize=9, color='#5a646b', linespacing=1.6)
    fig.subplots_adjust(left=.27, right=.98, top=.795, bottom=.14, wspace=.20)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'figure-data.json').write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
    fig.savefig(output / 'decisions-and-forecasts.png', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    render(args.report, args.output)
