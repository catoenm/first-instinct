"""Render the fixed, completed ToolSandbox cohort report without loading models."""

import argparse
import hashlib
import json
from pathlib import Path


REPORT_SHA256 = '1b3b4c71c7bfd82570155f7469bb77f83da18ad5100eb1a484b932b93fe0e037'
ROLE_IDS = tuple(f'{arm}-s{seed}/{role}' for arm in ('outcome', 'reward', 'hybrid')
                 for seed in (77, 83) for role in ('best', 'latest'))


def endpoints(summary):
    return {'decisions': {stratum: summary['decisions'][stratum]['forecast_controller']
                          for stratum in ('root_state', 'phone_prior_weighted')},
            'forecasts': {stratum: {kind: summary['forecasts'][stratum][kind]
                                    for kind in ('outcome', 'cost')}
                          for stratum in ('root_state', 'phone_prior_weighted')}}


def data(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Figure requires the reviewed, complete-eligible-cohort report.')
    report = json.loads(raw)
    if (report['schema'] != 'toolsandbox-transfer-cohort-report-v1'
            or report['status'] != 'partial'
            or report['role_status_counts'] != {'complete': 10, 'ineligible': 2}
            or tuple(role['id'] for role in report['roles']) != ROLE_IDS
            or len(report['units']) != 6
            or any(unit['status'] != 'complete' for unit in report['units'].values())):
        raise ValueError('Unexpected cohort coverage.')
    rows = [{'label': 'Supervised reference', 'roles': [], 'status': 'complete',
             **endpoints(report['supervised_reference']['summary'])}]
    seen = set()
    for role in report['roles']:
        identity = role['identity_sha256']
        if identity in seen:
            continue
        seen.add(identity)
        aliases = [r for r in report['roles'] if r['identity_sha256'] == identity]
        label = role['arm'].capitalize() + ' ' + str(role['seed'])
        label += ' selected/latest' if len(aliases) == 2 else ' ' + ('selected' if role['role'] == 'best' else 'latest')
        label += ' (' + str(role['update']) + ')'
        row = {'label': label, 'roles': [r['id'] for r in aliases], 'status': role['status'],
               'identity_sha256': identity, 'reason': role['reason']}
        if role['status'] == 'complete':
            row.update(endpoints(report['units'][identity]['summary']))
        elif role['status'] != 'ineligible':
            raise ValueError('Unexpected role status.')
        rows.append(row)
    if sorted(r for row in rows for r in row['roles']) != sorted(ROLE_IDS):
        raise ValueError('A role was duplicated or omitted.')
    return {'source_report_sha256': REPORT_SHA256, 'rows': rows,
            'scope': 'All 12 roles plus supervised reference; exact identities share a row. '
                     'Initial and phone-conditioned expected regrets are separate endpoints. '
                     'Forecast panels show initial-state excess expected Brier, excluding singleton costs. '
                     '48 related roots; one authored four-world mechanism; no adaptive policy execution.'}


def render(report_path, output):
    values = data(report_path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#c9d0d5',
                         'axes.labelcolor': '#25323a', 'text.color': '#25323a',
                         'xtick.color': '#47545c', 'ytick.color': '#25323a'})
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 7.3), sharey=True)
    specs = [('Initial decision\n(primary)', 'Expected regret\n(research credits)', (0, 25),
              lambda row: row['decisions']['root_state']['expected_regret']),
             ('After phone lookup\n(prior-weighted)', 'Expected regret\n(research credits)', (0, 50),
              lambda row: row['decisions']['phone_prior_weighted']['expected_regret']),
             ('Initial outcome\nforecast', 'Excess expected\nBrier score', (0, 1),
              lambda row: row['forecasts']['root_state']['outcome']['excess_expected_brier']),
             ('Initial cost\nforecast', 'Excess expected\nBrier score', (0, .65),
              lambda row: row['forecasts']['root_state']['cost']['excess_expected_brier'])]
    for axis, (title, xlabel, limits, metric) in zip(axes, specs):
        axis.axvline(metric(values['rows'][0]), color='#4f5a62', ls='--', lw=1, alpha=.8)
        for index, row in enumerate(values['rows']):
            if index % 2 == 0:
                axis.axhspan(index - .5, index + .5, color='#f3f5f6', zorder=0)
            if row['status'] == 'ineligible':
                axis.text(limits[1] * .08, index, 'not evaluated', color='#777777', va='center', fontsize=9)
                continue
            axis.scatter(metric(row), index, color='#26343f' if index == 0 else '#166a94',
                         marker='D' if index == 0 else 'o', s=42, zorder=3)
        axis.set_xlim(*limits)
        axis.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=15, linespacing=1.5)
        axis.set_xlabel(xlabel, labelpad=12, linespacing=1.5)
        axis.grid(axis='x', color='#e0e5e8', lw=.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis='y', length=0, pad=10)
    axes[0].set_yticks(range(len(values['rows'])), [row['label'] for row in values['rows']])
    axes[0].set_ylim(len(values['rows']) - .5, -.5)
    fig.suptitle('ToolSandbox transfer: no initial-decision gain', x=.255, y=.98,
                 ha='left', fontsize=16, fontweight='bold')
    fig.text(.255, .925, 'All six evaluated identities still stop at every initial state. Lower is better in all panels.', fontsize=10)
    fig.legend([Line2D([], [], ls='', marker='D', color='#26343f'),
                Line2D([], [], ls='', marker='o', color='#166a94'),
                Line2D([], [], ls='--', color='#4f5a62')],
               ['Supervised reference', 'Post-training checkpoint', 'Supervised-reference value'],
               loc='upper left', bbox_to_anchor=(.25, .905), ncol=3, frameon=False, fontsize=9)
    fig.text(.255, .038, '48 related roots from one authored mechanism. Values assume the fixed continuation; no adaptive policy was executed.\n'
             'Selected/latest aliases share rows; both ineligible Hybrid 77 roles remain visible. Phone forecast errors are in the report.',
             fontsize=9, color='#5a646b', linespacing=1.6)
    fig.subplots_adjust(left=.255, right=.985, top=.73, bottom=.195, wspace=.26)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'figure-data.json').write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
    fig.savefig(output / 'transfer-summary.png', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    render(args.report, args.output)
