"""Portable read-only report for the single frozen option-order audit.

Rescoring needs only the standard library. Matplotlib is optional via --plot.
No model, tokenizer, HTTP, service, or tool execution is performed.
"""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_lab import toolsandbox_transfer_order as frozen_api
from scale_lab.common import file_hash, write_json

FREEZE_SHA256 = 'f985639f18fdf6e90bd5158f759e47a5164dbf0f170b0b770216389f1265a54e'
FREEZE_CONTENT = 'afe69773a751c941b0bec11bc2f489232b18e573049fcb464ff7f53de66817e0'
INPUTS = ('freeze.json', 'questions.json', 'encoded-inputs.json', 'results/run.json',
          'results/attempts.jsonl', 'results/received.jsonl', 'results/responses.jsonl',
          'results/validated.jsonl', 'results/metrics.json', 'results/report.md')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def input_hashes(folder):
    return {name: file_hash(Path(folder) / name) for name in INPUTS}


def coverage(report):
    require(report['counts'] == {'roots': 48, 'new_reversed_predictions': 240,
            'original_predictions_reused': 240, 'deterministic_root_stop_costs': 48}, 'Incomplete audit coverage')
    rows = report['question_drifts']
    require(len(rows) == 240 and all(type(row['index']) is int and row['index'] == i for i, row in enumerate(rows))
            and len({row['question_id'] for row in rows}) == 240, 'Incomplete or reordered question records')
    roots = {row['root_id'] for row in rows}
    counts = Counter((row['root_id'], row['kind']) for row in rows)
    require(len(roots) == 48 and all(counts[root, 'outcome'] == 3 and counts[root, 'cost'] == 2 for root in roots),
            'Each root must retain three outcome and two cost questions')
    decisions = report['root_decisions']
    require(len(decisions) == 48 and {row['root_id'] for row in decisions} == roots, 'Incomplete paired root decisions')


def analyze(folder, reference_folder=None, corpus_folder=None):
    folder = Path(folder)
    before = input_hashes(folder)
    require(before['freeze.json'] == FREEZE_SHA256, 'Require the unchanged published option-order freeze')
    frozen = frozen_api.support.read(folder / 'freeze.json')
    require(frozen['content_sha256'] == FREEZE_CONTENT, 'Published freeze content changed')
    reference = Path(reference_folder) if reference_folder is not None else ROOT / 'results/toolsandbox-transfer-supervised-v1'
    corpus = Path(corpus_folder) if corpus_folder is not None else ROOT / 'results/toolsandbox-partial-v1'
    report = frozen_api.analyze(folder, reference_folder=reference, corpus_folder=corpus)
    coverage(report)
    require(input_hashes(folder) == before, 'Audit files changed during analysis; use a completed stable snapshot')
    receipt = frozen_api.support.read(folder / 'results/run.json')
    return {**report, 'report_provenance': {'schema': 'toolsandbox-option-order-public-report-v1',
        'status': 'verified_complete', 'freeze_sha256': FREEZE_SHA256, 'freeze_content_sha256': FREEZE_CONTENT,
        'input_sha256': before, 'wrapper_sha256': file_hash(__file__),
        'frozen_analyzer_sha256': file_hash(Path(frozen_api.__file__)),
        'model': frozen['binding']['expected_server_metadata'],
        'seconds': receipt['seconds'], 'model_seconds': receipt['model_seconds'],
        'note': 'All original and reversed records are retained. The frozen analyzer validates and recomputes them; this wrapper changes presentation only.'}}


def probability_points(report):
    """Pair every semantic option by ID, never by its presentation position."""
    coverage(report)
    result = {kind: {'original': [], 'reversed': [], 'questions': 0} for kind in ('outcome', 'cost')}
    for row in report['question_drifts']:
        old, new = row['original'], row['reversed']
        require(isinstance(old, dict) and isinstance(new, dict) and old and old.keys() == new.keys(), 'Semantic option IDs differ')
        for values in (old, new):
            require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values.values())
                    and math.isclose(sum(values.values()), 1., rel_tol=0, abs_tol=1e-6), 'Invalid plotted distribution')
        points = result[row['kind']]
        points['questions'] += 1
        points['original'].extend(old.values())
        points['reversed'].extend(new[key] for key in old)
    return result


def markdown(report, figure=None):
    lines = ['# One option-order reversal', '', report['scope'], '',
        'All 240 nontrivial initial-state questions across the same 48 authored roots are retained: 144 outcome forecasts and 96 future-cost forecasts. The original answers are reused from the completed reference; only reversed menus were queried again. The 48 singleton stop costs remain exact bypasses. Phone states are not compared.', '',
        '| Forecast | Questions | Mean total variation | Largest probability drift | Modal flips | Modal-set changes | Original / reversed ties |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind in ('outcome', 'cost'):
        row = report['by_kind'][kind]
        lines.append(f"| {kind} | {row['questions']} | {row['mean_total_variation']:.6f} | {row['max_absolute_drift']:.6f} | "
                     f"{row['modal_flips']} | {row['modal_set_changes']} | {row['original_ties']} / {row['reversed_ties']} |")
    lines += ['', 'Modal flips respect the first presented option when probabilities tie; modal-set changes distinguish changes in the tied set.', '',
        '| Presentation | Mean expected value | Mean expected regret | Exact-menu optimal action fraction |',
        '|---|---:|---:|---:|']
    for key, label in (('original', 'Original'), ('reversed', 'Reversed')):
        row = report['root_state_' + key]
        lines.append(f"| {label} | {row['expected_value']:.6f} | {row['expected_regret']:.6f} | {row['optimal_action_fraction']:.6f} |")
    lines += ['', f"Implied initial action changes: {sum(row['action_flip'] for row in report['root_decisions'])}/48. "
        'Values and regrets are in **research credits**, averaged equally across the paired 48 roots. They describe choosing once and following the fixed continuation, not newly executed policy returns.', '',
        '| Forecast | Original excess Brier | Reversed excess Brier | Original expected log loss | Reversed expected log loss |',
        '|---|---:|---:|---:|---:|']
    for kind in ('outcome', 'cost'):
        a, b = (report['root_forecasts_' + key][kind] for key in ('original', 'reversed'))
        lines.append(f"| {kind} | {a['excess_expected_brier']:.6f} | {b['excess_expected_brier']:.6f} | "
                     f"{a['exact_expected_clipped_log_loss']:.6f} | {b['exact_expected_clipped_log_loss']:.6f} |")
    lines += ['', 'Excess expected Brier is summed squared probability error against the stated finite distribution. Expected log loss uses the frozen probability floor. These are separate from option-order drift, which compares the two predictions directly.', '']
    if figure:
        lines += [f'![All semantic probabilities and paired-root means]({figure})', '',
            'Each scatter point is one semantic option, matched by ID across presentation orders; every question contributes all its options. Points on the diagonal have unchanged probabilities. The bars show equal-root means over the same 48 authored roots. Neither ordering is selected or promoted.', '']
    lines += [report['limits'], '',
        'This is one post-hoc, nonrandomized reversal at a later time, without contemporaneous original-order repeats. Temporal or numerical variation is not separately measured. The roots share one authored four-world mechanism, so this is not broad calibration evidence or 240 independent tasks.', '',
        'The JSON summary retains all question and root records plus the frozen analyzer’s full analysis provenance. '
        f"Available adapter: **{report['analysis_provenance']['adapter']}**; available tokenizer: **{report['analysis_provenance']['tokenizer']}**. "
        + report['analysis_provenance']['note'], '']
    return '\n'.join(lines)


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points = probability_points(report)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), layout='constrained')
        for axis, kind in zip(axes[:2], ('outcome', 'cost')):
            row = points[kind]
            axis.plot([0, 1], [0, 1], color='#a8adb3', linewidth=1, zorder=0)
            axis.scatter(row['original'], row['reversed'], s=13, alpha=.35, color='#245676', edgecolors='none')
            axis.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel='Original probability', ylabel='Reversed probability',
                     title=f"{kind.capitalize()} · {row['questions']} questions")
            axis.set_aspect('equal', adjustable='box')
        axis = axes[2]
        for offset, key, label, color in ((-.18, 'original', 'Original', '#8c9297'), (.18, 'reversed', 'Reversed', '#245676')):
            values = [report['root_state_' + key][name] for name in ('expected_value', 'expected_regret')]
            bars = axis.bar([i + offset for i in range(2)], values, width=.34, label=label, color=color)
            axis.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        axis.set_xticks([0, 1], ['Expected value', 'Expected regret'])
        axis.set_ylabel('Research credits')
        axis.set_title('Paired means · 48 authored roots')
        axis.axhline(0, color='#adb1b5', linewidth=.8)
        axis.margins(y=.18); axis.legend(frameon=False, fontsize=9)
        for axis in axes:
            axis.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Post-hoc option-order reversal · same supervised checkpoint', fontsize=13)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def write_report(folder, output, reference_folder=None, corpus_folder=None, include_plot=False):
    output = Path(output)
    require(not output.exists(), 'Output directory already exists; refusing overwrite')
    if include_plot and importlib.util.find_spec('matplotlib') is None:
        raise RuntimeError('Matplotlib is required only for --plot; omit it for standard-library rescoring')
    result = analyze(folder, reference_folder, corpus_folder)
    output.mkdir(parents=True, exist_ok=False)
    figure = 'option-order.png' if include_plot else None
    if figure:
        plot(result, output / figure)
    write_json(output / 'summary.json', result)
    (output / 'report.md').write_text(markdown(result, figure))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--reference-folder', type=Path)
    parser.add_argument('--corpus-folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    write_report(args.folder, args.output, args.reference_folder, args.corpus_folder, args.plot)


if __name__ == '__main__':
    main()
