"""Recompute and visualize the supplementary audit from saved local responses."""

import argparse
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, file_hash, write_json
from .robustness import run, verify_corpus


def analyze(folder, corpus_path, freeze_path):
    folder = Path(folder)
    receipt = json.loads((folder / 'run.json').read_text())
    if receipt.get('status') != 'complete' or receipt['freeze_sha256'] != file_hash(freeze_path):
        raise ValueError('Require a completed run with the expected frozen protocol')
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze.get('schema') != 'typed-robustness-local-v1' or freeze.get('selection_role') != 'none':
        raise ValueError('Unrecognized supplementary evaluation freeze')
    if not freeze.get('code_sha256') or not freeze.get('adapter_files_sha256'):
        raise ValueError('Frozen source and adapter provenance are required')
    for relative, expected in freeze['code_sha256'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_hash(ROOT / relative) != expected:
            raise ValueError('Frozen scorer/source mismatch: ' + relative)
    if receipt.get('expected_server_metadata') != freeze['expected_server_metadata']:
        raise ValueError('Run model metadata differs from the frozen model')
    if receipt.get('adapter_files_sha256') != freeze['adapter_files_sha256']:
        raise ValueError('Run adapter identity differs from the frozen artifacts')
    if freeze['corpus_file_sha256'] != file_hash(corpus_path):
        raise ValueError('Corpus bytes differ from the frozen evaluation')
    corpus = json.loads(Path(corpus_path).read_text())
    verified = verify_corpus(corpus)
    if verified['sha256'] != freeze['corpus_content_sha256'] or receipt['corpus'] != verified:
        raise ValueError('Corpus identity differs from the run receipt')
    responses = [json.loads(line) for line in (folder / 'responses.jsonl').read_text().splitlines()]
    if (len(responses) != verified['questions'] or
            any(not isinstance(r, dict) or type(r.get('index')) is not int for r in responses) or
            [r['index'] for r in responses] != list(range(len(responses)))):
        raise ValueError('Responses must cover every declared question exactly once in order')
    if type(receipt.get('questions')) is not int or receipt['questions'] != verified['questions']:
        raise ValueError('Completed receipt question count does not match the response stream')
    for item in responses:
        milliseconds = item.get('milliseconds')
        if type(milliseconds) not in (int, float) or not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError('Response milliseconds must be finite nonnegative numbers')
    model_seconds = sum(item['milliseconds'] for item in responses) / 1000
    for name in ('seconds', 'model_seconds'):
        value = receipt.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Recorded duration must be finite and nonnegative')
    if not math.isclose(model_seconds, receipt['model_seconds'], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('Model duration does not reproduce from the response stream')
    cursor = 0
    def saved_predictions(items):
        nonlocal cursor
        selected = responses[cursor:cursor + len(items)]
        cursor += len(items)
        return [r['probabilities'] for r in selected]
    report = run(corpus, saved_predictions)
    if report != json.loads((folder / 'metrics.json').read_text()):
        raise ValueError('Saved aggregate metrics do not reproduce from the response stream')
    summary = {'corpus': verified, 'model': receipt['expected_server_metadata'],
               'seconds': receipt['seconds'], 'model_seconds': model_seconds,
               'summary': report['summary'], 'families': report['families'],
               'log_probability_floor': report['log_probability_floor'],
               'input_sha256': {name: file_hash(folder / name) for name in ('run.json', 'responses.jsonl', 'metrics.json')},
               'freeze_sha256': file_hash(freeze_path), 'corpus_file_sha256': file_hash(corpus_path)}
    return report, summary


def markdown(summary):
    labels = {'routing': 'Routing rules', 'entailment': 'Partial-knowledge entailment', 'ordered_urgency': 'Ordered urgency'}
    model = summary['model']
    checkpoint = model['checkpoint']
    lines = ['# Supplementary typed-question audit', '',
             f"One {checkpoint['kind']} {model['model']} checkpoint, step {checkpoint['step']:,}. No model comparison or checkpoint selection.", '',
             '| Deterministic family | Roots | Questions | Mean root accuracy | Clipped log loss | Brier score |',
             '|---|---:|---:|---:|---:|---:|']
    for family, label in labels.items():
        entry = summary['families'][family]; m = entry['metrics']
        lines.append(f"| {label} | {entry['roots']} | {entry['questions']} | {m['accuracy']:.1%} | {m['log_loss']:.3f} | {m['brier']:.3f} |")
    entry = summary['families']['finite_forecast']; m = entry['metrics']
    lines += ['', f"Finite-experiment forecasts: {entry['roots']} roots / {entry['questions']} questions. "
              f"Modal accuracy {m['forecast_modal_accuracy']:.1%}; probability root mean squared error "
              f"{math.sqrt(m['forecast_distribution_mse']):.1%}. The latter measures error against the known "
              'conditional distribution, not error against sampled outcomes.', '',
              f"Exact expected clipped log score {m['forecast_exact_expected_log_loss']:.3f}; "
              f"exact expected Brier score {m['forecast_exact_expected_brier']:.3f}. "
              'Even a perfect forecast has positive expected loss for a random event.', '',
              '| Equivalent variant | Mean total variation | Answer flips |', '|---|---:|---:|']
    for name, value in summary['summary']['invariance'].items():
        lines.append(f"| {name.replace('_', ' ')} | {value['total_variation']:.1%} | {value['answer_flip']:.1%} |")
    paired = summary['summary']['evidence_change']
    lines += ['', f"Both required modal answers correct on evidence-change pairs: "
              f"{paired['both_required_modal_answers_correct']:.1%} (root-averaged across variants and all four families).", '',
              'Opaque option-ID changes produce identical model prompts by construction. Their consistency tests '
              'serialization and response mapping, not learned semantic robustness.', '',
              'All 456 questions are correlated views of 48 authored configurations from four templates. '
              'These are supplementary diagnostics, not broad calibration evidence or an independent benchmark. '
              'No inference about improvement over the foundation or parity with Jev follows from this run.', '',
              'The response stream records question order by index. Model metadata and separately recorded disk hashes '
              'are checked against the freeze; they do not attest to tensors resident in server memory.', '',
              f"Log scores clip probabilities at {summary['log_probability_floor']:.0e}. Every metric above was recomputed from the complete saved response stream.", '']
    return '\n'.join(lines)


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    labels = ['Routing', 'Entailment', 'Urgency']
    families = ['routing', 'entailment', 'ordered_urgency']
    values = [report['families'][name]['metrics']['accuracy'] * 100 for name in families]
    axes[0].barh(labels, values, color='#285c55', height=.5)
    axes[0].set_xlim(0, 108); axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_xlabel('Accuracy, mean across 12 roots per family (%)')
    axes[0].set_title('Explicit rules and partial knowledge', loc='left', weight='bold')
    for index, value in enumerate(values):
        axes[0].text(value + 1, index, f'{value:.1f}%', va='center', fontsize=9)
    forecast = [row for row in report['predictions'] if row['family'] == 'finite_forecast']
    for variant, marker, color in [('base', 'o', '#285c55'), ('wording', 's', '#98701a'), ('opaque_ids', '+', '#777777'),
                                   ('irrelevant_metadata', '^', '#ac5645'), ('reordered', 'x', '#55519b')]:
        rows = [row for row in forecast if row['variant'] == variant]
        axes[1].scatter([r['target']['yes'] for r in rows], [r['probabilities']['yes'] for r in rows],
                        label=variant.replace('_', ' '), marker=marker, color=color, s=28, alpha=.75)
    axes[1].plot([0, 1], [0, 1], '--', color='#888888', linewidth=1)
    axes[1].set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel='Exact probability of orange', ylabel='Model probability of orange')
    axes[1].set_title('Probability forecasts on specified experiments', loc='left', weight='bold')
    axes[1].legend(fontsize=8, loc='upper left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    figure.suptitle('First Instinct · supplementary typed-question audit', fontsize=13, weight='bold')
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--corpus', type=Path, default=Path('results/general-robustness-v1/corpus.json'))
    p.add_argument('--freeze', type=Path, default=Path('results/general-robustness-v1/freeze.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report, summary = analyze(args.run, args.corpus, args.freeze)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'summary.json', summary)
    (args.output / 'report.md').write_text(markdown(summary))
    plot(report, args.output / 'robustness.png')
    print(json.dumps(summary['summary'], indent=2))


if __name__ == '__main__':
    main()
