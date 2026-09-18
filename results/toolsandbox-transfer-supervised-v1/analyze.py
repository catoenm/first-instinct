"""Recompute the frozen supervised ToolSandbox report from retained files only.

No model, tokenizer, ToolSandbox runtime, HTTP, or inference calls are imported.
"""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale_lab.common import digest, file_hash, write_json

FROZEN_CONTENT = '5226aa9222813cea65b1ab5535ac92264ebecf21ace790d86e9eeb9600e4434b'
FROZEN_FILE = 'cd83e94da4d800cb2d793a03da3f138a95b4f12a70e5d43fba3708e6b49d4ae0'
SCHEMA = 'toolsandbox-transfer-supervised-local-v1'
MODEL_LABEL = 'Supervised Qwen3.5-9B, step 2742'


def _json(path):
    return json.loads(Path(path).read_text())


def _journal(path):
    text = Path(path).read_text()
    if not text.endswith('\n') or any(not line.strip() for line in text.splitlines()):
        raise ValueError('Completed journals require complete nonblank JSON lines')
    return [json.loads(line) for line in text.splitlines()]


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + ' must be a finite nonnegative number')
    return value


def _file_map(folder):
    return {str(path.relative_to(folder)): file_hash(path)
            for path in sorted(Path(folder).rglob('*')) if path.is_file()}


def _available_files(folder, expected, label):
    """Absent cache/weights are disclosed; present but changed files fail closed."""
    folder = Path(folder)
    if not folder.exists():
        return {'status': 'unavailable', 'files': len(expected),
                'note': f'{label} not present locally; only recorded provenance was checked.'}
    for relative, value in expected.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe provenance file path')
        path = folder / relative
        if not path.is_file() or file_hash(path) != value:
            raise ValueError(label + ' differs from frozen provenance: ' + relative)
    return {'status': 'verified', 'files': len(expected)}


def _available_provenance(frozen, adapter_run=None, tokenizer=None):
    adapter = Path(adapter_run) if adapter_run is not None else Path(frozen['adapter_run'])
    token_path = Path(tokenizer) if tokenizer is not None else Path(frozen['prompt_provenance']['tokenizer'])
    if adapter_run is not None and not adapter.exists():
        raise ValueError('Explicit adapter directory does not exist')
    if tokenizer is not None and not token_path.exists():
        raise ValueError('Explicit tokenizer directory does not exist')
    result = {'adapter': _available_files(adapter, frozen['adapter_files_sha256'], 'Adapter'),
              'tokenizer': _available_files(token_path, frozen['prompt_provenance']['tokenizer_files'], 'Tokenizer')}
    installed = {}
    for name, expected in frozen['packages'].items():
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = None
        installed[name] = {'recorded_inference_version': expected, 'analysis_installed_version': value}
    result['runtime_versions'] = installed
    result['runtime_note'] = ('The inference package versions are frozen historical provenance. '
                              'Read-only rescoring does not require model libraries or claim the current environment performed inference.')
    return result


def analyze(run_folder, corpus_folder=None, *, adapter_run=None, tokenizer=None):
    """Require a complete original run and independently reproduce all metrics."""
    run_folder = Path(run_folder)
    frozen_path = run_folder / 'freeze.json'
    frozen = _json(frozen_path)
    if (file_hash(frozen_path) != FROZEN_FILE or frozen.get('content_sha256') != FROZEN_CONTENT
            or digest({k: v for k, v in frozen.items() if k != 'content_sha256'}) != FROZEN_CONTENT
            or frozen.get('schema') != SCHEMA or frozen.get('selection_role') != 'none'
            or frozen.get('model_inference') is not False):
        raise ValueError('Require the unchanged published pre-inference freeze')
    for relative, expected in frozen['code_sha256'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_hash(ROOT / relative) != expected:
            raise ValueError('Frozen source/protocol mismatch: ' + relative)
    reference_path = ROOT / 'results/general-robustness-v1/freeze.json'
    if file_hash(reference_path) != frozen['reference_sha256']:
        raise ValueError('Published supervised reference changed')
    reference = _json(reference_path)
    if (frozen['expected_server_metadata'] != reference['expected_server_metadata']
            or frozen['adapter_files_sha256'] != reference['adapter_files_sha256']):
        raise ValueError('Supervised checkpoint differs from the released reference')
    corpus_folder = Path(corpus_folder) if corpus_folder is not None else ROOT / 'results/toolsandbox-partial-v1'
    if _file_map(corpus_folder) != frozen['corpus_sha256']:
        raise ValueError('Corpus file coverage or hashes differ from the frozen evaluation')
    from general_lab.toolsandbox_transfer import load_corpus, model_questions, score, markdown as frozen_markdown
    corpus = load_corpus(corpus_folder)
    questions = model_questions(corpus)
    mapping = [{key: row[key] for key in ('index', 'question_index', 'question_id')}
               | {'input_sha256': digest(row['input'])} for row in questions]
    if (len(mapping) != 720 or _json(run_folder / 'question-mapping.json') != mapping
            or digest(mapping) != frozen['question_mapping_sha256']):
        raise ValueError('Frozen question order or input mapping changed')
    audit = _json(corpus_folder / 'prompt-audit-final.json')
    expected_prompt = {'audit_sha256': file_hash(corpus_folder / 'prompt-audit-final.json'),
        'tokenizer': audit['tokenizer'], 'tokenizer_files': audit['tokenizer_files'],
        'model_questions': 720, 'known_singletons': 144, 'min_tokens': audit['min_tokens'],
        'max_tokens': audit['max_tokens'], 'max_options': audit['max_options']}
    if frozen['prompt_provenance'] != expected_prompt:
        raise ValueError('Prompt/tokenizer provenance differs from the audit')
    output = run_folder / 'results'
    receipt = _json(output / 'run.json')
    if receipt.get('schema') != SCHEMA or receipt.get('status') != 'complete':
        raise ValueError('Incomplete or failed runs cannot be reported as complete')
    for key, expected in {'freeze_sha256': FROZEN_FILE, 'freeze_content_sha256': FROZEN_CONTENT,
                          'question_mapping_sha256': frozen['question_mapping_sha256'],
                          'expected_server_metadata': frozen['expected_server_metadata'],
                          'adapter_files_sha256': frozen['adapter_files_sha256'], 'scope': frozen['scope']}.items():
        if receipt.get(key) != expected:
            raise ValueError('Run provenance differs from the frozen identity: ' + key)
    for key in ('model_calls_planned', 'attempted_model_calls', 'successful_model_calls', 'received_model_responses'):
        if type(receipt.get(key)) is not int or receipt[key] != 720:
            raise ValueError('Complete run must record exactly 720 planned/attempted/received/validated questions')
    if type(receipt.get('known_singletons')) is not int or receipt['known_singletons'] != 144:
        raise ValueError('Singleton bypass count changed')
    start = _finite(receipt.get('started_at_unix'), 'Run start')
    end = _finite(receipt.get('completed_at_unix'), 'Run completion')
    seconds = _finite(receipt.get('seconds'), 'Run duration')
    recorded_model_seconds = _finite(receipt.get('model_seconds'), 'Model duration')
    if start < frozen['created_at_unix'] or end < start:
        raise ValueError('Invalid freeze/start/completion chronology')
    attempts, responses = _journal(output / 'attempts.jsonl'), _journal(output / 'responses.jsonl')
    if len(attempts) != 720 or len(responses) != 720:
        raise ValueError('Require exactly 720 attempts and responses')
    previous = start
    for index, (attempt, response, expected) in enumerate(zip(attempts, responses, mapping)):
        if (set(attempt) != {'index', 'input_sha256', 'started_at_unix'}
                or type(attempt['index']) is not int or attempt['index'] != index
                or attempt['input_sha256'] != expected['input_sha256']):
            raise ValueError('Attempt ordering or public-input checksum mismatch')
        timestamp = _finite(attempt['started_at_unix'], 'Attempt timestamp')
        if not previous <= timestamp <= end:
            raise ValueError('Attempt timestamps must be ordered within the recorded run')
        previous = timestamp
        if (set(response) != {'index', 'probabilities', 'milliseconds'}
                or type(response['index']) is not int or response['index'] != index):
            raise ValueError('Response ordering or schema mismatch')
        _finite(response['milliseconds'], 'Response duration')
    model_seconds = sum(row['milliseconds'] for row in responses) / 1000
    if not math.isclose(model_seconds, recorded_model_seconds, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('Model duration does not reproduce from response timings')
    if model_seconds > seconds + 1e-3:
        raise ValueError('Serial model durations exceed total recorded wall time')
    report = score(corpus, [row['probabilities'] for row in responses])
    if report != _json(output / 'metrics.json'):
        raise ValueError('Saved aggregates do not reproduce from the complete raw stream')
    if (output / 'report.md').read_text() != frozen_markdown(report, MODEL_LABEL):
        raise ValueError('Saved report prose differs from the frozen renderer')
    available = _available_provenance(frozen, adapter_run, tokenizer)
    summary = {'schema': 'toolsandbox-transfer-supervised-analysis-v1', 'status': 'verified_complete',
        'model': frozen['expected_server_metadata'], 'scope': frozen['scope'], 'accounting': report['accounting'],
        'summary': report['summary'], 'by_operation': report['by_operation'], 'by_collection_split': report['by_collection_split'],
        'seconds': seconds, 'model_seconds': model_seconds, 'log_probability_floor': report['log_probability_floor'],
        'corpus_content_sha256': report['corpus_content_sha256'], 'available_provenance': available,
        'freeze_sha256': FROZEN_FILE, 'freeze_content_sha256': FROZEN_CONTENT,
        'input_sha256': {str(Path('results') / name): file_hash(output / name)
                         for name in ('run.json', 'attempts.jsonl', 'responses.jsonl', 'metrics.json', 'report.md')},
        'mapping_sha256': file_hash(run_folder / 'question-mapping.json'), 'analyzer_sha256': file_hash(__file__),
        'provenance_limit': 'Frozen source and recorded identity plus available disk hashes, not an attestation of server-resident tensors.'}
    return report, summary


def markdown(summary):
    primary = summary['summary']['decisions']['root_state']
    phone = summary['summary']['decisions']['phone_prior_weighted']
    policies = (('forecast_controller', 'Supervised forecast selector'), ('stop', 'Always stop'),
                ('complete_query', 'Always complete query'), ('cheap_continuation', 'Cheap continuation'),
                ('uniform_action', 'Uniform random action'), ('exact_menu_optimum', 'Exact menu optimum'))
    lines = ['# Supervised ToolSandbox transfer reference', '',
             f"The frozen supervised Qwen3.5-9B checkpoint at step 2,742 completed all 720 questions in {summary['seconds'] / 60:.1f} minutes. "
             'All 48 authored roots were retained; 144 singleton cost answers bypassed prediction. '
             'No reinforcement-learning checkpoint is included in this reference.', '',
             'The primary endpoint is expected value at the initial state, averaged equally across roots. '
             'Each choice is followed by the fixed, fallible continuation in the protocol.', '',
             '| Selector | Initial expected value | Initial expected regret | Phone-conditioned expected regret |',
             '|---|---:|---:|---:|']
    for key, name in policies:
        lines.append(f"| {name} | {primary[key]['expected_value']:.3f} | {primary[key]['expected_regret']:.3f} | {phone[key]['expected_regret']:.3f} |")
    lines += ['', 'The exact menu optimum maximizes expected value under the same continuation and visible prior. '
              'It is not an omniscient hidden-world oracle. These are values implied by retained execution forecasts, '
              'not realized returns from running a learned policy. Phone histories are weighted by their probability within each root; '
              'already-paid lookup costs are excluded.', '',
              '| Forecast at initial state | Model questions | Summed probability error / excess Brier | Expected Brier | Expected clipped log loss |',
              '|---|---:|---:|---:|---:|']
    for kind in ('outcome', 'cost'):
        value = summary['summary']['forecasts']['root_state'][kind]
        count = summary['accounting']['forecast_model_questions_by_context']['root_state'][kind]
        lines.append(f"| {kind} | {count} | {value['excess_expected_brier']:.6f} | {value['exact_expected_brier']:.6f} | {value['exact_expected_clipped_log_loss']:.6f} |")
    lines += ['', 'Summed probability error avoids rewarding larger menus merely by dividing by more classes. '
              'Exact expected losses integrate the stated finite distribution and retain irreducible uncertainty. '
              f"Log scores clip probabilities at {summary['log_probability_floor']:.0e}. Singleton costs are excluded from learned forecast scores.", '',
              '| Operation | Roots | Initial selector regret | Prior-weighted phone selector regret |', '|---|---:|---:|---:|']
    for name, group in summary['by_operation'].items():
        a = group['decisions']['root_state']['forecast_controller']['expected_regret']
        b = group['decisions']['phone_prior_weighted']['forecast_controller']['expected_regret']
        lines.append(f"| {name} | {group['roots']} | {a:.3f} | {b:.3f} |")
    lines += ['', 'All original collection splits—including create/train—were evaluation-only. '
              'The mechanism, finite prior, costs and tool semantics are explicitly supplied. '
              'The 48 roots share one authored four-world mechanism; 720 questions and 144 histories are correlated views. '
              'This is not broad calibration evidence, a test of arbitrary tool use, or a comparison with Jev.', '',
              'The outcome/reward/hybrid checkpoints have not been compared in this report. '
              'This supervised reference cannot establish improvement due to executed-outcome training.', '',
              'Every aggregate was recomputed from all 720 retained responses, matched to the pre-request input-hash ledger and published mapping. '
              f"Available adapter files: **{summary['available_provenance']['adapter']['status']}**; "
              f"available tokenizer files: **{summary['available_provenance']['tokenizer']['status']}**. "
              + summary['provenance_limit'], '',
              '[Scoring protocol](toolsandbox-transfer-v1-protocol.md) · '
              '[Local run protocol](toolsandbox-transfer-local-v1-protocol.md) · '
              '[Frozen artifacts](../results/toolsandbox-transfer-supervised-v1/)', '']
    return '\n'.join(lines)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    labels = ['Outcome', 'Future cost']
    for offset, stratum, label, color in ((-.16, 'root_state', 'Initial state', '#285c55'),
                                         (.16, 'phone_prior_weighted', 'Phone, prior weighted', '#a87826')):
        values = [summary['summary']['forecasts'][stratum][kind]['excess_expected_brier'] for kind in ('outcome', 'cost')]
        axes[0].barh([i + offset for i in range(2)], values, height=.28, label=label, color=color)
    axes[0].set_yticks([0, 1], labels)
    axes[0].set_xlabel('Summed squared probability error\n(excess expected Brier; lower is better)')
    axes[0].set_title('Forecast distributions', loc='left', weight='bold')
    axes[0].legend(fontsize=8)
    names = [('forecast_controller', 'Supervised selector'), ('stop', 'Always stop'),
             ('complete_query', 'Complete query'), ('cheap_continuation', 'Cheap continuation'),
             ('uniform_action', 'Uniform action'), ('exact_menu_optimum', 'Exact menu optimum')]
    values = [summary['summary']['decisions']['root_state'][key]['expected_regret'] for key, _ in names]
    axes[1].barh([name for _, name in names], values,
                 color=['#285c55', '#9da9a3', '#9da9a3', '#9da9a3', '#9da9a3', '#c5b88f'], height=.6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Initial-state expected regret (research credits)\nSame fixed continuation; lower is better')
    axes[1].set_title('Implied action choices', loc='left', weight='bold')
    axes[1].margins(x=.15)
    for index, value in enumerate(values):
        axes[1].text(value, index, f' {value:.1f}', va='center', fontsize=9)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Supervised ToolSandbox reference · 48 correlated authored roots', fontsize=13, weight='bold')
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--adapter-run', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, summary = analyze(args.run, args.corpus, adapter_run=args.adapter_run, tokenizer=args.tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'recomputed-metrics.json', report)
    (args.output / 'report.md').write_text(markdown(summary))
    plot(summary, args.output / 'forecast-and-decisions.png')
    print(json.dumps({'status': summary['status'], 'primary': summary['summary']['decisions']['root_state']}, indent=2))


if __name__ == '__main__':
    main()
