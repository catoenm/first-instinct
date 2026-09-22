"""Post-hoc, single-reversal audit of all 240 nontrivial initial-state questions.

Preparation uses a cached tokenizer, never weights or HTTP. Execution uses only
the already-owned supervised loopback demo; the original 720 results stay intact.
"""

import argparse
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scale_lab.common import digest, file_hash, validate_input
from . import toolsandbox_transfer as scorer
from . import toolsandbox_transfer_execute as support
from . import toolsandbox_transfer_local as local
from .robustness_local import LocalPredictor

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'toolsandbox-transfer-option-order-v1'
ANALYZER = 'results/toolsandbox-transfer-supervised-v1/analyze.py'
SUMMARY = 'results/toolsandbox-transfer-supervised-v1/summary.json'
SOURCES = ('general_lab/toolsandbox_transfer_order.py', 'tests/test_toolsandbox_transfer_order.py',
           'docs/toolsandbox-transfer-order-v1-protocol.md', ANALYZER, SUMMARY,
           *support.SOURCES, *support.cohort_api.SOURCE_FILES)
SETTINGS = {'port': 8766, 'model_calls': 240, 'batch_size': 1, 'max_tokens': 1536,
            'max_options': 36, 'http_timeout_seconds': 60, 'retries': 0,
            'max_wall_seconds': 3600, 'original_order_new_calls': 0}
SCOPE = ('Post-hoc sensitivity diagnostic motivated by the completed supervised reference. '
         'One reversal of every nontrivial root-state menu; no prompt tuning, model selection, '
         'training-gain claim, new tool execution, or change to the primary 720-question result.')


def questions(corpus):
    """Derive order and payloads without inspecting targets or predictions."""
    contexts = {(c['root_id'], c['history_sha256']) for c in corpus['contexts'] if c['history_kind'] == 'root'}
    by_index = {q['question_index']: q for q in corpus['questions']}
    rows = []
    for model in scorer.model_questions(corpus):
        source = by_index[model['question_index']]
        if (source['root_id'], source['history_sha256']) not in contexts:
            continue
        original = model['input']
        if set(original) != {'state', 'question', 'options'} or any(set(o) != {'id', 'description'} for o in original['options']):
            raise ValueError('Public input has unexpected fields')
        validate_input(original)
        reverse = {**deepcopy(original), 'options': list(reversed(deepcopy(original['options'])))}
        rows.append({'index': len(rows), 'original_index': model['index'],
                     'question_index': model['question_index'], 'question_id': model['question_id'],
                     'root_id': source['root_id'], 'kind': source['kind'],
                     'original_input_sha256': digest(original), 'input_sha256': digest(reverse),
                     'original_option_ids': [o['id'] for o in original['options']], 'input': reverse})
    if (len(contexts), len(rows), sum(r['kind'] == 'outcome' for r in rows), sum(r['kind'] == 'cost' for r in rows)) != (48, 240, 144, 96):
        raise ValueError('Require all 48 roots and exactly 144 outcome/96 cost questions')
    if any(sum(r['root_id'] == root and r['kind'] == kind for r in rows) != count
           for root, _ in contexts for kind, count in (('outcome', 3), ('cost', 2))):
        raise ValueError('Each root must contribute all five nontrivial questions')
    return rows


def reference(paths, *, analysis=False, expected_binding=None):
    """Recompute the published reference; require its exact original raw files."""
    path = ROOT / ANALYZER
    spec = importlib.util.spec_from_file_location('_order_reference_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter, cache = Path(paths['adapter_run']), Path(paths['tokenizer_cache'])
    _, summary = module.analyze(paths['reference_folder'], paths['corpus_folder'],
                               adapter_run=None if analysis and not adapter.exists() else adapter,
                               tokenizer=None if analysis and not cache.exists() else cache)
    published = support.read(ROOT / SUMMARY)
    for key in ('input_sha256', 'mapping_sha256', 'freeze_sha256', 'freeze_content_sha256', 'analyzer_sha256'):
        if summary[key] != published[key]:
            raise ValueError('Reference differs from the published completed 720 responses')
    if summary['status'] != 'verified_complete':
        raise ValueError('A completed original reference is required')
    frozen = support.read(Path(paths['reference_folder']) / 'freeze.json')
    if (not analysis or adapter.exists()) and local.adapter_hashes(adapter) != frozen['adapter_files_sha256']:
        raise ValueError('Supervised adapter files changed')
    if not analysis and local.packages() != frozen['packages']:
        raise ValueError('Runtime differs from the supervised reference')
    token_files = ({p.name: file_hash(p) for p in cache.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja')}
                   if cache.exists() else deepcopy(expected_binding['tokenizer_files_sha256']) if analysis else {})
    # Additional config files can affect AutoTokenizer; bind them as well as the
    # original audit's explicitly named tokenizer files, without loading weights.
    for name, value in frozen['prompt_provenance']['tokenizer_files'].items():
        if (not analysis or cache.exists()) and file_hash(cache / name) != value:
            raise ValueError('Cached tokenizer differs from the original reference')
    if frozen['expected_server_metadata']['device'] != 'mps':
        raise ValueError('Require the original MPS supervised runtime')
    corpus = scorer.load_corpus(paths['corpus_folder'])
    responses = support.journal(Path(paths['reference_folder']) / 'results/responses.jsonl')
    binding = {'reference_files_sha256': summary['input_sha256'], 'mapping_sha256': summary['mapping_sha256'],
               'reference_freeze_sha256': summary['freeze_sha256'], 'analyzer_sha256': summary['analyzer_sha256'],
               'published_summary_sha256': file_hash(ROOT / SUMMARY), 'corpus_sha256': frozen['corpus_sha256'],
               'adapter_files_sha256': frozen['adapter_files_sha256'], 'packages': frozen['packages'],
               'expected_server_metadata': frozen['expected_server_metadata'], 'tokenizer_files_sha256': token_files,
               'code_sha256': {name: file_hash(ROOT / name) for name in set(SOURCES) | set(frozen['code_sha256'])}}
    return binding, corpus, [row['probabilities'] for row in responses]


def prepare(folder, reference_folder, corpus_folder, adapter_run, tokenizer_cache, expected_server_pid):
    if type(expected_server_pid) is not int or expected_server_pid <= 1:
        raise ValueError('Declare the already-owned demo PID')
    paths = {name: str(Path(value).resolve()) for name, value in locals().copy().items()
             if name in ('reference_folder', 'corpus_folder', 'adapter_run', 'tokenizer_cache')}
    folder = Path(folder).resolve()
    if any(folder.is_relative_to(Path(value)) for value in paths.values()):
        raise ValueError('Audit output must be separate from immutable inputs')
    support.enable_offline()
    with support.no_network():
        binding, corpus, _ = reference(paths)
        rows = questions(corpus)
        encoded = support.tokenization(support.tokenizer(paths['tokenizer_cache']), rows)
        # Close preparation races without a model forward or network request.
        if reference(paths)[0] != binding:
            raise ValueError('Provenance changed during preparation')
    frozen = {'schema': SCHEMA, 'selection_role': 'none', 'post_hoc': True, 'model_inference_launched': False,
              'created_at_unix': time.time(), 'scope': SCOPE, 'paths': paths, 'binding': binding,
              'expected_server_pid': expected_server_pid, 'settings': SETTINGS,
              'questions_sha256': digest(rows), 'encoded_inputs_sha256': digest(encoded),
              'token_audit': {'count': 240, 'min_tokens': min(len(r['input_ids']) for r in encoded['rows']),
                              'max_tokens': max(len(r['input_ids']) for r in encoded['rows']),
                              'max_options': max(len(r['input']['options']) for r in rows)},
              'provenance_limit': 'Disk/source hashes and reported service identity; not an attestation of resident tensors.'}
    frozen['content_sha256'] = digest(frozen)
    folder.mkdir(parents=True, exist_ok=False)
    support.atomic_json(folder / 'questions.json', rows)
    support.atomic_json(folder / 'encoded-inputs.json', encoded)
    support.atomic_json(folder / 'freeze.json', frozen)
    return frozen


def verify(folder, *, tokenize=False, analysis=False, reference_folder=None, corpus_folder=None):
    folder = Path(folder)
    frozen = support.read(folder / 'freeze.json')
    if (frozen.get('schema') != SCHEMA or frozen.get('post_hoc') is not True or frozen.get('selection_role') != 'none'
            or frozen.get('model_inference_launched') is not False or frozen.get('scope') != SCOPE
            or frozen.get('settings') != SETTINGS
            or frozen.get('content_sha256') != digest({k: v for k, v in frozen.items() if k != 'content_sha256'})):
        raise ValueError('Invalid option-order freeze')
    for name, value in frozen['binding']['code_sha256'].items():
        if file_hash(ROOT / name) != value:
            raise ValueError('Frozen audit source/protocol changed')
    if (reference_folder is not None or corpus_folder is not None or analysis) and tokenize:
        raise ValueError('Read-only remapping may not alter inference verification')
    paths = dict(frozen['paths'])
    for key, value in (('reference_folder', reference_folder), ('corpus_folder', corpus_folder)):
        if value is not None:
            if not analysis:
                raise ValueError('Only read-only analysis permits artifact remapping')
            paths[key] = str(Path(value).resolve())
    binding, corpus, original = (reference(paths, analysis=True, expected_binding=frozen['binding'])
                                if analysis else reference(paths))
    if binding != frozen['binding']:
        raise ValueError('Frozen reference/data/adapter/runtime/tokenizer changed')
    rows, encoded = questions(corpus), support.read(folder / 'encoded-inputs.json')
    if (rows != support.read(folder / 'questions.json') or digest(rows) != frozen['questions_sha256']
            or digest(encoded) != frozen['encoded_inputs_sha256']):
        raise ValueError('Reversal mapping or tokenization changed')
    if tokenize and support.tokenization(support.tokenizer(frozen['paths']['tokenizer_cache']), rows) != encoded:
        raise ValueError('Reversed tokenization no longer reproduces')
    return frozen, corpus, rows, original


def listener(expected_pid):
    pids = subprocess.run(['/usr/sbin/lsof', '-nP', '-a', '-iTCP:8766', '-sTCP:LISTEN', '-t'],
                          capture_output=True, text=True, check=True, timeout=5).stdout.split()
    if set(pids) != {str(expected_pid)}:
        raise ValueError('Loopback listener is not the declared owned demo PID')
    started = subprocess.run(['/bin/ps', '-p', str(expected_pid), '-o', 'lstart='],
                             capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    if not started:
        raise ValueError('Missing demo process start identity')
    return {'pid': expected_pid, 'started': started}


class DurablePredictor(LocalPredictor):
    """Preserve the frozen HTTP serialization; journal receipt before validation."""
    def __init__(self, expected, output, received):
        self.received = received
        super().__init__(8766, expected, output)

    def call(self, path, payload=None):
        response = super().call(path, payload)
        if path == '/api/answer':
            self.received(response)
        return response


def compare(corpus, rows, original, reversed_predictions):
    if len(reversed_predictions) != 240 or len(original) != 720:
        raise ValueError('Comparison requires exactly 240 reversed and 720 original predictions')
    combined, drifts = deepcopy(original), []
    for row, value in zip(rows, reversed_predictions):
        reverse = scorer._prediction(row['input'], value)
        old_input = {**row['input'], 'options': list(reversed(row['input']['options']))}
        old = scorer._prediction(old_input, original[row['original_index']])
        a, b = max(old, key=old.get), max(reverse, key=reverse.get)
        old_modes, new_modes = sorted(k for k in old if old[k] == old[a]), sorted(k for k in reverse if reverse[k] == reverse[b])
        delta = {k: reverse[k] - old[k] for k in old}
        drifts.append({k: row[k] for k in ('index', 'original_index', 'question_id', 'root_id', 'kind')}
            | {'original': old, 'reversed': reverse, 'delta': delta,
               'total_variation': sum(map(abs, delta.values())) / 2,
               'max_absolute_drift': max(map(abs, delta.values())),
               'summed_squared_drift': sum(x*x for x in delta.values()),
               'original_modal_id': a, 'reversed_modal_id': b, 'modal_flip': a != b,
               'original_modal_set': old_modes, 'reversed_modal_set': new_modes,
               'modal_set_changed': old_modes != new_modes})
        combined[row['original_index']] = reverse
    before, after = scorer.score(corpus, original), scorer.score(corpus, combined)
    # The unchanged scorer canonicalizes menus back to corpus order. Its tied
    # modal accuracy would therefore not describe reversed presentation order.
    # Keep probability scores; report modal behavior only in the explicit ID
    # and set comparisons above, where each presented ordering is preserved.
    for report in (before, after):
        for marginal in report['summary']['forecasts']['root_state'].values():
            marginal.pop('modal_accuracy', None)
    decisions = []
    for a, b in zip(before['decision_contexts'], after['decision_contexts']):
        if a['history_kind'] == 'root':
            decisions.append({'root_id': a['root_id'], 'original_action': a['chosen_action'],
                              'reversed_action': b['chosen_action'], 'action_flip': a['chosen_action'] != b['chosen_action'],
                              'original': a['policies']['forecast_controller'], 'reversed': b['policies']['forecast_controller']})
    summaries = {}
    for kind in ('outcome', 'cost'):
        group = [r for r in drifts if r['kind'] == kind]
        summaries[kind] = {'questions': len(group), 'modal_flips': sum(r['modal_flip'] for r in group),
            'modal_set_changes': sum(r['modal_set_changed'] for r in group),
            'original_ties': sum(len(r['original_modal_set']) > 1 for r in group),
            'reversed_ties': sum(len(r['reversed_modal_set']) > 1 for r in group),
            'mean_total_variation': sum(r['total_variation'] for r in group) / len(group),
            'mean_summed_squared_drift': sum(r['summed_squared_drift'] for r in group) / len(group),
            'max_absolute_drift': max(r['max_absolute_drift'] for r in group)}
    return {'schema': SCHEMA, 'scope': SCOPE, 'counts': {'roots': 48, 'new_reversed_predictions': 240,
             'original_predictions_reused': 240, 'deterministic_root_stop_costs': 48},
            'by_kind': summaries, 'question_drifts': drifts, 'root_decisions': decisions,
            'root_state_original': before['summary']['decisions']['root_state']['forecast_controller'],
            'root_state_reversed': after['summary']['decisions']['root_state']['forecast_controller'],
            'root_forecasts_original': before['summary']['forecasts']['root_state'],
            'root_forecasts_reversed': after['summary']['forecasts']['root_state'],
            'limits': 'One nonrandomized reversal, with original/reversed calls at different times. No contemporaneous original repeats: temporal/numerical variation is not separately estimated. Authored exact distributions are not empirical general calibration. Equal-root implied fixed-continuation values, not newly executed policy returns. No ordering is selected or promoted.'}


def markdown(report):
    lines = ['# Post-hoc option-order sensitivity', '', SCOPE, '',
             '| Marginal | Questions | Mean total variation | Modal flips | Modal-set changes |',
             '|---|---:|---:|---:|---:|']
    for kind, row in report['by_kind'].items():
        lines.append(f"| {kind} | {row['questions']} | {row['mean_total_variation']:.6f} | {row['modal_flips']} | {row['modal_set_changes']} |")
    lines += ['', f"Implied initial action changes: {sum(r['action_flip'] for r in report['root_decisions'])}/48.",
              f"Original/reversed mean expected value: {report['root_state_original']['expected_value']:.6f} / {report['root_state_reversed']['expected_value']:.6f}.",
              '', 'Exact modal ties use first presented option; modal sets and tie counts are retained separately. '
              'The 48 singleton stop costs remain exact deterministic bypasses. Phone-state predictions are neither queried nor compared.',
              '', report['limits'], '']
    return '\n'.join(lines)


def run_local(folder):
    folder = Path(folder)
    started, deadline = time.monotonic(), time.monotonic() + 3600
    output = folder / 'results'
    output.mkdir(exist_ok=False)  # No automatic replay/resume of partial runs.
    receipt = {'schema': SCHEMA, 'status': 'starting', 'scope': SCOPE, 'started_at_unix': time.time(),
               'freeze_sha256': file_hash(folder / 'freeze.json'), 'attempted': 0, 'received': 0, 'validated': 0}
    support.atomic_json(output / 'run.json', receipt)
    def received(response):
        index = receipt['received']
        receipt['received'] += 1
        try:
            support.append(output / 'received.jsonl', {'index': index, 'response': response})
        except (TypeError, ValueError):
            support.append(output / 'received.jsonl', {'index': index, 'invalid_response': type(response).__name__})
            raise
        finally:
            support.atomic_json(output / 'run.json', receipt)
    try:
        with support.stop_signals(3600):
            support.enable_offline()
            with support.no_network():
                frozen, corpus, rows, original = verify(folder, tokenize=True)
            if file_hash(folder / 'freeze.json') != receipt['freeze_sha256']:
                raise ValueError('Freeze file changed during preflight')
            support.deadline_check(deadline)
            process = listener(frozen['expected_server_pid'])
            predictor = DurablePredictor(frozen['binding']['expected_server_metadata'], output, received)
            if predictor.limits['max_tokens'] != 1536 or predictor.limits['max_options'] != 36:
                raise ValueError('Resident service limits changed')
            receipt.update(status='running', process=process, freeze_content_sha256=frozen['content_sha256'])
            support.atomic_json(output / 'run.json', receipt)
            values = []
            for row in rows:
                support.deadline_check(deadline)
                if receipt['attempted'] >= 240 or row['index'] != receipt['attempted']:
                    raise ValueError('Refusing extra or reordered calls')
                support.append(output / 'attempts.jsonl', {'index': row['index'], 'input_sha256': row['input_sha256'], 'started_at_unix': time.time()})
                receipt['attempted'] += 1
                support.atomic_json(output / 'run.json', receipt)
                result = predictor([deepcopy(row['input'])])
                # LocalPredictor preserves the received distribution; fsync its
                # unchanged journal before any subsequent validation/request.
                with (output / 'responses.jsonl').open('r') as stream:
                    os.fsync(stream.fileno())
                if len(result) != 1 or predictor.calls != row['index'] + 1 or receipt['received'] != predictor.calls:
                    raise ValueError('Single-question response coverage changed')
                value = scorer._prediction(row['input'], result[0])
                raw = support.journal(output / 'responses.jsonl')[-1]
                if type(raw['milliseconds']) not in (int, float) or not math.isfinite(raw['milliseconds']) or raw['milliseconds'] < 0:
                    raise ValueError('Invalid response timing')
                support.append(output / 'validated.jsonl', {'index': row['index'], 'probabilities': value})
                values.append(value)
                receipt['validated'] += 1
                support.atomic_json(output / 'run.json', receipt)
                support.deadline_check(deadline)
            if any(receipt[k] != 240 for k in ('attempted', 'received', 'validated')):
                raise ValueError('Incomplete reversal coverage')
            predictor.verify_end()
            if listener(frozen['expected_server_pid']) != process:
                raise ValueError('Owned demo process changed')
            with support.no_network():
                verify(folder, tokenize=True)
            if file_hash(folder / 'freeze.json') != receipt['freeze_sha256']:
                raise ValueError('Freeze file changed during inference')
            report = compare(corpus, rows, original, values)
            support.deadline_check(deadline)
            support.atomic_json(output / 'metrics.json', report)
            with (output / 'report.md').open('w') as stream:
                stream.write(markdown(report)); stream.flush(); os.fsync(stream.fileno())
            receipt.update(status='complete', completed_at_unix=time.time(), model_seconds=predictor.model_milliseconds / 1000)
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__, stopped_at_unix=time.time())
        raise
    finally:
        receipt['seconds'] = time.monotonic() - started
        support.atomic_json(output / 'run.json', receipt)
    return receipt


def analyze(folder, reference_folder=None, corpus_folder=None):
    folder = Path(folder)
    frozen, corpus, rows, original = verify(folder, analysis=True, reference_folder=reference_folder, corpus_folder=corpus_folder)
    output = folder / 'results'
    run = support.read(output / 'run.json')
    if (run.get('schema') != SCHEMA or run.get('status') != 'complete' or run.get('scope') != SCOPE
            or run.get('freeze_sha256') != file_hash(folder / 'freeze.json')
            or run.get('freeze_content_sha256') != frozen['content_sha256']
            or any(type(run.get(k)) is not int or run[k] != 240 for k in ('attempted', 'received', 'validated'))):
        raise ValueError('Require the complete original audit run')
    process = run.get('process', {})
    if (not isinstance(process, dict) or type(process.get('pid')) is not int or process['pid'] != frozen['expected_server_pid']
            or not isinstance(process.get('started'), str) or not process['started'].strip()):
        raise ValueError('Recorded demo process identity differs from the freeze')
    journals = {name: support.journal(output / (name + '.jsonl')) for name in ('attempts', 'received', 'responses', 'validated')}
    if any(len(value) != 240 or [r['index'] for r in value] != list(range(240)) for value in journals.values()):
        raise ValueError('Completed journals require all 240 ordered rows')
    from .robustness_local import metadata
    previous = run['started_at_unix']
    for i, row in enumerate(rows):
        attempt, response, raw, valid = (journals[name][i] for name in ('attempts', 'responses', 'received', 'validated'))
        if attempt['input_sha256'] != row['input_sha256'] or not previous <= attempt['started_at_unix'] <= run['completed_at_unix']:
            raise ValueError('Attempt provenance differs')
        previous = attempt['started_at_unix']
        if (metadata(raw['response']) != frozen['binding']['expected_server_metadata']
                or raw['response']['answers']['audit']['probabilities'] != response['probabilities']
                or raw['response']['answers']['audit']['milliseconds'] != response['milliseconds']
                or scorer._prediction(row['input'], response['probabilities']) != valid['probabilities']
                or type(response['milliseconds']) not in (int, float) or not math.isfinite(response['milliseconds']) or response['milliseconds'] < 0):
            raise ValueError('Response/validation journals differ')
    elapsed = sum(r['milliseconds'] for r in journals['responses']) / 1000
    if not math.isclose(elapsed, run['model_seconds'], abs_tol=1e-6) or not 0 <= elapsed <= run['seconds'] <= 3600:
        raise ValueError('Recorded serial timing does not reproduce')
    report = compare(corpus, rows, original, [r['probabilities'] for r in journals['validated']])
    if report != support.read(output / 'metrics.json') or markdown(report) != (output / 'report.md').read_text():
        raise ValueError('Saved analysis does not reproduce')
    installed = {}
    for package, historical in frozen['binding']['packages'].items():
        try:
            current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current = None
        installed[package] = {'historical_inference': historical, 'current_analysis': current}
    return {**report, 'analysis_provenance': {
        'adapter': 'verified' if Path(frozen['paths']['adapter_run']).exists() else 'unavailable',
        'tokenizer': 'verified' if Path(frozen['paths']['tokenizer_cache']).exists() else 'unavailable',
        'packages': installed,
        'note': 'Read-only rescoring loads no model or tokenizer and requires no inference packages. Missing runtime artifacts are historical hash provenance only; available files were checked and changed files refused.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('folder', 'reference-folder', 'corpus-folder', 'adapter-run', 'tokenizer-cache'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-server-pid', type=int, required=True)
    sub.add_parser('run').add_argument('--folder', type=Path, required=True)
    p = sub.add_parser('analyze')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--reference-folder', type=Path)
    p.add_argument('--corpus-folder', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = prepare(**args) if command == 'prepare' else run_local(**args) if command == 'run' else analyze(**args)
    summary = {'command': command, 'status': result.get('status', 'verified'), 'model_inference_launched': command == 'run' and result['attempted'] > 0}
    if command == 'analyze':
        summary['analysis_provenance'] = result['analysis_provenance']
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
