"""Read-only public rescoring of the fixed twelve-role outcome transfer cohort."""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path

from scale_lab.common import digest, file_hash, write_json
from . import toolsandbox_transfer as scorer
from . import toolsandbox_transfer_cohort as cohort
from . import toolsandbox_transfer_execute as executor

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'toolsandbox-transfer-cohort-report-v1'
STRATA = ('root_state', 'phone_prior_weighted')
SOURCE_FILES = ('general_lab/toolsandbox_transfer_report.py', 'tests/test_toolsandbox_transfer_report.py',
                'docs/toolsandbox-transfer-report.md')


def read(path):
    return cohort.decode(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonnegative(value):
    value = cohort.numeric(value)
    require(value >= 0, 'Negative timing/count')
    return value


def files(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        require(not path.is_symlink(), 'Artifact symlinks are not allowed')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = file_hash(path)
    return result


def journal(path, partial=False):
    path = Path(path)
    if not path.exists():
        require(partial, 'Missing completed journal: ' + path.name)
        return [], False
    raw = path.read_bytes()
    tail = bool(raw and not raw.endswith(b'\n'))
    require(partial or not tail, 'Incomplete completed journal: ' + path.name)
    lines = raw.splitlines()
    if tail:
        lines = lines[:-1]
    require(all(line.strip() for line in lines), 'Blank journal line')
    return [cohort.decode(line) for line in lines], tail


def reference(folder, corpus):
    """Use the previously published, offline raw-reference analyzer unchanged."""
    path = ROOT / cohort.REFERENCE_ANALYZER
    spec = importlib.util.spec_from_file_location('_transfer_cohort_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze(folder, corpus)


def contract(folder, sft, corpus_folder, expected):
    require(cohort.valid_hash(expected) and file_hash(folder / 'freeze.json') == expected,
            'Execution freeze differs from the explicitly supplied published checksum')
    frozen, original = read(folder / 'freeze.json'), read(sft / 'freeze.json')
    require(frozen.get('schema') == executor.SCHEMA and frozen.get('settings') == executor.SETTINGS
            and frozen.get('model_inference') is False and frozen.get('model_inference_launched') is False
            and frozen.get('content_sha256') == digest({k: v for k, v in frozen.items() if k != 'content_sha256'}),
            'Execution freeze content/protocol changed')
    require(file_hash(sft / 'freeze.json') == cohort.TRANSFER_FREEZE_SHA256, 'Published supervised freeze changed')
    expected_sources = set(executor.SOURCES) | set(cohort.SOURCE_FILES) | set(original['code_sha256'])
    require(set(frozen['source_sha256']) == expected_sources, 'Frozen source coverage differs')
    for name, checksum in frozen['source_sha256'].items():
        require(file_hash(ROOT / cohort.relative(name)) == checksum, 'Frozen source changed: ' + name)
    require(frozen['runtime'] == cohort.inference_contract(original) == frozen['plan']['required_reference_inference_contract']
            and frozen['packages'] == frozen['runtime']['packages'], 'Frozen inference contract changed')
    require(frozen['model'] == {'id': original['expected_server_metadata']['model'],
            'revision': original['expected_server_metadata']['model_revision'], 'kind': 'qwen3_5'}, 'Pinned model changed')
    baseline, reference_summary = reference(sft, corpus_folder)
    bound = frozen['supervised_reference_predictions']
    require(bound.get('reuse_verified') is True and bound.get('status') == 'complete'
            and bound['files_sha256'] == reference_summary['input_sha256']
            and bound['analyzer_sha256'] == reference_summary['analyzer_sha256']
            and bound['mapping_file_sha256'] == reference_summary['mapping_sha256'], 'Bound supervised responses changed')
    corpus = scorer.load_corpus(corpus_folder)
    questions = scorer.model_questions(corpus)
    encoded = read(folder / 'encoded-inputs.json')
    hashes = [digest(row['input']) for row in questions]
    require(len(questions) == 720 and hashes == frozen['public_input_sha256']
            and digest(encoded) == frozen['tokenization_content_sha256'] and len(encoded['rows']) == 720,
            'Public input/tokenization content changed')
    for index, row in enumerate(encoded['rows']):
        require(type(row['index']) is int and row['index'] == index and row['input_sha256'] == hashes[index]
                and isinstance(row['input_ids'], list) and 0 < len(row['input_ids']) <= 1536
                and all(type(v) is int and v >= 0 for v in row['input_ids']), 'Invalid frozen token rows')
    require(frozen['token_audit'] == {'questions': 720,
            'input_tokens': sum(len(row['input_ids']) for row in encoded['rows']),
            'max_tokens': max(len(row['input_ids']) for row in encoded['rows'])}, 'Token audit differs from retained token rows')
    wanted = [f'{arm}-s{seed}/{role}' for arm in cohort.ARMS for seed in cohort.SEEDS for role in cohort.ROLES]
    require([row['id'] for row in frozen['roles']] == wanted, 'All twelve original roles must remain in fixed order')
    groups = {}
    for row in frozen['roles']:
        require(type(row['eligible']) is bool, 'Role eligibility must be explicit')
        if row['eligible']:
            require(row['status'] == 'ready' and set(row['adapter_files_sha256']) == set(cohort.PAIR), 'Inconsistent eligible role')
            identity = digest({'model': frozen['model'], 'adapter_files_sha256': row['adapter_files_sha256']})
            require(row['identity_sha256'] == identity, 'Role adapter identity changed')
            groups.setdefault(identity, []).append(row)
    units = frozen['plan']['units']
    require([unit['identity_sha256'] for unit in units] == list(groups), 'Unit identities/order differ from eligible role aliases')
    starting = {name: original['adapter_files_sha256']['best/' + name] for name in cohort.PAIR}
    for unit in units:
        roles = groups[unit['identity_sha256']]
        same = unit['adapter_files_sha256'] == starting
        require(unit['roles'] == [row['id'] for row in roles]
                and all(row['adapter_files_sha256'] == unit['adapter_files_sha256'] for row in roles)
                and unit['adapter_path'] == roles[0]['adapter_path']
                and unit['identical_to_sft'] is same and unit['reuse_completed_sft_predictions'] is same
                and unit['planned_new_model_questions'] == (0 if same else 720), 'Unit alias/reuse plan changed')
    return frozen, corpus, questions, encoded, baseline, reference_summary


def paired(report, baseline):
    left = {row['root_id']: row for row in baseline['root_results']}
    right = {row['root_id']: row for row in report['root_results']}
    require(len(left) == len(right) == 48 and left.keys() == right.keys(), 'Paired root coverage changed')
    rows = []
    for identity, row in right.items():
        require(all(row[key] == left[identity][key] for key in ('operation', 'collection_split')), 'Paired root grouping changed')
        deltas = {stratum: {key: row['decisions'][stratum]['forecast_controller'][key] -
                                    left[identity]['decisions'][stratum]['forecast_controller'][key]
                           for key in ('expected_value', 'expected_regret')} for stratum in STRATA}
        rows.append({'root_id': identity, 'operation': row['operation'], 'collection_split': row['collection_split'], 'delta': deltas})
    return {'direction': 'checkpoint minus supervised reference; positive value or negative regret is better',
            'roots': rows, 'mean': {stratum: {key: sum(row['delta'][stratum][key] for row in rows) / 48
                       for key in ('expected_value', 'expected_regret')} for stratum in STRATA}}


def unit_report(path, unit, frozen, checksum, corpus, questions, encoded, baseline, sft_rows):
    if not path.exists():
        return {'status': 'missing', 'reason': 'No unit folder; no metrics inferred'}
    observed = {'files_sha256': files(path)}
    if not (path / 'run.json').exists() and not (path / 'complete.json').exists():
        return {**observed, 'status': 'partial', 'reason': 'Unit folder exists without an initial durable receipt; no metrics inferred'}
    try:
        receipt = read(path / 'run.json')
        complete = (path / 'complete.json').exists()
        mode = 'reused_reference' if unit['reuse_completed_sft_predictions'] else 'new_inference'
        require(receipt['identity_sha256'] == unit['identity_sha256'] and receipt['freeze_sha256'] == checksum
                and receipt['roles'] == unit['roles'] and receipt['adapter_files_sha256'] == unit['adapter_files_sha256']
                and receipt['runtime'] == frozen['runtime'] and receipt['mode'] == mode, 'Unit receipt identity/runtime changed')
        streams = {name: journal(path / (name + '.jsonl'), partial=not complete)
                   for name in ('responses', 'attempts', 'received') if mode == 'new_inference' or name == 'responses'}
        responses = streams['responses'][0]
        observed.update(recorded_status=receipt['status'], mode=mode,
            durable_rows={key: len(value[0]) for key, value in streams.items()},
            incomplete_tail={key: value[1] for key, value in streams.items()})
        require(len(responses) <= 720, 'Extra prediction rows')
        probabilities = []
        for index, row in enumerate(responses):
            require(type(row['index']) is int and row['index'] == index, 'Response order changed')
            probabilities.append(scorer._prediction(questions[index]['input'], row['probabilities']))
            nonnegative(row['milliseconds'])
        if mode == 'reused_reference':
            require(responses == sft_rows[:len(responses)], 'Reused responses differ from original supervised rows')
        else:
            attempts, received = streams['attempts'][0], streams['received'][0]
            require(len(responses) <= len(received) <= len(attempts) <= 720, 'Journal coverage/order differs')
            previous = nonnegative(receipt['started_at_unix'])
            for index, attempt in enumerate(attempts):
                require(type(attempt['index']) is int and attempt['index'] == index
                        and attempt['input_sha256'] == frozen['public_input_sha256'][index]
                        and type(attempt['input_tokens']) is int and attempt['input_tokens'] == len(encoded['rows'][index]['input_ids']),
                        'Attempt order/input hash/token count changed')
                now = nonnegative(attempt['started_at_unix'])
                require(now >= previous, 'Attempt chronology changed'); previous = now
            for index, raw in enumerate(received):
                require(type(raw['index']) is int and raw['index'] == index, 'Received response order changed')
                nonnegative(raw['milliseconds'])
                if index < len(responses):
                    answer, row = raw['response'], responses[index]
                    p = scorer._prediction(questions[index]['input'], answer['probabilities'])
                    require(p == probabilities[index] and raw['milliseconds'] == row['milliseconds']
                            and answer['choice'] == max(p, key=p.get) and type(row['input_tokens']) is int
                            and row['input_tokens'] == answer['input_tokens'] == len(encoded['rows'][index]['input_ids']),
                            'Raw response/validated response contract differs')
                    nonnegative(answer['milliseconds'])
        if not complete:
            return {**observed, 'status': 'failed' if receipt['status'] == 'failed' else 'partial',
                    'reason': 'No valid completion marker; partial prefixes are never scored'}
        executor.finished(path, unit, frozen, checksum)
        require(len(responses) == 720, 'Complete unit requires all 720 predictions')
        start, end = nonnegative(receipt['started_at_unix']), nonnegative(receipt['completed_at_unix'])
        require(frozen['created_at_unix'] <= start <= end and receipt['network_guard'] == {'blocked_probes': 1, 'blocked_attempts': 0},
                'Completion chronology/offline guard differs')
        seconds, model_seconds = nonnegative(receipt['seconds']), nonnegative(receipt['model_seconds'])
        require(model_seconds <= seconds + .001 and seconds <= frozen['settings']['max_unit_seconds'],
                'Model/total timing exceeds the recorded wall-time bound')
        if mode == 'new_inference':
            require(previous <= end, 'Attempt occurs after completion')
            loaded = receipt['loaded_model']
            require(loaded['model'] == frozen['model'] and loaded['device'] == 'mps' and loaded['parameter_dtype'] == 'float32'
                    and loaded['training'] is False and loaded['use_cache'] is False and loaded['text_attention_implementation'] == 'sdpa',
                    'Recorded loaded model differs from the fixed inference contract')
        report = scorer.score(corpus, probabilities)
        require(read(path / 'metrics.json') == report, 'Saved aggregates do not reproduce from raw responses')
        require((path / 'report.md').read_text() == scorer.markdown(report, 'Outcome-v2 identity ' + unit['identity_sha256'][:12]),
                'Saved report differs from frozen renderer')
        return {**observed, 'status': 'complete', 'model_seconds_new': model_seconds,
                'summary': report['summary'], 'by_operation': report['by_operation'],
                'by_collection_split': report['by_collection_split'], 'paired_sft': paired(report, baseline)}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {**observed, 'status': 'invalid', 'reason': str(error)}


def analyze(execution_folder, sft_reference_folder, corpus_folder, expected_freeze_sha256):
    folder, sft, corpus_folder = map(Path, (execution_folder, sft_reference_folder, corpus_folder))
    frozen, corpus, questions, encoded, baseline, reference_summary = contract(folder, sft, corpus_folder, expected_freeze_sha256)
    before = {'freeze.json': file_hash(folder / 'freeze.json'), 'encoded-inputs.json': file_hash(folder / 'encoded-inputs.json'),
              **{'units/' + k: v for k, v in files(folder / 'units').items()}}
    require(before['freeze.json'] == expected_freeze_sha256, 'Execution freeze changed during contract verification')
    expected_ids = {unit['identity_sha256'] for unit in frozen['plan']['units']}
    if (folder / 'units').exists():
        require(all(path.is_dir() and path.name in expected_ids for path in (folder / 'units').iterdir()), 'Unplanned execution unit')
    sft_rows = journal(sft / 'results/responses.jsonl')[0]
    units = {unit['identity_sha256']: unit_report(folder / 'units' / unit['identity_sha256'], unit, frozen,
             expected_freeze_sha256, corpus, questions, encoded, baseline, sft_rows) for unit in frozen['plan']['units']}
    roles = []
    for row in frozen['roles']:
        result = {key: row.get(key) for key in ('id', 'arm', 'seed', 'role', 'identity_sha256', 'eligible',
                  'reason', 'update', 'optimizer_steps', 'selected_update_zero', 'retention_eligible')}
        result['recovered_role_status'] = row['status']
        result['status'] = units[row['identity_sha256']]['status'] if row['eligible'] else 'ineligible'
        result['aliases'] = next((unit['roles'] for unit in frozen['plan']['units'] if unit['identity_sha256'] == row.get('identity_sha256')), [])
        roles.append(result)
    after = {'freeze.json': file_hash(folder / 'freeze.json'), 'encoded-inputs.json': file_hash(folder / 'encoded-inputs.json'),
             **{'units/' + k: v for k, v in files(folder / 'units').items()}}
    require(before == after, 'Execution files changed during analysis; retry on a stable snapshot')
    return {'schema': SCHEMA, 'status': 'complete' if all(row['status'] == 'complete' for row in roles) else 'partial',
            'roles': roles, 'role_status_counts': dict(Counter(row['status'] for row in roles)), 'units': units,
            'supervised_reference': {key: baseline[key] for key in ('accounting', 'summary', 'by_operation', 'by_collection_split')},
            'input_sha256': before, 'execution_freeze_sha256': expected_freeze_sha256,
            'execution_content_sha256': frozen['content_sha256'], 'corpus_sha256': corpus['files_sha256'],
            'supervised_input_sha256': reference_summary['input_sha256'],
            'analysis_source_sha256': {name: file_hash(ROOT / name) for name in SOURCE_FILES},
            'limits': 'All twelve original roles retained; identity aliases are not independent replications. No transfer-based selection, confidence interval, or broad calibration claim. Forty-eight variants share one authored four-world mechanism. Expected values use the fixed continuation, not adaptive policy execution. Phone results are prior-weighted conditional endpoints, not additional independent roots.',
            'provenance_limit': 'Published freeze/source and retained journals are verified. Runtime locator, recovery archive, adapter tensors and foundation cache are not required or re-attested by this public report.'}


def markdown(report):
    lines = ['# Outcome ToolSandbox transfer cohort', '', report['limits'], '',
             'Differences below are paired checkpoint-minus-supervised means over the same 48 roots. Initial-state expected utility/regret is primary. Negative regret change is better; no role is selected from these results.', '',
             '| Original role | Status | Identity | Initial value change | Initial regret change | Phone-weighted regret change |',
             '|---|---|---|---:|---:|---:|']
    for role in report['roles']:
        unit = report['units'].get(role['identity_sha256'], {}) if role['status'] == 'complete' else {}
        values = unit.get('paired_sft', {}).get('mean')
        cells = ([f"{values['root_state']['expected_value']:.3f}", f"{values['root_state']['expected_regret']:.3f}",
                  f"{values['phone_prior_weighted']['expected_regret']:.3f}"] if values else ['—'] * 3)
        lines.append(f"| {role['id']} | {role['status']} | {(role['identity_sha256'] or '')[:12]} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Repeated identities reuse one result; they are aliases, not extra inference or extra evidence. Missing, partial, failed and invalid units have no computed checkpoint metric.', '',
              '| Unique identity / stratum | Outcome excess Brier | Cost excess Brier | Outcome expected log loss | Cost expected log loss |',
              '|---|---:|---:|---:|---:|']
    forecasts = [('Supervised reference', report['supervised_reference'])]
    forecasts += [(identity[:12], unit) for identity, unit in report['units'].items() if unit['status'] == 'complete']
    for label, unit in forecasts:
        for stratum in STRATA:
            m = unit['summary']['forecasts'][stratum]
            lines.append(f"| {label} / {stratum} | {m['outcome']['excess_expected_brier']:.6f} | {m['cost']['excess_expected_brier']:.6f} | "
                         f"{m['outcome']['exact_expected_clipped_log_loss']:.6f} | {m['cost']['exact_expected_clipped_log_loss']:.6f} |")
    lines += ['', 'Excess expected Brier is summed squared probability error; singleton costs are excluded. Log scores use the frozen scorer’s probability floor. Operation, collection-split and paired-root records remain in JSON. No fresh-process versus HTTP latency comparison is made.', '', report['provenance_limit'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'sft-reference', 'corpus', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    require(not any(args.output.resolve().is_relative_to(path.resolve()) for path in
                    (args.execution, args.sft_reference, args.corpus)), 'Write outside the immutable input folders')
    result = analyze(args.execution, args.sft_reference, args.corpus, args.freeze_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'report.json', result)
    (args.output / 'report.md').write_text(markdown(result))


if __name__ == '__main__':
    main()
