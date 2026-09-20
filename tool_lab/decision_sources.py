"""Audit existing sources and preserve uncertainty in a prospective learning index.

No task execution, network, model inference or optimizer updates occur here.
Existing execution auditors remain authoritative for source truth.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, messages, read_rows, write_json, write_rows

VERSION = 'decision-source-registry-v1'
OWNERS = dict(config='train_candidate', sqlite='train_candidate',
              application_delivery='train_candidate', filesystem_scope='train_candidate',
              reservation='train_candidate', report='development', calendar='reserved_transfer',
              publish='exposed_diagnostic')
PATHS = dict(shell='output/decision-curriculum-v3-qualified',
             application='output/application-curriculum-v1-replay-qualified',
             filesystem='output/filesystem-decisions-v1-qualified',
             calendar='output/calendar-decisions-v1-qualified',
             publication_history='output/trajectory-decisions-v1-qualified',
             reservation='results/reservation-language-v1/dynamics')


def target_vector(row):
    """Categorical outcome probabilities are never acceptable-action sets."""
    item = row['input']
    if set(item) != {'state', 'question', 'options'}:
        raise ValueError('Only public input fields may be rendered')
    messages(item)  # Validate the public contract before interpreting labels.
    ids = [o['id'] for o in item['options']]
    target = row['target']
    if set(target) == {'option_id'}:
        if target['option_id'] not in ids:
            raise ValueError('Outcome not in offered options')
        return [float(k == target['option_id']) for k in ids], 'observed_outcome'
    if set(target) == {'probabilities', 'exact_fractions'}:
        if set(target['probabilities']) != set(ids) or set(target['exact_fractions']) != set(ids):
            raise ValueError('Probability vocabulary differs from options')
        values = [target['probabilities'][k] for k in ids]
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid probability')
        fractions = [Fraction(*target['exact_fractions'][k]) for k in ids]
        if any(v < 0 for v in fractions) or sum(fractions) != 1:
            raise ValueError('Invalid exact probability mass')
        if any(abs(float(f) - p) > 1e-12 for f, p in zip(fractions, values)):
            raise ValueError('Exact and floating targets disagree')
        if abs(sum(values) - 1) > 1e-12:
            raise ValueError('Probability mass must equal one')
        return values, 'conditional_distribution'
    raise ValueError('Forecast target must be an outcome or distribution, never an acceptable-action set')


def group_forecasts(records, tokenizer=None):
    """Collapse visible duplicates without losing labels, mass, or lineage."""
    groups = defaultdict(list)
    seen = set()
    for record in records:
        key = (record['source'], record['row']['id'])
        if key in seen:
            raise ValueError('Repeated source identity is not another observation')
        seen.add(key)
        family = record['family']
        if record['role'] != OWNERS[family]:
            raise ValueError('Whole-mechanism ownership changed')
        vector, semantics = target_vector(record['row'])
        groups[digest(messages(record['row']['input']))].append((record, vector, semantics))
    output = []
    token_owners = {}
    for rendered_hash, members in sorted(groups.items()):
        first = members[0][0]
        if len({m[0]['role'] for m in members}) != 1:
            raise ValueError('Rendered input crosses source roles')
        if len({m[0]['family'] for m in members}) != 1:
            raise ValueError('Rendered collision between distinct mechanisms requires review')
        if len({m[2] for m in members}) != 1:
            raise ValueError('Cannot mix sampled outcomes and preaggregated distributions')
        if members[0][2] == 'conditional_distribution':
            if any(any(abs(a-b) > 1e-12 for a, b in zip(m[1], members[0][1])) for m in members):
                raise ValueError('Same visible input has conflicting exact distributions')
        probabilities = [sum(m[1][i] for m in members) / len(members)
                         for i in range(len(members[0][1]))]
        item = first['row']['input']
        row = dict(id=digest([VERSION, rendered_hash]), family=first['family'], role=first['role'],
                   input=item, option_ids=[o['id'] for o in item['options']], soft_target=probabilities,
                   rendered_input_sha256=rendered_hash, source_question_count=len(members),
                   target_semantics=members[0][2],
                   source_members=[dict(source=r['source'], question_id=r['row']['id'],
                                        original_split=r['original_split'], vector=v, lineage=r['lineage'])
                                   for r, v, _ in members])
        if tokenizer is not None:
            row['input_ids'] = encode(tokenizer, item, 4096)
            row['token_sha256'] = digest(row['input_ids'])
            if row['token_sha256'] in token_owners:
                raise ValueError('Distinct rendered inputs collapse to identical tokens; review before use')
            token_owners[row['token_sha256']] = (row['family'], row['role'])
        output.append(row)
    if sum(r['source_question_count'] for r in output) != len(records):
        raise AssertionError('Source member mass was lost')
    return output


def audit_sources():
    from tool_lab.decision_audit import audit as shell
    from tool_lab.application_audit import audit as application
    from tool_lab.local_data_audit import audit as local
    from tool_lab.calendar_audit import audit as calendar
    from puffer_lab.dynamics_data import verify as reservation
    auditors = dict(shell=shell, application=application, filesystem=local,
                    calendar=calendar, publication_history=local, reservation=reservation)
    reports = {key: auditors[key](ROOT/path) for key, path in PATHS.items()}
    if any(r['status'] not in ('passed', 'verified') for r in reports.values()):
        raise ValueError('Source reconstruction failed')
    for key in ('filesystem', 'calendar'):
        p = ROOT / f'output/{key}-decisions-v1-linux-parity/parity.json'
        result = json.loads(p.read_text())
        receipt_flag = ('all_public_observations_and_byte_link_receipts_exact'
                        if key == 'filesystem' else 'all_receipts_exact')
        if result['status'] != 'passed' or not result[receipt_flag]:
            raise ValueError('New runtime parity failed: '+key)
        reports[key]['linux_parity'] = dict(sha256=file_hash(p), report=result)
    return reports


def source_records():
    result, summaries = [], {}
    for source, path in PATHS.items():
        folder = ROOT/path
        reservation = source == 'reservation'
        all_rows = read_rows(folder/('rows.jsonl' if reservation else 'questions.jsonl'))
        rows = [r for r in all_rows if r.get('variant') == 'original'] if reservation else [r for r in all_rows if r['kind'] == 'forecast']
        summaries[source] = dict(available_source_questions=len(all_rows), selected_forecast_rows=len(rows),
            selected_canonical_ids=len({r.get('event_id', r['id']) for r in rows}),
            all_rendered_forecast_inputs=len({digest(messages(r['input'])) for r in all_rows
                                             if reservation or r['kind'] == 'forecast'}),
            selected_rendered_forecast_inputs=len({digest(messages(r['input'])) for r in rows}),
            source_question_sha256=file_hash(folder/('rows.jsonl' if reservation else 'questions.jsonl')))
        if reservation:
            from puffer_lab.dynamics_data import BRANCHES
            primary = [b for b in read_rows(BRANCHES) if b['replay'] == 0]
            summaries[source].update(mechanisms=1, concrete_initial_worlds=6,
                new_sqlite_executions=0, untouched_test_rows=0,
                qualification_branches_sha256=file_hash(BRANCHES),
                caveat='Existing exposed development corpus. Canonical events and presentation variants are not independent tasks.')
        for r in rows:
            family = 'reservation' if reservation else r['family']
            original_split = r['role'] if reservation else r['split']
            if reservation:
                refs = [dict(primary_branch_index=i, branch_sha256=digest(primary[i]))
                        for i in r['provenance']['primary_branch_indices']]
                lineage = dict(event_id=r['event_id'], group_id=r['group_id'],
                    event=r['event'], variant=r['variant'], branches=refs, provenance=r['provenance'])
            else:
                lineage = {k:r[k] for k in ('case_id', 'group_id', 'receipt_ids', 'receipt_sha256')}
            result.append(dict(source=source, family=family, role=OWNERS[family],
                               original_split=original_split, row=r, lineage=lineage))
    if sum(v['available_source_questions'] for v in summaries.values()) > 15000:
        raise ValueError('Source question budget exceeded')
    return result, summaries


def check_general_pools(rows):
    old = ROOT/'output/evidence-decisions-v2-data'
    freeze = json.loads((old/'freeze.json').read_text())
    our = {r['token_sha256'] for r in rows}
    result = {}
    for name in ('replay.jsonl', 'retention.jsonl', 'transfer.jsonl'):
        sha = file_hash(old/name)
        if sha != freeze['files'][name]:
            raise ValueError('Original general pool changed: '+name)
        pool = read_rows(old/name)
        overlap = our & {digest(r['input_ids']) for r in pool}
        if overlap:
            raise ValueError('Forecast inputs overlap inherited general pool: '+name)
        result[name] = dict(path=str((old/name).relative_to(ROOT)), sha256=sha,
                           questions=len(pool), exact_token_overlap=0, new_consumption=0)
    return result


def prepare(output):
    from transformers import AutoTokenizer
    output.mkdir(parents=True, exist_ok=False)
    sources = [p for parent in ('tool_lab', 'puffer_lab', 'scale_lab', 'general_lab') for p in (ROOT/parent).glob('*.py')]
    sources += [ROOT/'docs/decision-source-registry-v1-protocol.md', ROOT/'test_decision_sources.py']
    inputs = {str(p.relative_to(ROOT)):file_hash(p) for path in PATHS.values()
              for p in (ROOT/path).glob('*') if p.is_file()}
    write_json(output/'preparation-freeze.json', dict(version=VERSION, sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        inputs=inputs, ownership=OWNERS, model=MODELS['qwen35-9b'], max_source_questions=15000,
        max_distinct_forecast_inputs=6000, max_output_bytes=150_000_000, model_calls=0, optimizer_steps=0))
    try:
        audits = audit_sources()
        records, sources_summary = source_records()
        model = MODELS['qwen35-9b']
        tok = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], token=False, local_files_only=True)
        rows = group_forecasts(records, tok)
        if len(rows) > 6000:
            raise ValueError('Distinct-input preparation cap')
        pools = check_general_pools(rows)
        for role in sorted(set(OWNERS.values())):
            write_rows(output/(role+'-forecasts.jsonl'), [r for r in rows if r['role'] == role])
        write_json(output/'source-audits.json', audits)
        by_family = {}
        for family, role in OWNERS.items():
            group = [r for r in rows if r['family'] == family]
            by_family[family] = dict(role=role, unique_rendered_and_token_inputs=len(group),
                source_forecast_rows=sum(r['source_question_count'] for r in group),
                ambiguous_inputs=sum(sum(p > 0 for p in r['soft_target']) > 1 for r in group),
                maximum_tokens=max(len(r['input_ids']) for r in group))
        report = dict(status='qualified_source_index', version=VERSION, training_ready=False,
            source_summaries=sources_summary, by_family=by_family, inherited_general_pools=pools,
            total_selected_source_forecasts=len(records), total_distinct_forecast_inputs=len(rows),
            source_member_count=sum(r['source_question_count'] for r in rows),
            maximum_tokens=max(len(r['input_ids']) for r in rows), truncated=0,
            training_candidate_mechanisms=5, new_task_executions=0, model_inference=False,
            new_optimizer_steps=0, new_training_questions_consumed=0,
            note='Prepared source index, not a live learning run. Sampling multiplicities and weights are not yet a learning recipe. '
                 'Calendar remains reserved; all exposed publication questions are diagnostic only. '
                 'Only public input fields enter messages; soft_target is verifier supervision, never prompt content.')
        report['files'] = {p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'report.json', report)
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 150_000_000:
            raise ValueError('Output byte cap')
        # Serialized member and target conservation is part of qualification.
        reloaded = [r for role in sorted(set(OWNERS.values())) for r in read_rows(output/(role+'-forecasts.jsonl'))]
        if sorted(reloaded, key=lambda r:r['id']) != sorted(rows, key=lambda r:r['id']):
            raise ValueError('Serialization changed lineage or target mass')
        return report
    except BaseException as exc:
        write_json(output/'REJECTED.json', dict(error=type(exc).__name__, message=str(exc)[:1500], training_ready=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
