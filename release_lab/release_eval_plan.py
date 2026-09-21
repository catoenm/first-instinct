"""Prepare existing reserved evidence without scoring or changing ownership."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scale_lab.common import ROOT, digest, encode, file_hash, read_rows, write_json, write_rows

VERSION = 'release-evaluation-v1'


def presentation(row, tokenizer, reverse=False):
    item = dict(row['input'], options=list(reversed(row['input']['options'])) if reverse else list(row['input']['options']))
    names = [o['id'] for o in item['options']]
    old = row['option_ids']
    if set(names) != set(old) or len(names) != len(old):
        raise ValueError('Question/menu identity changed')
    ids = encode(tokenizer, item, 4096)
    if not reverse and ids != row['input_ids']:
        raise ValueError('Original source tokens changed')
    targets = {old[i] for i in row['target_indices']}
    soft = row.get('soft_target'); utility = row.get('option_utilities')
    return dict(id=digest([VERSION, row['source_id'], 'reversed' if reverse else 'original']),
                source_id=row['source_id'], input_ids=ids, option_ids=names,
                target_indices=[i for i,name in enumerate(names) if name in targets],
                soft_target=[soft[old.index(name)] for name in names] if soft is not None else None,
                option_utilities=[utility[old.index(name)] for name in names] if utility is not None else None,
                suite=row['suite'], task=row['task'], family=row['family'], metric_groups=row['metric_groups'],
                role='reserved_transfer', original_source_role=row['original_source_role'],
                source_sha256=row['source_sha256'], request_key=row.get('request_key'),
                corrected_usage_sha256=row.get('corrected_usage_sha256'))


def prepare(output):
    from transformers import AutoTokenizer
    from .toucan_audit import load_qualified, require_use
    from tool_lab.telecom_usage import require_use as telecom_use
    from tool_lab.telecom_question_audit import audit as telecom_audit
    from tool_lab.telecom_questions import load_records, utilities, menu
    tools = ROOT/'output/release-tool-data-v1'; telecom = ROOT/'output/telecom-questions-v1-usage'
    manifest = load_qualified(tools)
    usage = json.loads((telecom/'usage-private.json').read_text())
    public = json.loads((ROOT/'results/telecom-questions-v1/summary.json').read_text())
    if file_hash(telecom/'usage-private.json') != public['usage_manifest_sha256']:
        raise ValueError('Corrected telecom ownership changed')
    for name,sha in usage['files'].items():
        if file_hash(telecom/name) != sha:
            raise ValueError('Telecom data changed')
    checked = telecom_audit(ROOT/'output/telecom-questions-v1')
    if checked['status'] != 'passed':
        raise ValueError('Telecom executed labels failed reconstruction')
    records = load_records(ROOT/'output/telecom-hidden-causes-v1')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model']['id'], revision=manifest['model']['revision'],
                                             local_files_only=True, trust_remote_code=False)
    source_rows = []
    for row in read_rows(tools/'reserved_transfer-private.jsonl'):
        require_use(row, manifest, 'reserved_transfer')
        source_rows.append(dict(row, source_id=row['id'], suite='tools', soft_target=None,
                                original_source_role=row['role'], source_sha256=digest(row), metric_groups=row['servers']))
    if len(source_rows) != 2431 or len({g for r in source_rows for g in r['metric_groups']}) != 38:
        raise ValueError('Reserved tool cohort changed')
    for row in read_rows(telecom/'reserved_transfer-private.jsonl'):
        telecom_use(row, usage, 'reserved_transfer')
        item = dict(row, source_id=row['id'], suite='telecom', original_source_role=row['role'],
                    source_sha256=digest(row), metric_groups=[row['family']], target_indices=[],
                    corrected_usage_sha256=file_hash(telecom/'usage-private.json'))
        if row['task'] in ('next_procedure','observation_value'):
            item['target_indices'] = [i for i,value in enumerate(row['soft_target']) if value > 0]
            item['soft_target'] = None
            if row['task'] == 'next_procedure':
                semantic = row['semantic_sources'][0]
                utility = utilities(records, row['family'], semantic['fee'])
                item['option_utilities'] = [float(utility[menu(row['family'])[int(name)]]) for name in row['option_ids']]
                winners = [i for i,v in enumerate(item['option_utilities']) if v == max(item['option_utilities'])]
                if winners != item['target_indices']:
                    raise ValueError('Executed utility and acceptable decisions differ')
        source_rows.append(item)
    if len(source_rows) != 2465:
        raise ValueError('Reserved telecom cohort changed')
    primary = []; reversed_rows = []; rejected = []
    for row in source_rows:
        primary.append(presentation(row, tokenizer))
        try:
            reversed_rows.append(presentation(row, tokenizer, True))
        except ValueError as error:
            if not str(error).startswith('Input has '):
                raise
            rejected.append(dict(source_id=row['source_id'], reason='reversed_input_over_4096'))
    eval_tokens = {digest(r['input_ids']) for r in primary+reversed_rows}
    overlap_paths = [ROOT/'output/release-mixture-v1'/name for name in ('train-private.jsonl','development-private.jsonl')]
    overlap_paths += [ROOT/'output/trajectory-admission-v1/train-private.jsonl', ROOT/'output/history-development-v1/development.jsonl']
    scanned = 0
    for path in overlap_paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line); scanned += 1
                if digest(row['input_ids']) in eval_tokens:
                    raise ValueError('Reserved presentation overlaps training/development')
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output/'original-private.jsonl', primary); write_rows(output/'reversed-private.jsonl', reversed_rows)
    write_rows(output/'excluded-private.jsonl', rejected)
    summary = dict(status='prepared_unscored_reserved_evaluation', primary_questions=len(primary),
        primary_by_suite=dict(Counter(r['suite'] for r in primary)), reverse_presentations=len(reversed_rows),
        reverse_overlength_exclusions=len(rejected), tool_server_groups=38,
        distinct_tool_requests=len({r['request_key'] for r in primary if r['suite']=='tools'}),
        telecom_by_question_kind=dict(Counter(r['task'] for r in primary if r['suite']=='telecom')),
        telecom_connected_ownership_components=1, telecom_independent_task_warning='Two related families in one connected physical-world component.',
        input_tokens=sum(len(r['input_ids']) for r in primary+reversed_rows),
        overlap_presentations_scanned=scanned, exact_training_development_overlap=0,
        model_calls=0, optimizer_steps=0, new_world_executions=0, expected_models=3,
        maximum_scored_presentations=3*(len(primary)+len(reversed_rows)))
    write_json(output/'summary.json', summary)
    sources = ['release_lab/release_eval_plan.py','release_lab/release_eval_metrics.py','release_lab/release_eval_run.py',
               'release_lab/release_eval_select.py','scale_lab/common.py','scale_lab/model.py',
               'tests/test_release_eval.py','docs/release-evaluation-v1-protocol.md']
    freeze = dict(version=VERSION, model=manifest['model'], label_token_ids=manifest['label_token_ids'],
        pad_id=tokenizer.pad_token_id, max_tokens=4096, batch_size=4, max_seconds=5400,
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        sources={name:file_hash(ROOT/name) for name in sources},
        source_manifests=dict(tools=file_hash(tools/'manifest-private.json'), telecom_usage=file_hash(telecom/'usage-private.json')),
        scanned_input_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in overlap_paths},
        candidate_selection_pending=True, new_rental_authorized_by_this_file=False)
    write_json(output/'freeze.json', freeze)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
