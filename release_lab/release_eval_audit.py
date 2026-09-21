"""Reconstruct saved release presentations directly from admitted source rows."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, digest, encode, file_hash, read_rows, write_json
from .release_eval_run import verify_data


def audit(data):
    from transformers import AutoTokenizer
    from tool_lab.telecom_questions import load_records, menu
    from tool_lab.telecom_usage import require_use
    freeze = verify_data(data)
    tools = ROOT/'output/release-tool-data-v1'; telecom = ROOT/'output/telecom-questions-v1-usage'
    paths = dict(tools=tools/'manifest-private.json', telecom_usage=telecom/'usage-private.json')
    for name,path in paths.items():
        if file_hash(path) != freeze['source_manifests'][name]:
            raise ValueError('Source manifest changed')
    manifest = json.loads(paths['tools'].read_text()); usage = json.loads(paths['telecom_usage'].read_text())
    for directory,expected in ((tools,manifest['outputs']), (telecom,usage['files'])):
        path=directory/'reserved_transfer-private.jsonl'
        if file_hash(path) != expected[path.name]:
            raise ValueError('Reserved source changed')
    originals = read_rows(tools/'reserved_transfer-private.jsonl')+read_rows(telecom/'reserved_transfer-private.jsonl')
    source = {r['id']:r for r in originals}
    if len(source) != 2465:
        raise ValueError('Reserved source coverage changed')
    records = load_records(ROOT/'output/telecom-hidden-causes-v1')
    tokenizer = AutoTokenizer.from_pretrained(freeze['model']['id'], revision=freeze['model']['revision'],
                                             local_files_only=True, trust_remote_code=False)
    seen = {}; checked = Counter()
    for kind in ('original','reversed'):
        seen[kind] = set()
        for row in read_rows(data/(kind+'-private.jsonl')):
            raw = source[row['source_id']]
            if raw['id'] in seen[kind] or row['source_sha256'] != digest(raw) or row['role'] != 'reserved_transfer':
                raise ValueError('Changed source identity, role or repeated presentation')
            seen[kind].add(raw['id'])
            options = list(raw['input']['options'])
            if kind == 'reversed':
                options.reverse()
            names = [o['id'] for o in options]
            expected = encode(tokenizer, dict(raw['input'],options=options),4096)
            if row['input_ids'] != expected or row['option_ids'] != names or row['original_source_role'] != raw['role']:
                raise ValueError('Saved public presentation differs from the source')
            if row['suite'] == 'tools':
                if raw['role'] != 'reserved_transfer' or manifest['rows'][raw['id']] != dict(role='reserved_transfer',sha256=digest(raw)):
                    raise ValueError('Tool ownership or original row changed')
                targets = {raw['option_ids'][i] for i in raw['target_indices']}
                soft = None; utility = None
                if row['metric_groups'] != raw['servers']:
                    raise ValueError('Tool metric grouping changed')
            elif row['suite'] == 'telecom':
                require_use(raw,usage,'reserved_transfer')
                if row['corrected_usage_sha256'] != file_hash(paths['telecom_usage']) or row['metric_groups'] != [raw['family']]:
                    raise ValueError('Corrected telecom role or grouping changed')
                q = dict(zip(raw['option_ids'],raw['soft_target']))
                forecast = raw['task'] in ('immediate_success','continued_success')
                soft = [q[name] for name in names] if forecast else None
                targets = set() if forecast else {name for name,value in q.items() if value > 0}
                utility = None
                if raw['task'] == 'next_procedure':
                    fee = float(raw['semantic_sources'][0]['fee']); utility = []
                    for name in names:
                        alternative = menu(raw['family'])[int(name)]
                        worlds = [r for (case,_,action,replica),r in records.items()
                                  if case==raw['family'] and action==alternative and replica==0]
                        if len(worlds) != 4:
                            raise ValueError('Changed compatible-world prior')
                        utility.append(math.fsum(20*int(r['continued']['outcome']['success'])-fee*r['actor_reads']
                            -.25*r['actor_writes']-float(r['continued']['billed_delta']) for r in worlds)/4)
            else:
                raise ValueError('Unknown evaluation contract')
            if row['target_indices'] != [i for i,name in enumerate(names) if name in targets] or row['soft_target'] != soft:
                raise ValueError('Source labels changed during evaluation packing')
            if utility is None:
                if row['option_utilities'] is not None:
                    raise ValueError('Invented action utility')
            elif len(utility) != len(row['option_utilities']) or any(abs(a-b)>1e-9 for a,b in zip(utility,row['option_utilities'])):
                raise ValueError('Utility differs from recorded outcomes and charged actions')
            checked[kind+'/'+row['suite']] += 1
    excluded = read_rows(data/'excluded-private.jsonl')
    excluded_ids = {r['source_id'] for r in excluded}
    if len(excluded) != len(excluded_ids) or seen['original'] != set(source) or seen['reversed']|excluded_ids != set(source) or seen['reversed']&excluded_ids:
        raise ValueError('Incomplete or repeated source coverage')
    for ident in excluded_ids:
        raw=source[ident]
        try:
            encode(tokenizer,dict(raw['input'],options=list(reversed(raw['input']['options']))),4096)
        except ValueError as error:
            if not str(error).startswith('Input has '):
                raise
        else:
            raise ValueError('Fitting reversal was incorrectly excluded')
    return dict(status='independently_verified_unscored_evaluation',checked_presentations=dict(checked),
        data_freeze_sha256=file_hash(data/'freeze.json'), source_questions=len(source),
        source_labels_preserved=True, utility_recomputed_from_primary_worlds=True,
        auditor_sha256=file_hash(Path(__file__)), model_calls=0, new_world_executions=0, optimizer_steps=0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=audit(a.data);write_json(a.output,result);print(json.dumps(result,indent=2))
