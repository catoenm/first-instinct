"""Qualify public evidence and lossless sizing without admitting model questions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scale_lab.common import MODELS, ROOT, encode, file_hash, write_json, write_rows
from tool_lab.appworld_controller_diagnostic import key, menu, select
from tool_lab.appworld_public_clock import audit as clock_audit
from tool_lab.appworld_questions import history_closure as old_closure
from tool_lab.appworld_questions_v2 import build_history, history_closure
from tool_lab.appworld_qualification import read
from tool_lab.appworld_shared_input import decode_row, encode_history
from tool_lab.appworld_shortcuts import choose
from tool_lab.appworld_transfer_branch_audit import audit as branch_audit
from tool_lab.appworld_transfer_branches import variants
from tool_lab.public_parameter_constraints import documented_domains, require_api_coverage
from tool_lab.shared_json import canonical


def qualify(source, clock, schemas_extra, output):
    from transformers import AutoTokenizer
    if output.exists(): raise ValueError('Preserve earlier interface qualifications')
    execution = branch_audit(source)
    evidence = clock_audit(clock)
    if not execution['all_four_programs_qualify'] or evidence['status'] != 'passed':
        raise ValueError('Executed evidence has not qualified')
    plan = read(source/'freeze-private.json')
    schema_path = Path(plan['root'])/'api-schemas-private.json'
    schemas = {**read(schema_path), 'venmo': read(schemas_extra)}
    references = [read(source/f'{i}-capture.json') for i in range(4)]
    require_api_coverage(schemas, references)
    source_names = ['tool_lab/'+n+'.py' for n in ('appworld_evidence_interface', 'appworld_questions_v2',
        'public_argument_witness', 'public_parameter_constraints', 'shared_json', 'appworld_shared_input',
        'appworld_questions', 'record_codec', 'appworld_controller_diagnostic', 'appworld_shortcuts')]
    source_names += ['docs/appworld-evidence-interface-v2-protocol.md', 'scale_lab/common.py']
    freeze = dict(version='appworld-evidence-interface-v2', required_programs=4, maximum_tokens=8192,
        tokenizer=MODELS['qwen35-9b'], branch_freeze_sha256=file_hash(source/'freeze-private.json'),
        clock_freeze_sha256=file_hash(clock/'freeze-private.json'), clock_audit=evidence,
        schema_sha256=file_hash(schema_path), additional_schema_sha256=file_hash(schemas_extra),
        sources={name:file_hash(ROOT/name) for name in source_names})
    output.mkdir(parents=True)
    write_json(output/'freeze.json', freeze)
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],
        revision=MODELS['qwen35-9b']['revision'], token=False, local_files_only=True, trust_remote_code=False)
    summaries, all_rows = [], []
    aliases = {'redundant_read':'repeat_observation', 'recover_auth':'recover_authentication'}
    for i, (task, reference) in enumerate(zip(plan['tasks'], references, strict=True)):
        reference['api_domains'] = documented_domains(schemas)
        reference['public_clock'] = read(clock/f'{i}-0-private.json')['observations'][0]['response']
        for point in reference['points']:
            before = old_closure(reference, point)
            after = history_closure(reference, point)
            branches, reference['proposal_traces'] = {}, {}
            for variant in variants(reference['trace'], point):
                name = aliases.get(variant, variant)
                branches[name] = read(source/f'{i}-{point}-{variant}.json')
                if variant in aliases: reference['proposal_traces'][name] = branches[name]['trace']
            rows, report = build_history(task, reference, branches, point)
            packed = encode_history(rows)
            lengths, controller_checks = [], 0
            for original, shared in zip(rows, packed, strict=True):
                restored = decode_row(shared)
                if 'utility_by_option' in original:
                    if canonical(menu(original['input'])) != canonical(menu(restored['input'])):
                        raise ValueError('Decoded history, commands or costs changed')
                    base, scripts, _ = menu(original['input'])
                    # Authored values, not truth or model estimates; they only
                    # test invariance of the public controller computation.
                    probabilities = [(j+1)/(len(scripts)*(len(scripts)+1)/2) for j in range(len(scripts))]
                    success = {key(base,s): .3+.01*j for j,s in enumerate(scripts)}
                    for method in ('direct_choice','exclude_cost_dominated','forecast_expected_return'):
                        if select(original['input'], probabilities, success, method) != select(restored['input'], probabilities, success, method):
                            raise ValueError('Controller behavior changed after decoding')
                        controller_checks += 1
                    for rule in ('stop','cheapest_completion','longest_completion'):
                        if choose(original,rule) != choose(restored,rule):
                            raise ValueError('Fixed command rule changed after decoding')
                        controller_checks += 1
                old_n = len(encode(tokenizer, original['input'], 1_000_000))
                new_n = len(encode(tokenizer, shared['input'], 1_000_000))
                lengths.append(dict(question_type=original['task'], plain=old_n, shared=new_n))
            all_rows.extend(packed)
            summaries.append(dict(program_index=i, history_point=point, prior_argument_failures=len(before),
                remaining_argument_failures=len(after), builder=report, question_variants=len(rows),
                exact_roundtrips=len(rows), controller_invariance_checks=controller_checks,
                plain_fit=sum(n['plain']<=8192 for n in lengths), shared_fit=sum(n['shared']<=8192 for n in lengths),
                maximum_plain_tokens=max((n['plain'] for n in lengths),default=0),
                maximum_shared_tokens=max((n['shared'] for n in lengths),default=0), lengths=lengths))
    write_rows(output/'sizing-candidates-private.jsonl', all_rows)
    write_json(output/'histories-private.json', summaries)
    result = dict(status='passed_integrity' if all(s['remaining_argument_failures']==0 and
        s['question_variants'] and s['exact_roundtrips']==s['question_variants'] for s in summaries) else 'rejected',
        programs=4, underlying_task_instances=4, histories=len(summaries),
        raw_question_variants=len(all_rows), exact_question_roundtrips=sum(s['exact_roundtrips'] for s in summaries),
        prior_argument_failures=sum(s['prior_argument_failures'] for s in summaries),
        remaining_argument_failures=sum(s['remaining_argument_failures'] for s in summaries),
        controller_invariance_checks=sum(s['controller_invariance_checks'] for s in summaries),
        plain_fit_variants=sum(s['plain_fit'] for s in summaries), shared_fit_variants=sum(s['shared_fit'] for s in summaries),
        maximum_plain_tokens=max(s['maximum_plain_tokens'] for s in summaries),
        maximum_shared_tokens=max(s['maximum_shared_tokens'] for s in summaries),
        by_program=[dict(program_index=i, raw_variants=sum(s['question_variants'] for s in summaries if s['program_index']==i),
            shared_fit=sum(s['shared_fit'] for s in summaries if s['program_index']==i),
            maximum_shared_tokens=max(s['maximum_shared_tokens'] for s in summaries if s['program_index']==i)) for i in range(4)],
        question_types=dict(Counter(r['task'] for r in all_rows)),
        new_world_executions=0, new_model_calls=0, new_outcome_labels=0, accepted_questions=0,
        new_training_questions=0, optimizer_steps=0, freeze_sha256=file_hash(output/'freeze.json'),
        candidates_sha256=file_hash(output/'sizing-candidates-private.jsonl'),
        limitation='Local interface qualification only. All protected values stay private. Context admission is a separate frozen stage. No prediction invariance or model performance claim.')
    write_json(output/'qualification.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','clock','schemas-extra','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(qualify(a.source,a.clock,a.schemas_extra,a.output),indent=2))
