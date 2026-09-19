"""Reuse audited ToolSandbox executions as paired DIAGNOSTIC questions only.

No tool replay, training, model call, dependency installation or benchmark
split reassignment. This does not make the previously inspected cohort new.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

from general_lab.toolsandbox_transfer import load_corpus
from scale_lab.common import digest, file_hash, validate_input, write_json, write_rows

DESCRIPTIONS={
    'stop':'Stop now without any further calls.',
    'lookup_contact':'Look up the contact by the exact phone number, then follow the declared continuation.',
    'query_fast':'Run the bounded content-and-time reminder query, then follow the declared continuation.',
    'query_window':'Run the complete time-window reminder query, then follow the declared continuation.'}


def prepare(corpus_folder,output):
    corpus=load_corpus(corpus_folder)
    output.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list);rows=[]
    for f in corpus['forecasts']:groups[(f['root_id'],digest(f['input']['history']))].append(f)
    questions={(q['root_id'],q['history_sha256'],q['action'],q['kind']):q for q in corpus['questions']}
    for q in corpus['questions']:
        if q['deterministic_bypass']:continue
        rows.append(dict(id=q['question_id'],group_id=q['root_id'],split='diagnostic_only',
            kind='forecast_'+q['kind'],input=q['input'],
            target_distribution=q['target'],corpus_content_sha256=corpus['content_sha256']))
    contexts=[]
    for (root,history),fs in groups.items():
        actions=fs[0]['input']['actions']
        if {f['input']['offered_action'] for f in fs}!=set(actions):raise ValueError('Incomplete executed menu')
        byaction={f['input']['offered_action']:f for f in fs}
        values={a:Fraction(byaction[a]['expected_utility']) for a in actions}
        best=max(values.values());targets=[a for a in actions if values[a]==best]
        states={questions[(root,history,a,'outcome')]['input']['state'] for a in actions}
        if len(states)!=1:raise ValueError('Action-dependent public context')
        state=states.pop();ident=digest([root,history,'paired-diagnostic'])
        provenance=dict(group_id=root,split='diagnostic_only',context_id=ident,
            corpus_content_sha256=corpus['content_sha256'],
            source_forecast_sha256=[digest(byaction[a]) for a in actions])
        item=dict(state=state,question='Which next tool action has the greatest expected terminal utility minus future tool costs, '
                  'under the declared continuation and visible prior? Prior tool costs are sunk. A stop ends immediately.',
                  options=[dict(id=a,description=DESCRIPTIONS[a]) for a in actions])
        rows.append(dict(**provenance,id=digest([ident,'decision']),kind='decision',input=item,target={'option_ids':targets}))
        gain=values['query_window']-values['stop']
        item=dict(state=state,question='Is the complete time-window reminder query followed by the declared continuation '
                  'worth its future cost compared with stopping immediately? Use expected terminal utility minus '
                  'future tool costs, excluding already-paid costs. A tie counts as no.',
                  options=[dict(id='yes',description='The query and continuation have strictly greater expected value.'),
                           dict(id='no',description='Stopping is at least as good.')])
        rows.append(dict(**provenance,id=digest([ident,'value']),kind='observation_value',input=item,
                         target={'option_id':'yes' if gain>0 else 'no'}))
        contexts.append(dict(id=ident,root_id=root,expected_action_values={a:str(v) for a,v in values.items()},
            observation_value_over_stop=str(gain),best_actions=targets))
    for row in rows:validate_input(row['input'])
    write_rows(output/'questions.jsonl',rows);write_rows(output/'contexts-private.jsonl',contexts)
    summary=dict(status='prepared_diagnostic_only',source_corpus_content_sha256=corpus['content_sha256'],
        source_files_sha256=corpus['files_sha256'],adapter_source_sha256=file_hash(Path(__file__)),
        underlying_authored_mechanisms=1,existing_roots=48,existing_database_worlds=192,
        existing_distinct_world_program_labels=1152,existing_replay_verified_production_branches=2304,
        new_executed_branches=0,new_tool_calls=0,questions=len(rows),
        forecast_questions=720,decision_questions=len(contexts),observation_value_questions=len(contexts),
        trained_questions=0,optimizer_presentations=0,model_predictions=0,
        files={p.name:file_hash(p) for p in output.glob('*.jsonl')},
        limits='Previously examined four-world reminder mechanism. Collection split names never authorize training. '
               'Values concern the specified continuation. Extra questions reuse evidence; they are not new executed tasks.')
    write_json(output/'manifest.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.corpus,a.output),indent=2))
