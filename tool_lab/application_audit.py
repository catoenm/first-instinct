"""Independent saved-artifact and input-boundary audit; no tool/model execution."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, file_hash, read_rows, write_json
from tool_lab import application_curriculum as app


def same_aggregate(a,b):
    """Allow only rounding noise in derived aggregate floats, never labels."""
    if isinstance(a,float) and isinstance(b,float):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_aggregate(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same_aggregate(x,y) for x,y in zip(a,b))
    return a==b


def audit_initial(case,trace):
    before=trace['before'];contacts={r['phone_number']:r for r in before['contact']}
    if set(contacts)!=set(case['phones'].values()) or len(before['contact'])!=3:
        raise ValueError('Wrong initial contact database')
    if [r['phone_number'] for r in before['contact'] if r['is_self']]!=[case['phones']['self']]:
        raise ValueError('Wrong self contact')
    messages=before['messaging']
    if len(messages)!=int(case['ledger']!='none'):
        raise ValueError('Wrong initial ledger cardinality')
    if messages:
        message=messages[0]
        recipient=contacts[case['phones'][case['ledger']]]
        expected=dict(content=case['content'],recipient_phone_number=recipient['phone_number'],
            recipient_person_id=recipient['person_id'],sender_phone_number=case['phones']['self'],
            sender_person_id=contacts[case['phones']['self']]['person_id'],creation_timestamp=app.NOW)
        if {k:v for k,v in message.items() if k!='message_id'}!=expected:
            raise ValueError('Wrong initial message fields')
    settings=before['setting']
    if len(settings)!=1:raise ValueError('Wrong setting cardinality')
    low=case['connection']=='low_battery'
    for key,expected in dict(low_battery_mode=low,cellular=case['connection']=='ready',wifi=not low,location_service=not low).items():
        if settings[0][key]!=expected:raise ValueError('Wrong initial setting')
    if any(before[k] for k in before if k not in ('contact','messaging','setting')):
        raise ValueError('Unexpected initial database rows')
    prefix=trace['prefix']
    expected_actions=(list(app.READS) if case['regime']=='fresh' else ['send_'+case['goal']]
                      if case['regime']=='failed_send' else ['messages'] if case['regime']=='stale' else [])
    if [p['action'] for p in prefix]!=expected_actions:
        raise ValueError('Changed prefix actions')
    for p in prefix:
        obs=p['observation'];action=p['action']
        if p['historical']!=(case['regime']=='stale') or obs['call']!=app.menu(case)[action]:
            raise ValueError('Changed prefix time or command')
        if case['regime']=='failed_send':
            if obs.get('error',{}).get('type')!='ConnectionError' or settings[0]['cellular']:
                raise ValueError('Failed-send observation inconsistent')
        else:
            expected=([] if case['regime']=='stale' else sorted(messages,key=app.canonical)
                      if action=='messages' else settings[0]['cellular' if action=='cellular_status' else 'low_battery_mode'])
            if 'error' in obs or obs.get('result')!=expected:
                raise ValueError('Initial observation inconsistent with database')
    # Rebuild the prompt from fixture ownership, the audited prefix and public
    # constructor fields. Do not trust the trace's own request/cost declaration.
    public=object.__new__(app.Episode)
    public.case=case;public.commands=app.menu(case);public.history=prefix;public.depth=0
    if public.input()!=trace['input']:
        raise ValueError('Public input disagrees with fixture goal, costs or history')


def audit(folder,tokenizer=None,max_tokens=1536):
    if (folder/'REJECTED.json').exists():raise ValueError('Rejected collection cannot be qualified by read-back')
    frozen=json.loads((folder/'pre-execution-freeze.json').read_text())
    qualification=json.loads((folder/'qualification.json').read_text())
    for name,expected in qualification['files'].items():
        if file_hash(folder/name)!=expected:raise ValueError('Changed saved artifact: '+name)
    for name,expected in frozen['sources'].items():
        path=Path(name)
        relative=Path(*path.parts[-2:]) if path.is_absolute() else path
        if relative.as_posix() not in ('tool_lab/application_curriculum.py','general_lab/toolsandbox_pilot.py',
                                      'tests/test_application_curriculum.py','docs/application-curriculum-v1-protocol.md'):
            raise ValueError('Unexpected source in freeze')
        if file_hash(ROOT/relative)!=expected:raise ValueError('Changed executed source: '+name)
    if file_hash(folder/'cases.jsonl')!=frozen['cases_sha256']:raise ValueError('Fixture freeze differs')
    cases=read_rows(folder/'cases.jsonl');traces=read_rows(folder/'executions.jsonl')
    if cases!=app.fixtures():raise ValueError('Fixture registry differs')
    bycase={c['id']:c for c in cases};expected={(c['id'],a,p) for c in cases for a in app.menu(c) for p in app.PLANS}
    if len(traces)!=len(expected) or {(t['case_id'],t['action'],t['plan']) for t in traces}!=expected:
        raise ValueError('Incomplete executed alternatives')
    for trace in traces:
        audit_initial(bycase[trace['case_id']],trace)
        app.reconstruct(bycase[trace['case_id']],trace)
    rows,contexts=app.questions(cases,traces)
    if rows!=read_rows(folder/'questions.jsonl') or not same_aggregate(contexts,read_rows(folder/'contexts-private.jsonl')):
        raise ValueError('Labels or counterfactual values differ after read-back')
    n_setup=sum(len(t['setup']) for t in traces)*2
    n_prefix=sum(len(t['prefix']) for t in traces)*2
    n_branch=sum(sum(e['observation'] is not None for e in t['events']) for t in traces)*2
    if any(qualification[k]!=v for k,v in dict(branches=len(traces)*2,setup_calls=n_setup,prefix_calls=n_prefix,branch_calls=n_branch).items()):
        raise ValueError('Execution accounting differs')
    # Historical setup must not create extra current-world identifier variants.
    concrete=len({app.digest(t['before']) for t in traces})
    semantic=len({(c['ledger'],c['connection']) for c in cases})
    if concrete!=semantic:raise ValueError('Historical setup perturbed base-world identity')
    guard=qualification['guard']
    if len(guard['blocked_attempts'])!=guard['post_import_attempt_count']:
        raise ValueError('Unexpected execution-phase capability attempt')
    already_done=[t for t in traces if bycase[t['case_id']]['ledger']==bycase[t['case_id']]['goal'] and t['action']=='finish']
    if not all(t['outcome']=='completed' and t['reward']==1 for t in already_done):
        raise ValueError('Stopping on satisfied goals failed')
    context_map={c['id']:c for c in contexts}
    fresh=[r for r in rows if r['kind']=='decision' and context_map[r['context_id']]['regime']=='fresh']
    def shortcut_ceiling(projection):
        grouped=defaultdict(list)
        for row in fresh:grouped[app.digest(projection(row['input']))].append(row['target']['option_ids'])
        return sum(max(Counter(a for choices in targets for a in choices).values()) for targets in grouped.values())/len(fresh)
    def without_goal(item):
        state=json.loads(item['state']);state.pop('goal');return dict(state=state,options=item['options'])
    def without_history(item):
        state=json.loads(item['state']);state.pop('history');return dict(state=state,options=item['options'])
    ceilings=dict(menu_only=shortcut_ceiling(lambda i:i['options']),
                  observations_without_goal=shortcut_ceiling(without_goal),
                  goal_without_observations=shortcut_ceiling(without_history),fresh_contexts=len(fresh))
    if ceilings['menu_only']>=.5 or ceilings['goal_without_observations']>=.5 or ceilings['observations_without_goal']>=.8:
        raise ValueError('Fresh-context shortcut gate failed: '+json.dumps(ceilings))
    token_audit=None
    if tokenizer is not None:
        from scale_lab.common import messages
        from tool_lab.decision_rl import actor_input
        live_inputs={}
        for t in traces:
            for e in t['events']:
                item=actor_input(e['input'])
                live_inputs[app.digest(item)]=item
        # Encode full inputs, never infer safety from a truncated encoding.
        def length(item):
            text=tokenizer.apply_chat_template(messages(item),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            return len(tokenizer(text,add_special_tokens=False)['input_ids'])
        question_lengths=[length(r['input']) for r in rows]
        actor_lengths=[length(item) for item in live_inputs.values()]
        token_audit=dict(max_question=max(question_lengths),max_actor=max(actor_lengths),max_tokens=max_tokens,
                         questions=len(rows),unique_actor_histories=len(live_inputs))
        if max(question_lengths+actor_lengths)>max_tokens:
            raise ValueError('Untruncated input exceeds token limit: '+json.dumps(token_audit))
    return dict(status='passed',root_fixtures=1,semantic_world_configurations=semantic,
        concrete_initial_database_snapshots=concrete,base_world_goal_tasks=18,world_goal_context_cases=len(cases),
        distinct_executed_alternatives=len(traces),replay_executions=len(traces),production_and_replay_branches=len(traces)*2,
        setup_calls=n_setup,prefix_calls=n_prefix,branch_calls=n_branch,total_top_level_tool_calls=n_setup+n_prefix+n_branch,
        questions=dict(Counter(r['kind'] for r in rows)),public_contexts=len(contexts),token_audit=token_audit,
        fresh_context_shortcut_ceilings=ceilings,auditor_sha256=file_hash(Path(__file__)),
        aggregate_float_tolerance=1e-12,exact_question_and_label_match=True,
        trained_questions=0,optimizer_presentations=0,
        scope='Labels, frame conditions, visible inputs, full counterfactual menus and counts independently rechecked. '
        'Two deterministic executions per label; nine concrete initial databases. No learned performance claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenize',action='store_true')
    a=p.parse_args();tokenizer=None
    if a.tokenize:
        from transformers import AutoTokenizer
        from scale_lab.common import MODELS
        m=MODELS['qwen35-9b'];tokenizer=AutoTokenizer.from_pretrained(m['id'],revision=m['revision'],local_files_only=True)
    result=audit(a.folder,tokenizer);write_json(a.output,result);print(json.dumps(result,indent=2))
