"""Actual offline ToolSandbox messaging/settings decisions and counterfactuals.

The registered upstream send tool changes only its simulated database. Never
substitute a real messaging API. This module has no model/provider client.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import datetime
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from general_lab.toolsandbox_pilot import (COMMIT, NOW, SOURCE_HASHES, canonical,
    digest, load_backend, replay_guard, snapshot)
from scale_lab.common import file_hash, read_rows, validate_input, write_json, write_rows

VERSION = 'application-decisions-v1'
HORIZON = 7
MAX_BRANCHES, MAX_CALLS = 4536, 80000
PLANS = ('stop_now', 'evidence_then_complete', 'no_more_observations')
READS = ('messages', 'cellular_status', 'battery_status')
OUTCOMES = ('completed', 'incorrect', 'unfinished')
CONTINUATION = ('Use only this request and visible history. Stop when current message evidence proves '
    'completion. Otherwise query the current message ledger unless a query costs at least 1; if it does, '
    'stop while the ledger is unknown. With known missing delivery, obtain cellular status and, when '
    'cellular is off, low-battery status; disable low-battery mode if needed, then enable cellular and '
    'send the requested message. Stop after a successful send or an irreversible mistake. '
    'At most seven total decisions including the offered action.')
NO_READ = ('Never query again. Stop if the current message ledger is unknown or proves completion. '
    'With known missing delivery, try sending when cellular status is unknown or on. After a '
    'disconnected-service error, disable low-battery mode if its status is unknown or on, then enable '
    'cellular, then retry. Stop after a successful send or irreversible mistake. '
    'At most seven total decisions including the offered action.')
TOOLS = ('add_contact', 'search_messages', 'send_message_with_phone_number',
         'get_cellular_service_status', 'get_low_battery_mode_status',
         'set_cellular_service_status', 'set_low_battery_mode_status')


def fixtures():
    result = []
    for ledger in ('none', 'a', 'b'):
        for connection in ('ready', 'offline', 'low_battery'):
            for goal in ('a', 'b'):
                for regime in ('hidden', 'fresh', 'expensive', 'failed_send', 'stale'):
                    if regime == 'failed_send' and connection == 'ready':
                        continue
                    row = dict(group_id='message-connectivity:0', family='application_delivery',
                        split='train', ledger=ledger, connection=connection, goal=goal, regime=regime,
                        phones={'self': '+12025550100', 'a': '+12025550101', 'b': '+12025550102'},
                        content='Order Cedar is ready for collection.')
                    row['id'] = digest([VERSION, row])
                    result.append(row)
    return result


def backend_at(source):
    backend = load_backend(source)
    backend.messaging = importlib.import_module('tool_sandbox.tools.messaging')
    backend.setting = importlib.import_module('tool_sandbox.tools.setting')
    backend.functions.update({name: getattr(backend.messaging if name in
        ('search_messages', 'send_message_with_phone_number') else backend.setting, name)
        for name in TOOLS if name != 'add_contact'})
    return backend


@contextmanager
def deterministic(backend, root):
    counter = 0

    def next_uuid():
        nonlocal counter
        counter += 1
        return uuid.uuid5(uuid.NAMESPACE_URL, f'{VERSION}:{root}:{counter}')

    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(NOW, tz=tz or datetime.timezone.utc)

    with patch.object(backend.contact, 'uuid4', next_uuid), \
         patch.object(backend.messaging, 'uuid4', next_uuid), \
         patch.object(backend.messaging, 'datetime', SimpleNamespace(datetime=Clock)):
        yield


def call(tool, **arguments):
    return dict(tool=tool, arguments=arguments)


def menu(case):
    return dict(messages=call('search_messages', content=case['content']),
        cellular_status=call('get_cellular_service_status'),
        battery_status=call('get_low_battery_mode_status'),
        battery_off=call('set_low_battery_mode_status', on=False),
        cellular_on=call('set_cellular_service_status', on=True),
        battery_on=call('set_low_battery_mode_status', on=True),
        send_a=call('send_message_with_phone_number', phone_number=case['phones']['a'], content=case['content']),
        send_b=call('send_message_with_phone_number', phone_number=case['phones']['b'], content=case['content']),
        finish=None)


def invoke(backend, action, budget, phase):
    if sum(budget[k] for k in ('setup_calls', 'prefix_calls', 'branch_calls')) >= MAX_CALLS:
        raise RuntimeError('Application tool-call cap reached')
    budget[phase+'_calls'] += 1
    if action['tool'] not in TOOLS:
        raise ValueError('Tool outside immutable local allowlist')
    try:
        result = backend.functions[action['tool']](**action['arguments'])
        if isinstance(result, list):
            result = sorted(result, key=canonical)
        return dict(call=action, result=result)
    except (ValueError, PermissionError, ConnectionError) as error:
        if phase == 'setup':
            raise
        return dict(call=action, error=dict(type=type(error).__name__, message=str(error)))


def initialize(backend, case, budget):
    backend.context.set_current_context(backend.context.ExecutionContext(tool_allow_list=list(TOOLS)))
    setup = [call('add_contact', name=f'Contact {who}', phone_number=case['phones'][who], is_self=who=='self')
             for who in ('self', 'a', 'b')]
    if case['ledger'] != 'none':
        setup.append(menu(case)['send_'+case['ledger']])
    if case['connection'] == 'offline':
        setup.append(call('set_cellular_service_status', on=False))
    elif case['connection'] == 'low_battery':
        setup.append(call('set_low_battery_mode_status', on=True))
    receipts = [invoke(backend, action, budget, 'setup') for action in setup]
    return snapshot(backend), receipts


def verify(case, before, after):
    """Independent row/frame predicate; never consults a command's return code."""
    phone, content = case['phones'][case['goal']], case['content']
    protected = all(before[k] == after[k] for k in before if k not in ('messaging', 'setting'))
    mask = lambda rows: [{k: v for k, v in r.items() if k not in ('cellular', 'low_battery_mode')} for r in rows]
    protected = protected and mask(before['setting']) == mask(after['setting'])
    old = {r['message_id']: r for r in before['messaging']}
    current = {r['message_id']: r for r in after['messaging']}
    protected = protected and len(current) == len(after['messaging']) and all(current.get(k) == v for k,v in old.items())
    added = [r for k,r in current.items() if k not in old]
    self_contact = next(r for r in before['contact'] if r['is_self'])
    recipient = next(r for r in before['contact'] if r['phone_number'] == phone)
    expected = dict(sender_person_id=self_contact['person_id'], sender_phone_number=self_contact['phone_number'],
        recipient_person_id=recipient['person_id'], recipient_phone_number=phone, content=content, creation_timestamp=NOW)
    protected = protected and all({k:v for k,v in r.items() if k!='message_id'} == expected for r in added)
    matches = [r for r in after['messaging'] if r['recipient_phone_number']==phone and r['content']==content]
    unsafe = not protected or len(matches)>1
    return dict(completed=protected and len(matches)==1, incorrect=unsafe, frame_preserved=bool(protected))


class Episode:
    def __init__(self, backend, case, budget):
        self.backend, self.case, self.budget = backend, case, budget
        self.commands = menu(case)
        self.prefix, self.history, self.events = [], [], []
        self.setup = []
        # Execute an explicitly historical query in a separate old context.
        if case['regime'] == 'stale':
            # Its UUID counter must not perturb the current-world fixture.
            with deterministic(backend,case['group_id']+':historical'):
                _, calls = initialize(backend, dict(case, ledger='none', connection='ready'), budget)
                self.setup += calls
                old = invoke(backend, self.commands['messages'], budget, 'prefix')
            self.prefix.append(dict(action='messages', historical=True, observation=old))
        self.before, setup = initialize(backend, case, budget)
        self.setup += setup
        initial_actions = READS if case['regime']=='fresh' else ('send_'+case['goal'],) if case['regime']=='failed_send' else ()
        for action in initial_actions:
            observation = invoke(backend, self.commands[action], budget, 'prefix')
            if case['regime']=='failed_send' and observation.get('error',{}).get('type')!='ConnectionError':
                raise ValueError('Failed-send prefix did not fail as specified')
            self.prefix.append(dict(action=action, historical=False, observation=observation))
        if snapshot(backend) != self.before:
            raise ValueError('Read/failure prefix changed current world')
        self.history = list(self.prefix)
        self.done, self.cost, self.depth = False, 0., 0
        self.initial_input = self.input()

    def input(self):
        price = 1.2 if self.case['regime']=='expensive' else .02
        state = dict(goal=dict(recipient_phone=self.case['phones'][self.case['goal']], content=self.case['content'],
            requirement='Exactly one matching message; preserve all existing rows and unrelated settings throughout. '
                        'Only cellular and low-battery settings may change.'),
            prior='Equal prior: no matching messages, message to first recipient, or message to second recipient; '
                  'independently, cellular ready, cellular disabled, or low-battery mode. Historical rows are not current evidence.',
            tool_contract='Cellular must be enabled to send. Low-battery mode blocks enabling cellular. Disabling '
                'low-battery mode does not re-enable cellular. Enabling low-battery mode disables cellular, Wi-Fi and location.',
            history=self.history, remaining_decisions=HORIZON-self.depth,
            costs={a:price if a in READS else 0. if a=='finish' else .02 for a in self.commands})
        return dict(state=canonical(state), question='Which next action has greatest expected verified reward minus '
            'future costs under this continuation? '+CONTINUATION,
            options=[dict(id=a, description=canonical(cmd) if cmd else 'Stop without further calls.') for a,cmd in self.commands.items()])

    def step(self, action):
        if self.done or action not in self.commands:
            raise ValueError('Illegal or post-terminal action')
        item = self.input(); cost=json.loads(item['state'])['costs'][action]
        before = snapshot(self.backend)
        observation = invoke(self.backend,self.commands[action],self.budget,'branch') if action!='finish' else None
        after = snapshot(self.backend); verdict=verify(self.case,self.before,after)
        self.cost += cost; self.depth += 1
        self.done = action=='finish' or verdict['incorrect'] or (action.startswith('send_') and 'error' not in observation) or self.depth==HORIZON
        event=dict(action=action, input=item, input_sha256=digest(item), observation=observation,
                   before=before, after=after, cost=cost, terminal=self.done)
        self.events.append(event)
        if observation is not None:
            self.history.append(dict(action=action, historical=False, observation=observation))
        return event

    def receipt(self):
        after=snapshot(self.backend); verdict=verify(self.case,self.before,after)
        outcome='incorrect' if verdict['incorrect'] else 'completed' if verdict['completed'] else 'unfinished'
        return dict(case_id=self.case['id'],group_id=self.case['group_id'],family=self.case['family'],
            split=self.case['split'],regime=self.case['regime'],input=self.initial_input,
            setup=self.setup,prefix=self.prefix,before=self.before,after=after,events=self.events,
            outcome=outcome,cost=self.cost,reward={'completed':1,'incorrect':-1,'unfinished':0}[outcome]-self.cost,
            verdict=verdict)


def continuation(item, allow_reads=True):
    state=json.loads(item['state']); messages=None; cellular=None; battery=None
    for event in state['history']:
        if event['historical']:
            continue
        action=event['action']; obs=event['observation']; error=obs.get('error',{}).get('type')
        if error:
            if error=='ConnectionError': cellular=False
            if action=='cellular_on' and error=='PermissionError': battery=True; cellular=False
            if action=='battery_off' and error=='ValueError': battery=False
            continue
        if action=='messages': messages=obs['result']
        elif action=='cellular_status': cellular=obs['result']
        elif action=='battery_status': battery=obs['result']
        elif action=='battery_off': battery=False
        elif action=='battery_on': battery=True; cellular=False
        elif action=='cellular_on': cellular=True
    goal=state['goal']
    if messages is None:
        return 'messages' if allow_reads and state['costs']['messages']<1 else 'finish'
    if any(r['recipient_phone_number']==goal['recipient_phone'] and r['content']==goal['content'] for r in messages):
        return 'finish'
    if cellular is None and allow_reads: return 'cellular_status'
    if cellular is False:
        if battery is None and allow_reads: return 'battery_status'
        if battery is not False: return 'battery_off'
        return 'cellular_on'
    return next(o['id'] for o in item['options'] if o['id'].startswith('send_') and
                json.loads(o['description'])['arguments']['phone_number']==goal['recipient_phone'])


def forecast_input(item, action, plan):
    description=next(o['description'] for o in item['options'] if o['id']==action)
    contract='Execute only the offered action, then stop immediately, even after an error.' if plan=='stop_now' else CONTINUATION
    return dict(state=item['state'],question=f'What is the final verified outcome if I execute {description}? {contract}',
        options=[dict(id=x,description={'completed':'Goal completed with all protected state preserved.',
            'incorrect':'Duplicate/wrong delivery or protected state damaged.',
            'unfinished':'Goal incomplete without an irreversible mistake.'}[x]) for x in OUTCOMES])


def execute_branch(backend, case, action, plan, budget):
    if budget['branches']>=MAX_BRANCHES:
        raise RuntimeError('Application branch cap reached')
    budget['branches']+=1
    with deterministic(backend,case['group_id']):
        episode=Episode(backend,case,budget); episode.step(action)
        while not episode.done and plan!='stop_now':
            episode.step(continuation(episode.input(),plan!='no_more_observations'))
        trace=episode.receipt()
    trace.update(action=action,plan=plan,id=digest([VERSION,case['id'],action,plan]))
    if plan!='no_more_observations':
        trace['forecast_input']=forecast_input(trace['input'],action,plan)
    return trace


def reconstruct(case, trace):
    history=list(trace['prefix']); before=trace['before']; cost=0.
    if trace['case_id']!=case['id'] or trace['group_id']!=case['group_id']:
        raise ValueError('Changed fixture ownership')
    initial=json.loads(trace['input']['state'])
    if initial['history']!=history or not trace['events'] or trace['events'][0]['action']!=trace['action']:
        raise ValueError('Changed public prefix or offered action')
    commands=menu(case)
    for index,event in enumerate(trace['events']):
        state=dict(initial,history=history,remaining_decisions=HORIZON-index)
        item=dict(trace['input'],state=canonical(state))
        if item!=event['input'] or digest(item)!=event['input_sha256'] or event['before']!=before:
            raise ValueError('Changed public history or database chain')
        action=event['action'];obs=event['observation']
        if index and action!=continuation(item,trace['plan']!='no_more_observations'):
            raise ValueError('Hidden-state continuation or changed plan')
        if action not in commands or (action=='finish')!=(obs is None):
            raise ValueError('Illegal command')
        if obs is not None:
            if obs['call']!=commands[action]: raise ValueError('Command differs from menu')
            if action in READS and 'error' not in obs:
                expected=(sorted([r for r in before['messaging'] if r['content']==case['content']],key=canonical)
                    if action=='messages' else before['setting'][0]['cellular' if action=='cellular_status' else 'low_battery_mode'])
                if obs['result']!=expected: raise ValueError('Query result differs from actual database')
            if (action in READS or 'error' in obs) and before!=event['after']:
                raise ValueError('Read/failed command unexpectedly changed state')
            history=history+[dict(action=action,historical=False,observation=obs)]
        elif before!=event['after']: raise ValueError('Stopping changed state')
        if event['cost']!=state['costs'][action]: raise ValueError('Cost differs')
        cost+=event['cost'];before=event['after'];verdict=verify(case,trace['before'],before)
        terminal=(action=='finish' or verdict['incorrect'] or
                  (action.startswith('send_') and 'error' not in obs) or index+1==HORIZON)
        if terminal!=event['terminal'] or (terminal and index!=len(trace['events'])-1):
            raise ValueError('Changed terminal semantics')
    if trace['plan']=='stop_now' and len(trace['events'])!=1:raise ValueError('Immediate plan continued')
    if trace['plan']!='stop_now' and not trace['events'][-1]['terminal']:raise ValueError('Unfinished rollout')
    if before!=trace['after']:raise ValueError('Changed final state')
    verdict=verify(case,trace['before'],before)
    outcome='incorrect' if verdict['incorrect'] else 'completed' if verdict['completed'] else 'unfinished'
    reward={'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost
    if verdict!=trace['verdict'] or outcome!=trace['outcome'] or abs(reward-trace['reward'])>1e-9 or abs(cost-trace['cost'])>1e-9:
        raise ValueError('Verifier/reward differs')
    if trace['plan']!='no_more_observations' and trace['forecast_input']!=forecast_input(trace['input'],trace['action'],trace['plan']):
        raise ValueError('Changed forecast contract')


def questions(cases, traces):
    bycase={c['id']:c for c in cases};groups=defaultdict(list);rows=[];contexts=[]
    for trace in traces:
        case=bycase[trace['case_id']]
        reconstruct(case,trace)
        groups[(case['group_id'],digest(trace['input']))].append(trace)
        if trace['plan']!='no_more_observations':
            rows.append(dict(id=trace['id'],group_id=case['group_id'],case_id=case['id'],family=case['family'],
                split='train',kind='forecast',plan=trace['plan'],input=trace['forecast_input'],
                target={'option_id':trace['outcome']},receipt_ids=[trace['id']],receipt_sha256=[digest(trace)]))
    for (root,hash_),ts in groups.items():
        item=ts[0]['input'];actions=[o['id'] for o in item['options']];cells=defaultdict(list)
        compatible={t['case_id'] for t in ts}
        for t in ts:cells[t['plan'],t['action']].append(t)
        if set(cells)!={(p,a) for p in PLANS for a in actions} or any({t['case_id'] for t in v}!=compatible or len(v)!=len(compatible) for v in cells.values()):
            raise ValueError('Incomplete counterfactual menu')
        values={p:{a:sum(t['reward'] for t in cells[p,a])/len(compatible) for a in actions} for p in PLANS}
        best=max(values['evidence_then_complete'].values())
        targets=[a for a in actions if abs(values['evidence_then_complete'][a]-best)<1e-9]
        ident=digest([root,hash_]);common=dict(group_id=root,context_id=ident,split='train',family='application_delivery')
        deps=[t for t in ts if t['plan']=='evidence_then_complete']
        rows.append(dict(**common,id=digest([ident,'decision']),kind='decision',input=item,
            target={'option_ids':targets},receipt_ids=[t['id'] for t in deps],receipt_sha256=[digest(t) for t in deps]))
        alternatives={a:v for a,v in values['no_more_observations'].items() if a not in READS}
        gain=values['evidence_then_complete']['messages']-max(alternatives.values())
        value_item=dict(state=item['state'],question='Is a current message lookup worth its cost? Follow it with: '+
            CONTINUATION+' Compare with the best non-query action followed by: '+NO_READ+
            ' Include all future costs. A tie counts as no.',options=[dict(id='yes',description='Lookup plan has strictly greater expected return.'),
                dict(id='no',description='Best plan without observations is at least as good.')])
        deps=[t for t in ts if (t['plan']=='evidence_then_complete' and t['action']=='messages') or
              (t['plan']=='no_more_observations' and t['action'] not in READS)]
        rows.append(dict(**common,id=digest([ident,'value']),kind='observation_value',input=value_item,
            target={'option_id':'yes' if gain>1e-9 else 'no'},receipt_ids=[t['id'] for t in deps],receipt_sha256=[digest(t) for t in deps]))
        contexts.append(dict(id=ident,group_id=root,regime=ts[0]['regime'],input_sha256=hash_,
            compatible_worlds=len(compatible),values=values,best_actions=targets,observation_value=gain,
            distributions={p:{a:{o:sum(t['outcome']==o for t in cells[p,a])/len(compatible) for o in OUTCOMES}
                              for a in actions} for p in PLANS}))
    for row in rows:validate_input(row['input'])
    return rows,contexts


def collect(source,output):
    output.mkdir(parents=True,exist_ok=False);cases=fixtures();write_rows(output/'cases.jsonl',cases)
    source_paths=[Path(__file__),Path('general_lab/toolsandbox_pilot.py'),Path('tests/test_application_curriculum.py'),
                  Path('docs/application-curriculum-v1-protocol.md')]
    write_json(output/'pre-execution-freeze.json',dict(schema=VERSION,upstream_commit=COMMIT,
        upstream_hashes=SOURCE_HASHES,sources={str(p):file_hash(p) for p in source_paths},
        cases_sha256=file_hash(output/'cases.jsonl'),case_count=len(cases),max_branches=MAX_BRANCHES,max_calls=MAX_CALLS,
        root_fixtures=1,base_database_worlds=9,base_world_goal_tasks=18,plans=PLANS,split='train',trained_questions=0))
    for source_path in source_paths:
        relative=source_path.resolve().relative_to(Path.cwd())
        target=output/'executed-source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    budget=Counter(branches=0,setup_calls=0,prefix_calls=0,branch_calls=0)
    with replay_guard() as guard:
        backend=backend_at(source)
        guard['import_blocked_attempts']=guard['blocked_attempts'][guard['probe_attempt_count']:]
        # urllib3 probes IPv6 capability while importing requests. The socket
        # constructor remains blocked; no network access is permitted here.
        if guard['import_blocked_attempts'] not in ([],['socket.socket']):
            write_json(output/'execution-progress.json',dict(budget=budget,guard=guard))
            raise ValueError('Unexpected dependency-import capability attempt')
        guard['post_import_attempt_count']=len(guard['blocked_attempts'])
        write_json(output/'execution-progress.json',dict(budget=budget,guard=guard))
        with (output/'executions.jsonl').open('w') as stream:
            for i,case in enumerate(cases):
                for action in menu(case):
                    for plan in PLANS:
                        first=execute_branch(backend,case,action,plan,budget)
                        repeat=execute_branch(backend,case,action,plan,budget)
                        if first!=repeat:raise ValueError('Deterministic execution replay mismatch')
                        stream.write(canonical(first)+'\n')
                stream.flush()
                write_json(output/'execution-progress.json',dict(budget=budget,guard=guard))
                if len(guard['blocked_attempts'])!=guard['post_import_attempt_count']:
                    raise ValueError('Unexpected execution-phase network or subprocess attempt')
                if (i+1)%12==0:print(json.dumps(dict(completed_cases=i+1,**budget)),flush=True)
    traces=read_rows(output/'executions.jsonl');restored=read_rows(output/'cases.jsonl')
    rows,contexts=questions(restored,traces)
    uncertain=defaultdict(set)
    for row in rows:
        if row['kind']=='forecast':uncertain[digest(row['input'])].add(row['target']['option_id'])
    if not any(len(v)>1 for v in uncertain.values()):raise ValueError('Hidden uncertainty disappeared')
    for c in contexts:
        if c['regime']=='expensive' and (c['observation_value']>=0 or 'finish' not in c['best_actions']):
            raise ValueError('Cost-sensitive stopping failed')
        if c['regime'] in ('hidden','stale','failed_send') and c['observation_value']<=0:
            raise ValueError('Useful observation gate failed')
    if not all(any(t['case_id']==c['id'] and t['action']=='messages' and t['plan']=='evidence_then_complete' and t['outcome']=='completed' for t in traces)
               for c in cases if c['regime']!='expensive'):
        raise ValueError('Reference continuation failed an affordable case')
    write_rows(output/'questions.jsonl',rows);write_rows(output/'contexts-private.jsonl',contexts)
    result=dict(status='qualified',root_fixtures=1,base_database_worlds=9,base_world_goal_tasks=18,
        world_goal_context_cases=len(cases),distinct_executed_alternatives=len(traces),**budget,
        top_level_tool_calls=sum(budget[k] for k in ('setup_calls','prefix_calls','branch_calls')),
        public_contexts=len(contexts),questions=dict(Counter(r['kind'] for r in rows)),
        uncertain_forecast_inputs=sum(len(v)>1 for v in uncertain.values()),training_questions_consumed=0,
        optimizer_presentations=0,guard=guard,files={p.name:file_hash(p) for p in output.glob('*.jsonl')},
        limits='One authored application mechanism and one root. Actual upstream local simulated database tools, fixed menus, '
               'specified continuations. Two executions per branch, not twice as many independent labels. No Harbor, model call or 9B training.')
    write_json(output/'qualification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(collect(a.source,a.output),indent=2))
