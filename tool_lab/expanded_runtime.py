"""Live adapters over qualified executors. No model client or training launcher."""
from copy import deepcopy
import json
from pathlib import Path
import random
import time

from scale_lab.common import digest
from tool_lab import filesystem_decisions as fs, calendar_decisions as calendar
from puffer_lab.sql_oracle import SqlEpisode
from puffer_lab.contract import ACTIONS, CODES, WORLDS, continuation as reservation_reference
from puffer_lab.text_render import DESCRIPTIONS as RESERVATION_DESCRIPTIONS, profiles

VERSION = 'expanded-runtime-v1'
UTILITY = {'completed':1., 'incorrect':-1., 'unfinished':0.}
ACTOR_QUESTION = ('Choose the next action to maximize expected total future reward. You choose again after '
    'each nonterminal action; no fixed continuation takes over. At termination, verified completion earns 1, '
    'an incorrect final state earns -1, and an unfinished valid state earns 0. Subtract every future action '
    'cost in these same normalized reward units. Costs already paid are sunk.')


class Budget:
    def __init__(self, journal, max_episodes=800, max_actions=9000, max_seconds=900):
        self.journal=Path(journal)
        if self.journal.exists():raise FileExistsError('Use a fresh runtime attempt journal')
        self.maximum=max_episodes;self.max_actions=max_actions;self.max_seconds=max_seconds
        self.episodes=0;self.actions=0;self.started=time.monotonic()
        self.counts=dict(file_commands=0,reservation_actions_including_finish=0,sql_statements_traced=0)
        self.journal.parent.mkdir(parents=True,exist_ok=True)
        self.journal.touch()

    def check(self):
        if time.monotonic()-self.started>self.max_seconds:raise TimeoutError('Runtime qualification deadline')

    def log(self, entry):
        with self.journal.open('a') as f:f.write(json.dumps(entry,sort_keys=True)+'\n');f.flush()

    def claim(self, identity):
        self.check()
        if self.episodes>=self.maximum:raise RuntimeError('Physical episode cap')
        self.episodes+=1;self.log(dict(phase='started_episode',attempt=self.episodes,identity=identity))

    def action(self, identity):
        self.check()
        if self.actions>=self.max_actions:raise RuntimeError('Tool-action cap')
        self.actions+=1;self.log(dict(phase='started_action',number=self.actions,identity=identity))


def actor_input(item):
    return dict(state=item['state'],options=deepcopy(item['options']),question=ACTOR_QUESTION)


def filesystem_after(before, action):
    """Independent logical transition, without executing or importing programs."""
    after=deepcopy(before)
    if action.startswith(('write_', 'replace_')):
        names=['a.txt','b.txt'] if action.endswith('_ab') else ['a.txt']
        for name in names:
            if action.startswith('write_'):
                for alias in list(after[name]['aliases']):after[alias]['contents']=b'new'.hex()
            else:
                for alias in list(after[name]['aliases']):
                    if alias!=name:after[alias]['aliases'].remove(name)
                after[name]=dict(contents=b'new'.hex(),mode=0o600,aliases=[name])
    return after


class FileEpisode:
    def __init__(self, case, budget):
        self.case=case;self.budget=budget;self.env=fs if case['family']=='filesystem_scope' else calendar
        if case['family'] not in ('filesystem_scope','calendar'):raise ValueError('Unsupported file-backed mechanism')
        budget.claim(case['id']);self.attempt=budget.episodes;self.world=self.env.World();self.depth=0;self.done=False;self.closed=False
        self.history=[];self.prefix=[];self.events=[];self.cost=0.
        try:
            self.world.reset(case)
            first=('inspect' if case['regime']=='fresh' else None) if self.env is fs else {
                'fresh':'inspect','stale':'cached','failed':'bad_calendar'}.get(case['regime'])
            if first:
                budget.action(case['id']+':prefix:'+first)
                e=dict(action=first,observation=self.world.execute(first));self.prefix.append(e);self.history.append(e)
                if self.world.current!=self.world.initial:raise ValueError('Prefix changed committed state')
            self.initial_input=self.input()
        except BaseException:self.world.close();raise

    def input(self):return actor_input(self.env.public_input(self.case,self.history,self.depth))

    def outcome(self):
        if self.env is calendar:return calendar.classify(self.case,self.world.initial,self.world.current)
        return 'completed' if fs.verify(self.case,self.world.initial,self.world.current) else (
            'incorrect' if self.world.current!=self.world.initial else 'unfinished')

    def step(self, action):
        if self.done:raise ValueError('Cannot act after termination')
        before=self.input()
        if action not in {o['id'] for o in before['options']}:raise ValueError('Action outside offered menu')
        self.budget.action(self.case['id']+':'+str(self.depth)+':'+action)
        old=self.world.current
        obs=None if action=='finish' else self.world.execute(action)
        price=self.env.costs(self.case)[action];self.cost+=price;self.depth+=1
        if obs is not None:self.history.append(dict(action=action,observation=obs))
        self.done=bool(self.world.current!=old or action=='finish' or self.depth==self.env.HORIZON)
        reward=-price+(UTILITY[self.outcome()] if self.done else 0.)
        event=dict(input=before,action=action,observation=obs,cost=price,reward=reward,
                   terminal=self.done,after=deepcopy(self.world.current))
        self.events.append(event);return event

    def receipt(self):
        if not self.done:raise ValueError('Receipt requires a terminal episode')
        return dict(version=VERSION,case_id=self.case['id'],family=self.case['family'],regime=self.case['regime'],
            prefix=deepcopy(self.prefix),events=deepcopy(self.events),input=self.initial_input,
            before=self.world.initial,after=self.world.current,outcome=self.outcome(),cost=self.cost,
            reward=UTILITY[self.outcome()]-self.cost,policy_contract='live_replanning')

    def close(self):
        if not self.closed:
            self.budget.counts['file_commands']+=self.world.count;self.world.close();self.closed=True


def sql_snapshot(engine):
    db=engine.db
    return dict(schema=[list(r) for r in db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name')],
        accounts=[list(r) for r in db.execute('SELECT * FROM accounts ORDER BY id')],
        inventory=[list(r) for r in db.execute('SELECT * FROM inventory ORDER BY sku')],
        requests=[list(r) for r in db.execute('SELECT * FROM requests ORDER BY id')],
        allocations=[list(r) for r in db.execute('SELECT * FROM allocations ORDER BY request,sku')],
        audit=[list(r) for r in db.execute('SELECT * FROM audit ORDER BY id')],
        integrity=[list(r) for r in db.execute('PRAGMA integrity_check')],
        foreign_key_violations=[list(r) for r in db.execute('PRAGMA foreign_key_check')])


def verify_reservation(case, initial, final, all_actions):
    """Independent final goal/frame/conservation predicate over actual SQL rows."""
    start_a,start_b,account=WORLDS[case['world']];added={'A':0,'B':0}
    previous=initial
    for event in all_actions:
        if event['action'] in ('replenish_a','replenish_b'):
            sku=event['action'][-1].upper()
            if dict(previous['inventory'])[sku]==0:added[sku]+=1
        previous=event['after']
    quantities=dict(final['inventory'])
    own=[r for r in final['allocations'] if r[0]==17];allocated={r[1]:r[2] for r in own}
    request=[r for r in final['requests'] if r[0]==17]
    accounts=[r for r in final['accounts'] if r[0]==7]
    allowed_accounts=[] if not accounts else [[7,'target',23 if account else 0]]
    invalid=(final['schema']!=initial['schema'] or final['integrity']!=[['ok']] or final['foreign_key_violations'] or
        final['audit']!=[[1,'preserve']] or final['accounts']!=allowed_accounts+[[99,'protected',700]] or
        (account and not accounts) or set(quantities)!={'A','B','C'} or quantities.get('C')!=7 or
        any(q<0 for q in quantities.values()) or final['requests']!=(request+[[999,99]]) or
        request not in ([],[[17,7]]) or (request and not accounts) or
        final['allocations']!=sorted(own+[[999,'C',1]]) or
        any(sku not in ('A','B') or q!=1 for sku,q in allocated.items()) or len(own)!=len(allocated) or
        (own and not request) or quantities.get('A',-999)+allocated.get('A',0)!=start_a+added['A'] or
        quantities.get('B',-999)+allocated.get('B',0)!=start_b+added['B'])
    if invalid:return 'incorrect'
    if request and allocated=={'A':1,'B':1}:return 'completed'
    return 'incorrect' if request or own else 'unfinished'


def reservation_transition(before, public, action, profile):
    """Verifier-only relational/observation transition, separate from SQLite code."""
    after=deepcopy(before);p=deepcopy(public);stock=dict(after['inventory']);saved=deepcopy(p)
    code='unchanged'
    if action=='inspect_stock':p['stock']={k:stock[k] for k in ('A','B')};code='stock_read'
    elif action=='inspect_account':p['account']=int(any(r[0]==7 for r in after['accounts']));code='account_read'
    elif action in ('atomic','sequential'):
        code='reserved'
        if any(r[0]==17 for r in after['requests']):code='duplicate'
        elif not any(r[0]==7 for r in after['accounts']):code='missing_account'
        else:
            after['requests'].append([17,7]);p['account']=1
            for sku in ('A','B'):
                if stock[sku]==0:code='short_'+sku.lower();break
                stock[sku]-=1
                if p['stock'][sku]>=0:p['stock'][sku]-=1
                after['allocations'].append([17,sku,1])
        if code!='reserved':
            if action=='atomic':after=deepcopy(before);stock=dict(after['inventory']);p=deepcopy(saved)
            if code=='missing_account':p['account']=0
            elif code in ('short_a','short_b'):p['account']=1;p['stock'][code[-1].upper()]=0
    elif action in ('replenish_a','replenish_b'):
        sku=action[-1].upper()
        if stock[sku]==0:stock[sku]=1;p['stock'][sku]=1;code='replenished'
    elif action=='create_account':
        if not any(r[0]==7 for r in after['accounts']):after['accounts'].append([7,'target',0]);code='account_created'
        p['account']=1
    elif action=='undo':
        for _,sku,q in [r for r in after['allocations'] if r[0]==17]:
            stock[sku]+=q
            if p['stock'][sku]>=0:p['stock'][sku]+=q
        after['allocations']=[r for r in after['allocations'] if r[0]!=17]
        after['requests']=[r for r in after['requests'] if r[0]!=17];code='undone'
    elif action=='finish':code='finished'
    else:raise ValueError('Unknown reservation action')
    after['inventory']=[[k,v] for k,v in sorted(stock.items())]
    for name in ('accounts','requests','allocations'):after[name].sort()
    own={r[1]:r[2] for r in after['allocations'] if r[0]==17}
    p.update(step=p['step']+1,remaining=p['remaining']-1,last_result=code,
        cost_quarters=p['cost_quarters']+profile['costs'][ACTIONS.index(action)],
        request=dict(header=int(any(r[0]==17 for r in after['requests'])),A=own.get('A',0),B=own.get('B',0)))
    return after,p


def reservation_initial_public(case):
    return dict(step=0,remaining=case['profile']['horizon'],stock={'A':-1,'B':-1},account=-1,
        request={'header':0,'A':0,'B':0},last_result='none',cost_quarters=0,
        costs_quarters=case['profile']['costs'],prior_weights=case['profile']['prior'])


def reservation_input(public, history):
    costs={a:n/400. for a,n in zip(ACTIONS,public['costs_quarters'])}
    state=dict(task='Create request 17 for account 7 with exactly one allocation each of A and B. '
        'Preserve unrelated accounts, inventory, requests, allocations, audit and schema. Replenishments and account creation are allowed.',
        semantics='Reservation inserts its header, decrements A and adds its allocation, then decrements B and adds its allocation. '
        'A missing account, duplicate request or zero stock rejects the corresponding statement. Atomic failure rolls back all writes; '
        'sequential failure keeps earlier writes. Undo removes only this request and restores allocated stock. '
        'Partial writes can be repaired before termination. A complete request earns 1 only when stopping or reaching the horizon; '
        'an unrepaired partial request or damaged protected state earns -1, otherwise 0. Fees below use these same normalized units.',
        observation=deepcopy(public),history=deepcopy(history),costs=costs,
        initial_prior=[dict(A=a,B=b,account=bool(c),weight=w) for (a,b,c),w in zip(WORLDS,public['prior_weights']) if w],
        decisions_remaining=public['remaining'])
    # Remove quarter-credit vocabulary from the normalized actor interface.
    state['observation'].pop('costs_quarters');paid=state['observation'].pop('cost_quarters')
    state['observation']['sunk_cost']=paid/400.
    return dict(state=json.dumps(state,sort_keys=True),question=ACTOR_QUESTION,
        options=[dict(id=a,description=d+f' Normalized cost: {costs[a]}.')
                 for a,d in zip(ACTIONS,RESERVATION_DESCRIPTIONS)])


class ReservationEpisode:
    def __init__(self, case, budget):
        self.case=case;self.budget=budget;self.events=[];self.prefix=[];self.history=[];self.depth=0;self.cost=0.;self.done=False
        # SqlEpisode claims its database initialization before constructing it.
        self.engine=SqlEpisode(case['world'],case['profile'],budget,case['id']);self.attempt=budget.episodes;self.closed=False
        try:
            self.before=sql_snapshot(self.engine)
            for action in case['prefix']:
                budget.action(case['id']+':prefix:'+action);self.engine.step(action)
                event=dict(action=action,observation=self.engine.public(),after=sql_snapshot(self.engine))
                self.prefix.append(event);self.history.append(dict(action=action,result=event['observation']['last_result']))
                if self.engine.done:raise ValueError('Prefix exhausted reservation episode')
            self.sunk_cost=self.engine.cost/400.;self.initial_input=self.input()
        except BaseException:self.engine.close();raise

    def input(self):return reservation_input(self.engine.public(),self.history)

    def step(self, action):
        if self.done:raise ValueError('Cannot act after termination')
        before=self.input()
        if action not in {o['id'] for o in before['options']}:raise ValueError('Action outside offered menu')
        self.budget.action(self.case['id']+':'+str(self.depth)+':'+action)
        reward=self.engine.step(action);self.depth+=1;self.done=bool(self.engine.done)
        price=self.case['profile']['costs'][ACTIONS.index(action)]/400.;self.cost+=price
        obs=self.engine.public();self.history.append(dict(action=action,result=obs['last_result']))
        event=dict(input=before,action=action,observation=obs,after=sql_snapshot(self.engine),
                   cost=price,reward=reward,terminal=self.done)
        self.events.append(event)
        if self.done:
            outcome=verify_reservation(self.case,self.before,event['after'],self.prefix+self.events)
            if abs(reward-(-price+UTILITY[outcome]))>1e-9:raise ValueError('Independent reservation reward disagrees')
        return event

    def receipt(self):
        if not self.done:raise ValueError('Receipt requires a terminal episode')
        after=sql_snapshot(self.engine);outcome=verify_reservation(self.case,self.before,after,self.prefix+self.events)
        return dict(version=VERSION,case_id=self.case['id'],family='reservation',regime=self.case['regime'],
            prefix=deepcopy(self.prefix),events=deepcopy(self.events),input=self.initial_input,before=self.before,after=after,
            outcome=outcome,cost=self.cost,sunk_prefix_cost=self.sunk_cost,
            reward=UTILITY[outcome]-self.cost,policy_contract='live_replanning')

    def close(self):
        if not self.closed:
            self.budget.counts['reservation_actions_including_finish']+=self.engine.steps
            self.budget.counts['sql_statements_traced']+=len(self.engine.statements)
            self.engine.close();self.closed=True


def reservation_cases():
    result=[]
    for profile in profiles():
        for world,weight in enumerate(profile['prior']):
            if not weight:continue
            for prefix in ([],['inspect_stock'],['sequential']):
                c=dict(family='reservation',group_id=digest(['reservation-mechanism']),split='train_candidate',
                    world=world,profile=profile,prefix=prefix,regime='hidden' if not prefix else prefix[0])
                c['id']=digest([VERSION,c]);result.append(c)
    return result


def make_episode(case, budget):
    return ReservationEpisode(case,budget) if case['family']=='reservation' else FileEpisode(case,budget)


def reference_action(episode):
    if isinstance(episode,ReservationEpisode):return reservation_reference(episode.engine.public())
    return episode.env.continuation(episode.input())


def audit_trace(case, trace):
    """Public history/cost/termination audit, independent of actor choice."""
    if trace['case_id']!=case['id'] or trace['policy_contract']!='live_replanning':raise ValueError('Trace identity differs')
    if not trace['events'] or not trace['events'][-1]['terminal']:raise ValueError('Incomplete trajectory')
    total=0.;cost=0.;history=[];previous=trace['before']
    reservation=case['family']=='reservation';env=fs if case['family']=='filesystem_scope' else calendar
    def fs_observation(before, action):
        if action=='finish':return None
        if action=='inspect':value=json.dumps({'groups':sorted({tuple(before[n]['aliases']) for n in fs.PATHS[:3]})},sort_keys=True)
        elif action=='contents':value=json.dumps({n:bytes.fromhex(before[n]['contents']).decode() for n in fs.PATHS[:3]},sort_keys=True)
        else:value={'write_a':'Write completed','write_ab':'Writes completed','replace_a':'Replacement completed','replace_ab':'Replacements completed'}[action]
        return dict(returncode=0,stdout=value+'\n',stderr='')
    if not reservation:
        if env is calendar:
            # Reuse the independent initial-schema and prefix checker on a
            # synthetic no-action control; no fabricated execution is recorded.
            item=calendar.public_input(case,trace['prefix'],0)
            control=dict(id=digest([calendar.VERSION,case['id'],'finish','stop_now']),action='finish',plan='stop_now',
                before=previous,after=previous,prefix=trace['prefix'],input=item,
                forecast_input=calendar.forecast(item,'finish','stop_now'),outcome='unfinished',cost=0.,reward=0.,
                events=[dict(input=item,action='finish',observation=None,after=previous,cost=0.,terminal=True)])
            calendar.audit_branch(case,control)
        else:
            if previous!=fs.expected_initial(case):raise ValueError('Wrong initial filesystem world')
            expected_prefix=[dict(action='inspect',observation=fs_observation(previous,'inspect'))] if case['regime']=='fresh' else []
            if trace['prefix']!=expected_prefix:raise ValueError('Wrong initial filesystem evidence')
        history=deepcopy(trace['prefix'])
    else:
        a,b,account=WORLDS[case['world']]
        expected=dict(accounts=([[7,'target',23]] if account else [])+[[99,'protected',700]],
            inventory=[['A',a],['B',b],['C',7]],requests=[[999,99]],allocations=[[999,'C',1]],
            audit=[[1,'preserve']],integrity=[['ok']],foreign_key_violations=[])
        if any(previous[k]!=v for k,v in expected.items()):raise ValueError('Wrong initial reservation world')
        public=reservation_initial_public(case)
        if [p['action'] for p in trace['prefix']]!=case['prefix']:raise ValueError('Reservation prefix differs')
        for p in trace['prefix']:
            after,observed=reservation_transition(previous,public,p['action'],case['profile'])
            if after!=p['after'] or observed!=p['observation']:raise ValueError('Reservation prefix transition differs')
            previous=after;public=observed;history.append(dict(action=p['action'],result=observed['last_result']))
        if abs(trace['sunk_prefix_cost']-public['cost_quarters']/400.)>1e-9:raise ValueError('Prefix costs differ')
    for index,e in enumerate(trace['events']):
        if reservation:
            expected=reservation_input(public,history)
            price=case['profile']['costs'][ACTIONS.index(e['action'])]/400.
            terminal=e['action']=='finish' or len(case['prefix'])+index+1==case['profile']['horizon']
            after,observed=reservation_transition(previous,public,e['action'],case['profile'])
            if after!=e['after'] or observed!=e['observation']:raise ValueError('Reservation executed transition differs')
            public=observed;history.append(dict(action=e['action'],result=observed['last_result']))
        else:
            expected=actor_input(env.public_input(case,history,index));price=env.costs(case)[e['action']]
            if env is calendar:
                after,obs=(previous,None) if e['action']=='finish' else calendar.independent_transition(case,previous,e['action'])
                if after!=e['after'] or obs!=e['observation']:raise ValueError('Calendar executed transition differs')
            elif (filesystem_after(previous,e['action'])!=e['after'] or fs_observation(previous,e['action'])!=e['observation']):
                raise ValueError('Filesystem executed transition differs')
            terminal=e['action']=='finish' or e['after']!=previous or index+1==env.HORIZON
            if e['observation'] is not None:history.append(dict(action=e['action'],observation=e['observation']))
        if e['input']!=expected or e['action'] not in {o['id'] for o in expected['options']}:
            raise ValueError('Actual public input/action differs')
        if e['cost']!=price or terminal!=e['terminal'] or (terminal and index!=len(trace['events'])-1):
            raise ValueError('Cost or terminal rule differs')
        if abs(e['reward']-(-price+(UTILITY[trace['outcome']] if terminal else 0.)))>1e-9:
            raise ValueError('Terminal reward was not paid exactly once')
        total+=e['reward'];cost+=price;previous=e['after']
    outcome=(verify_reservation(case,trace['before'],trace['after'],trace['prefix']+trace['events']) if reservation else
             calendar.classify(case,trace['before'],trace['after']) if env is calendar else
             'completed' if fs.verify(case,trace['before'],trace['after']) else
             'incorrect' if trace['before']!=trace['after'] else 'unfinished')
    if (outcome!=trace['outcome'] or previous!=trace['after'] or trace['input']!=trace['events'][0]['input'] or
        abs(total-trace['reward'])>1e-9 or abs(cost-trace['cost'])>1e-9):raise ValueError('Executed final return differs')


def execute(case, budget, mode='reference', actions=None):
    ep=make_episode(case,budget)
    try:
        while not ep.done:
            if actions is not None:
                if ep.depth>=len(actions):raise ValueError('Replay action list ended early')
                action=actions[ep.depth]
            elif mode=='reference':action=reference_action(ep)
            elif mode=='random':
                # The seed uses visible input only, never a private case/world identity.
                action=random.Random('runtime-743:'+digest(ep.input())).choice([o['id'] for o in ep.input()['options']])
            else:raise ValueError('Unknown qualification controller')
            ep.step(action)
        if actions is not None and len(actions)!=ep.depth:raise ValueError('Replay has actions after termination')
        trace=ep.receipt();audit_trace(case,trace)
        budget.log(dict(phase='completed_episode',attempt=ep.attempt,identity=case['id'],receipt_sha256=digest(trace)))
        return trace
    finally:ep.close()
