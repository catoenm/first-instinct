"""Execute the predeclared bounded witnesses and their independent replays."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tool_lab.revisioned_sqlite import execute,GOALS
from tool_lab.revisioned_sqlite_audit import audit
from scale_lab.common import digest,file_hash,write_json,write_rows

PLANS={
    'stop':['finish'],
    'cached_write':['cached_write'],
    'conditional_then_stop':['checked_write','finish'],
    'atomic_increment':['increment'],
    'read_then_conditional':['read','checked_write','finish'],
    'conditional_recovery':['checked_write','read','checked_write','finish'],
    'failed_read_recovery':['missing_read','read','checked_write','finish'],
    'redundant_read':['read','read','checked_write','finish']}
PROFILES=('cheap','expensive_read','expensive_write')


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sources=['tool_lab/revisioned_sqlite.py','tool_lab/revisioned_sqlite_audit.py',
             'tool_lab/revisioned_sqlite_qualify.py','tests/test_revisioned_sqlite.py',
             'docs/revisioned-sqlite-v1-protocol.md','docs/mechanism-overlap-v1.md']
    freeze=dict(stage='revisioned-sqlite-v1',sources={n:file_hash(root/n) for n in sources},
                witness_plans=PLANS,profiles=PROFILES,world_prior=[.5,.5],maximum_resets=192,
                maximum_actor_commands=768,model_calls=0,training_questions=0,ownership='training_candidate')
    write_json(output/'pre-execution-freeze.json',freeze)
    rows=[];first={};initial={};counts=Counter()
    with (output/'executions-private.jsonl').open('x') as stream:
        for replica in (0,1):
            for goal in GOALS:
                for world in (False,True):
                    for profile in PROFILES:
                        for name,actions in PLANS.items():
                            key=(goal,world,profile,name)
                            t=execute(goal,world,profile,actions);result=audit(t)
                            record=dict(identity=list(key),replica=replica,trace=t,verified=result)
                            stream.write(json.dumps(record,sort_keys=True)+'\n');stream.flush()
                            counts.update(database_resets=1,prefix_reads=1,colleague_writes=int(world),
                                          actor_commands=result['actor_commands'],failed_commands=result['failed_commands'])
                            if replica:
                                if t!=first[key]['trace'] or result!=first[key]['verified']:raise ValueError('Independent replay differs')
                            else:
                                first[key]=record;rows.append(record)
                                public=t['events'][0]['input'];group=(goal,profile)
                                if group in initial and initial[group]!=public:raise ValueError('Private schedule reached visible input')
                                initial[group]=public
    if counts['database_resets']!=192 or counts['actor_commands']>768:raise ValueError('Execution cap differs')
    means={}
    for goal in GOALS:
        means[goal]={}
        for profile in PROFILES:
            means[goal][profile]={name:sum(first[(goal,w,profile,name)]['verified']['utility'] for w in (False,True))/2 for name in PLANS}
    increment=means['increment_latest'];value={}
    no_read=['stop','cached_write','conditional_then_stop','atomic_increment']
    for profile,values in increment.items():
        value[profile]=values['read_then_conditional']-max(values[n] for n in no_read)
    if not value['cheap']>0 or not value['expensive_read']<0:raise ValueError('Observation-cost reversal absent')
    if max(increment['expensive_write'],key=increment['expensive_write'].get)!='stop':raise ValueError('Stopping witness absent')
    if first[('increment_latest',True,'cheap','atomic_increment')]['verified']['outcome']!='completed' or first[('approved_revision',True,'cheap','atomic_increment')]['verified']['outcome']!='incorrect':
        raise ValueError('Goal-dependent action reversal absent')
    forecasts={}
    for goal in GOALS:
        forecasts[goal]={}
        for plan in no_read:
            records=[first[(goal,w,'cheap',plan)] for w in (False,True)]
            forecasts[goal][plan]=dict(goal_outcomes={outcome:sum(r['verified']['outcome']==outcome for r in records)/2 for outcome in ('completed','unfinished','incorrect')},
                first_command_returned_zero=sum(r['trace']['events'][0]['observation']['returncode']==0 for r in records)/2,
                horizon='execute the named first command, then stop if it did not terminate')
    summary=dict(status='qualified_executor_witnesses',fixture_roots=1,initial_databases=1,post_schedule_states=2,
                 goals=2,world_goal_tasks=4,fee_profiles=3,primary_executed_branches=96,replay_branches=96,
                 execution=dict(counts),distinct_initial_visible_inputs=len(initial),mechanisms=1,
                 model_calls=0,training_questions=0,optimizer_presentations=0,
                 means=means,observation_value_against_best_no_more_reads=value,immediate_forecasts=forecasts,
                 transfer_eligible=False,limitations='Authored finite witness plans; no global-optimality claim, question rendering or model/token qualification.')
    write_json(output/'summary.json',summary)
    # Read-back audit includes every replay. Predictions never supply labels.
    reloaded=[json.loads(line) for line in (output/'executions-private.jsonl').read_text().splitlines()]
    if len(reloaded)!=192 or any(audit(r['trace'])!=r['verified'] for r in reloaded):raise ValueError('Read-back audit differs')
    write_json(output/'freeze.json',dict(sources=freeze['sources'],files={n:file_hash(output/n) for n in
               ['pre-execution-freeze.json','executions-private.jsonl','summary.json']},status=summary['status']))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('means','immediate_forecasts')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
