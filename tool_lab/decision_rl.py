"""Live-policy interface for the v3 environment; no training launcher.

Forecasts name a fixed continuation. The reinforcement actor instead controls
all remaining choices itself. Never train its return under a prompt promising
that a different fixed continuation will take over.
"""
import json
import math

import torch

from general_lab.outcome_train import normalize_advantages
from general_lab.rl import prepare
from scale_lab.common import digest
from tool_lab.decision_curriculum import Episode, HORIZON


def actor_input(item):
    return dict(state=item['state'],options=item['options'],
        question='Choose the next action to maximize total expected task reward. You will choose again after each '
                 'nonterminal action until the remaining decision budget expires. Verified completion earns 1; '
                 'an incorrect irreversible mutation earns -1; unfinished earns 0. Subtract all future command '
                 'costs. Costs already paid are sunk. No fixed continuation takes over your choices.')


@torch.no_grad()
def collect(policy,tokenizer,cases,executors,max_tokens,check,sample=True,*,episode_factory=None,horizon=None):
    if not executors:raise ValueError('At least one executor is required')
    factory=episode_factory or Episode;limit=HORIZON if horizon is None else horizon
    if not 1<=limit<=16:raise ValueError('Invalid rollout horizon')
    policy.eval();records=[];traces=[]
    for start in range(0,len(cases),len(executors)):
        active=[factory(c,e) for c,e in zip(cases[start:start+len(executors)],executors)]
        episode_records=[[] for _ in active];actor_events=[[] for _ in active]
        for depth in range(limit):
            check();pending=[(i,e) for i,e in enumerate(active) if not e.done]
            if not pending:break
            inputs=[actor_input(e.input()) for _,e in pending]
            rows=[prepare(tokenizer,item,digest([e.case['id'],depth,'live-policy',digest(item)]),max_tokens)
                  for item,(_,e) in zip(inputs,pending)]
            for row in rows:row['task']='shell_action'
            logits,values,_=policy(rows);distribution=torch.distributions.Categorical(logits=logits)
            choices=distribution.sample() if sample else logits.argmax(-1)
            logps=distribution.log_prob(choices).cpu().tolist();probs=distribution.probs.cpu().tolist()
            for j,((i,ep),row,index) in enumerate(zip(pending,rows,choices.cpu().tolist())):
                if actor_input(ep.input())!=inputs[j]:raise ValueError('Actor observation changed')
                ep.step(row['option_ids'][index])
                # Pay costs as they occur, and terminal utility exactly once.
                # A world can already satisfy its goal before the first action:
                # stopping there must earn +1, not a zero difference from an
                # initial potential that the actor was never actually paid.
                reward=-ep.events[-1]['cost']
                if ep.done:
                    reward+={'completed':1.,'incorrect':-1.,'unfinished':0.}[ep.receipt()['outcome']]
                event=dict(input=inputs[j],input_sha256=digest(inputs[j]),encoded_input=row,
                    action=row['option_ids'][index],old_probabilities=probs[j][:len(row['option_ids'])],
                    sampled_log_probability=logps[j],value=float(values[j]),reward=reward,
                    observation=ep.events[-1]['observation'],terminal=ep.done)
                actor_events[i].append(event)
                episode_records[i].append(dict(row=row,action=index,old_logp=logps[j],
                    old_probabilities=event['old_probabilities'],old_value=float(values[j]),reward=reward))
        for ep,rs,events in zip(active,episode_records,actor_events):
            if not ep.done:raise ValueError('Incomplete actor episode')
            future=0.
            for r in reversed(rs):future+=r['reward'];r['return']=future
            trace=ep.receipt()
            if not math.isclose(future,trace['reward'],abs_tol=1e-9):raise ValueError('Actor rewards do not match executed return')
            # Original environment inputs remain in events for replay; these are
            # the separate questions actually passed to the language policy.
            trace['actor_events']=events;trace['policy_contract']='live_replanning'
            traces.append(trace);records.extend(rs)
    if sample:normalize_advantages(records)
    return records,traces


@torch.no_grad()
def forecast_metrics(policy,rows,batch_size,check):
    policy.eval();predictions=[]
    for start in range(0,len(rows),batch_size):
        check();chunk=rows[start:start+batch_size];logits,_,_=policy(chunk)
        for row,values in zip(chunk,logits.softmax(-1).cpu().tolist()):
            target=row['target_indices'][0]
            p=values[:len(row['option_ids'])]
            predictions.append(dict(id=row['id'],group_id=row['group_id'],
                option_ids=row['option_ids'],target_index=target,probabilities=p,
                brier=sum((v-int(i==target))**2 for i,v in enumerate(p)),
                log_loss=-math.log(max(1e-12,p[target])),correct=max(range(len(p)),key=p.__getitem__)==target))
    return dict(n=len(predictions),
        brier=sum(p['brier'] for p in predictions)/len(predictions),
        log_loss=sum(p['log_loss'] for p in predictions)/len(predictions),
        accuracy=sum(p['correct'] for p in predictions)/len(predictions),
        contract='Multiclass Brier sums squared errors across all three outcomes; not directly comparable with v2 binary-component Brier.'),predictions


def audit_actor_trace(trace):
    """Check actual actor prompts, action likelihoods and telescoping rewards."""
    if len(trace['events'])!=len(trace['actor_events']):raise ValueError('Actor/event coverage mismatch')
    reward=0.
    for original,actor in zip(trace['events'],trace['actor_events']):
        if actor['input']!=actor_input(original['input']) or actor['input_sha256']!=digest(actor['input']):
            raise ValueError('Actual actor prompt differs from its declared contract')
        if actor['action']!=original['action'] or actor['observation']!=original['observation']:
            raise ValueError('Actor choice was not executed')
        expected=-original['cost']+({'completed':1.,'incorrect':-1.,'unfinished':0.}[trace['outcome']]
                                   if original['terminal'] else 0.)
        if abs(actor['reward']-expected)>1e-9:raise ValueError('Incorrect terminal utility or per-action cost')
        ids=actor['encoded_input']['option_ids'];p=actor['old_probabilities']
        if ids!=[o['id'] for o in actor['input']['options']] or len(ids)!=len(p):raise ValueError('Changed action menu')
        if any(not math.isfinite(v) or v<0 or v>1 for v in p) or abs(sum(p)-1)>1e-5:raise ValueError('Invalid actor probabilities')
        if abs(math.log(p[ids.index(actor['action'])])-actor['sampled_log_probability'])>1e-5:
            raise ValueError('Stored likelihood differs')
        reward+=actor['reward']
    if abs(reward-trace['reward'])>1e-9:raise ValueError('Actor return mismatch')
    return len(trace['actor_events'])
