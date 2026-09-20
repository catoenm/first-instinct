"""Proper uncertain-outcome learning with guards on native and behavior policies."""
import math

import torch
from torch.nn import functional as F

from general_lab.outcome_train import DetachedValuePolicy, component_loss, gradient_vector
from general_lab.rl import clipped_policy_loss, snapshot, _hash_trainable
from puffer_lab.consequence_train import soft_loss
from scale_lab.common import digest
from scale_lab.model import loss_for
from tool_lab.guarded_update import attempt_update


class Policy(DetachedValuePolicy):
    def __init__(self, *args, exploration_floor=.2, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0<=exploration_floor<1:raise ValueError('Invalid exploration mixture')
        self.exploration_floor=exploration_floor

    def forward(self, rows):
        logits,values,acceptable=super().forward(rows)
        result=[]
        for row,score in zip(rows,logits):
            if row['task']=='shell_action':
                n=len(row['option_ids']);epsilon=self.exploration_floor
                score=torch.cat([((1-epsilon)*score[:n].softmax(-1)+epsilon/n).log(),score[n:]])
            result.append(score)
        return torch.stack(result),values,acceptable


def outcome_loss(policy, rows):
    if any(r['task']=='shell_action' for r in rows):raise ValueError('Forecasts cannot use action exploration')
    logits,_,_=policy(rows);targets=torch.zeros_like(logits);mask=torch.zeros_like(logits,dtype=torch.bool)
    for i,row in enumerate(rows):
        q=row['soft_target'];n=len(row['option_ids'])
        if len(q)!=n or row['target_indices']:raise ValueError('Outcome distributions cannot use acceptable-set targets')
        targets[i,:n]=torch.tensor(q,dtype=logits.dtype,device=logits.device);mask[i,:n]=True
    return soft_loss(logits,targets,mask).mean()


@torch.no_grad()
def forecast_metrics(policy, rows, batch_size, check):
    policy.eval();predictions=[]
    for start in range(0,len(rows),batch_size):
        check();chunk=rows[start:start+batch_size];logits,_,_=policy(chunk)
        logprobs=logits.log_softmax(-1)
        for row,logp in zip(chunk,logprobs):
            n=len(row['soft_target']);q=row['soft_target'];p=logp[:n].exp().cpu().tolist();lp=logp[:n].cpu().tolist()
            if len(q)!=len(row['option_ids']) or abs(sum(q)-1)>1e-8 or min(q)<0:raise ValueError('Invalid forecast truth')
            ce=-sum(v*x for v,x in zip(q,lp));entropy=-sum(v*math.log(v) for v in q if v)
            excess=sum((a-b)**2 for a,b in zip(p,q));noise=1-sum(v*v for v in q)
            result=dict(id=row['id'],family=row['family'],probabilities=p,soft_target=q,option_ids=row['option_ids'],
                log_loss=ce,excess_log_loss=ce-entropy,expected_brier=excess+noise,
                excess_brier=excess,irreducible_brier=noise,ambiguous=sum(v>0 for v in q)>1,
                expected_choice_accuracy=q[max(range(n),key=p.__getitem__)])
            if any(not math.isfinite(result[k]) for k in ('log_loss','expected_brier','excess_brier')):
                raise ValueError('Nonfinite probability score')
            predictions.append(result)
    if not predictions:raise ValueError('Empty forecast evaluation')
    fields=('log_loss','excess_log_loss','expected_brier','excess_brier','irreducible_brier','expected_choice_accuracy')
    def average(values):return {k:sum(v[k] for v in values)/len(values) for k in fields}
    by_family={f:dict(n=len(group),**average(group)) for f in sorted({p['family'] for p in predictions})
               for group in [[p for p in predictions if p['family']==f]]}
    result=dict(n=len(predictions),macro=average(list(by_family.values())),by_family=by_family,
        weighting='Equal unique public inputs within mechanism, then equal mechanisms; source multiplicity is not extra evidence.',
        contract='Expected Brier includes irreducible uncertainty. Excess Brier is squared distance to the verified distribution.')
    for name,predicate in [('ambiguous',lambda p:p['ambiguous']),('deterministic',lambda p:not p['ambiguous'])]:
        group=[p for p in predictions if predicate(p)]
        result[name]=dict(n=len(group),**average(group)) if group else dict(n=0)
    return result,predictions


@torch.no_grad()
def guard_reference(policy, probes, records, batch_size, check):
    """Freeze both policy distributions before the single optimizer attempt."""
    policy.eval();unique={}
    for row in probes+[r['row'] for r in records]:
        unique.setdefault(digest([row['input_ids'],row['option_ids']]),row)
    if not unique:raise ValueError('An action guard set is required in every arm')
    result=[]
    for contract in ('native','behavior'):
        rows=[{**r,'task':'native_action_guard' if contract=='native' else 'shell_action'} for r in unique.values()]
        for start in range(0,len(rows),batch_size):
            check();chunk=rows[start:start+batch_size];logits,_,_=policy(chunk)
            for row,lp in zip(chunk,logits.log_softmax(-1).cpu().tolist()):
                result.append(dict(contract=contract,row=row,old_log_probabilities=lp[:len(row['option_ids'])]))
    return result


@torch.no_grad()
def guard_measure(policy, references, batch_size, check):
    policy.eval();values={'native':[],'behavior':[]}
    for start in range(0,len(references),batch_size):
        check();chunk=references[start:start+batch_size];logits,_,_=policy([r['row'] for r in chunk])
        for ref,logp in zip(chunk,logits.log_softmax(-1)):
            old=torch.tensor(ref['old_log_probabilities'],device=logp.device,dtype=logp.dtype)
            kl=float((old.exp()*(old-logp[:len(old)])).sum())
            values[ref['contract']].append(kl)
    result={k:dict(mean=sum(v)/len(v),maximum=max(v),inputs=len(v)) for k,v in values.items()}
    if any(not math.isfinite(x) or x < -1e-6 for v in values.values() for x in v):raise ValueError('Invalid policy divergence')
    return dict(mean_full_kl=max(v['mean'] for v in result.values()),
                max_full_kl=max(v['maximum'] for v in result.values()),by_contract=result)


def component(policy, name, rows, args):
    return outcome_loss(policy,rows) if name=='outcome' else component_loss(policy,name,rows,args)


def gradient_diagnostic(policy, records, outcomes, replay, args, check, record=lambda entry:None):
    policy.train();parameters=[p for p in policy.language.parameters() if p.requires_grad]
    initial=_hash_trainable(snapshot(policy.language));vectors={};result=dict(components={},cosines={},optimizer_steps=0)
    for name,rows in [('actor',records),('value',records),('outcome',outcomes),('replay',replay)]:
        if not rows:continue
        policy.zero_grad(set_to_none=True);total=0.
        for start in range(0,len(rows),args.batch_size):
            check();chunk=rows[start:start+args.batch_size]
            ids=[(r['row'] if name in ('actor','value') else r)['id'] for r in chunk]
            entry=dict(component=name,ids=ids,batch_start=start)
            record(dict(phase='started_diagnostic_backward',**entry))
            loss=component(policy,name,chunk,args)
            scale=len(chunk)/len(rows);(loss*scale).backward();total+=float(loss.detach())*scale
            record(dict(phase='completed_diagnostic_backward',**entry))
        vector=gradient_vector(parameters);critic=gradient_vector(list(policy.value.parameters()))
        if not torch.isfinite(vector).all() or not torch.isfinite(critic).all():raise ValueError('Nonfinite component gradients')
        if name=='value' and torch.count_nonzero(vector):raise ValueError('Critic changed the language gradient')
        if name in ('actor','outcome') and not torch.count_nonzero(vector):raise ValueError('Objective has no language gradient')
        result['components'][name]=dict(rows=len(rows),loss=total,language_l2=float(vector.norm()),critic_l2=float(critic.norm()))
        vectors[name]=vector
    for a in vectors:
        for b in vectors:
            if a>=b:continue
            denominator=float(vectors[a].norm()*vectors[b].norm())
            result['cosines'][a+':'+b]=float(torch.dot(vectors[a],vectors[b]))/denominator if denominator else None
    policy.zero_grad(set_to_none=True)
    if initial!=_hash_trainable(snapshot(policy.language)):raise ValueError('Diagnostic changed language weights')
    result.update(unchanged_parameters=True,critic_backbone_connection='detached');return result


def learning_step(policy, optimizer, records, outcomes, replay, probes, args, check, record):
    references=guard_reference(policy,probes,records,args.batch_size,check)
    def apply(committed):
        policy.train();optimizer.zero_grad(set_to_none=True);metrics={};components=[]
        if args.arm in ('reward','hybrid'):
            if not records:raise ValueError('Reward learning needs fresh executed trajectories')
            components.append(('policy',records,1.))
        if args.arm in ('outcome','hybrid'):
            if not outcomes:raise ValueError('Forecast learning needs executed consequence targets')
            components.append(('outcome',outcomes,args.forecast_weight))
        if not replay:raise ValueError('General replay required')
        components.append(('replay',replay,args.replay_weight))
        for name,rows,weight in components:
            total=0.
            for start in range(0,len(rows),args.batch_size):
                check();chunk=rows[start:start+args.batch_size]
                ids=[(r['row'] if name=='policy' else r)['id'] for r in chunk]
                entry=dict(component=name,ids=ids,batch_start=start,weight=weight)
                record(dict(phase='started_backward',**entry))
                if name=='policy':
                    logits,values,_=policy([r['row'] for r in chunk]);dist=torch.distributions.Categorical(logits=logits)
                    tensor=lambda v:torch.tensor(v,device=policy.device)
                    actor=clipped_policy_loss(dist.log_prob(tensor([r['action'] for r in chunk])),
                        tensor([r['old_logp'] for r in chunk]),tensor([r['advantage'] for r in chunk]),args.clip)[0]
                    loss=actor+args.value_weight*F.mse_loss(values,tensor([r['return'] for r in chunk]))-args.entropy_weight*dist.entropy().mean()
                else:loss=component(policy,name,chunk,args)
                if not torch.isfinite(loss):raise ValueError('Nonfinite objective '+name)
                scale=len(chunk)/len(rows);(loss*weight*scale).backward();total+=float(loss.detach())*scale
                record(dict(phase='completed_backward',**entry))
            metrics[name+'_loss']=total
        metrics['language_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(
            [p for p in policy.language.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        metrics['critic_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(policy.value.parameters(),1.,error_if_nonfinite=True))
        check();optimizer.step();committed(metrics);return metrics
    return attempt_update(policy,optimizer,apply,
        lambda:guard_measure(policy,references,args.batch_size,check),max_mean_kl=args.max_kl,
        max_individual_kl=args.max_individual_kl,record=record)
