"""One guarded learning step with explicit, actual backward-presentation receipts."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from general_lab.outcome_train import component_loss, rollout_kl
from general_lab.rl import clipped_policy_loss
from tool_lab.guarded_update import attempt_update


@torch.no_grad()
def probe_reference(policy, rows, batch_size, check):
    policy.eval();result=[]
    for start in range(0,len(rows),batch_size):
        check();chunk=rows[start:start+batch_size];logits,_,_=policy(chunk)
        for row,p in zip(chunk,logits.softmax(-1).cpu().tolist()):
            p=p[:len(row['option_ids'])]
            result.append(dict(row=row,old_probabilities=p,action=0,old_logp=math.log(p[0])))
    if not result:raise ValueError('An explicit action-distribution guard set is required')
    return result


def learning_step(policy,optimizer,records,outcomes,replay,probes,args,check,record):
    """No retry: a rejected attempt restores weights, optimizer and random state.

    completed_backward entries are physical objective presentations. A later
    optimizer rejection does not erase them. An interrupted started batch is
    explicitly partial, rather than reported as fully consumed.
    """
    references=probe_reference(policy,probes,args.batch_size,check)
    guard=references+records
    def apply(committed):
        policy.train();optimizer.zero_grad(set_to_none=True);metrics={}
        components=[]
        if args.arm in ('reward','hybrid'):
            if not records:raise ValueError('Reward learning requires executed policy transitions')
            components.append(('policy',records,1.))
        if args.arm in ('outcome','hybrid'):
            if not outcomes:raise ValueError('Forecast learning requires executed outcomes')
            components.append(('outcome',outcomes,1.))
        if not replay:raise ValueError('General replay is required in every arm')
        components.append(('replay',replay,args.replay_weight))
        for component,rows,weight in components:
            total=0.
            for start in range(0,len(rows),args.batch_size):
                check();chunk=rows[start:start+args.batch_size]
                ids=[(r['row'] if component=='policy' else r)['id'] for r in chunk]
                receipt=dict(component=component,ids=ids,batch_start=start)
                record(dict(phase='started_backward',**receipt))
                if component=='policy':
                    logits,values,_=policy([r['row'] for r in chunk])
                    distribution=torch.distributions.Categorical(logits=logits)
                    tensor=lambda v:torch.tensor(v,device=policy.device)
                    new=distribution.log_prob(tensor([r['action'] for r in chunk]))
                    actor=clipped_policy_loss(new,tensor([r['old_logp'] for r in chunk]),
                                             tensor([r['advantage'] for r in chunk]),args.clip)[0]
                    value=F.mse_loss(values,tensor([r['return'] for r in chunk]))
                    loss=actor+args.value_weight*value-args.entropy_weight*distribution.entropy().mean()
                else:loss=component_loss(policy,component,chunk,args)
                if not torch.isfinite(loss):raise ValueError('Nonfinite '+component+' loss')
                scale=len(chunk)/len(rows);(loss*weight*scale).backward()
                total+=float(loss.detach())*scale
                record(dict(phase='completed_backward',**receipt))
            metrics[component+'_loss']=total
        metrics['language_gradient_norm']=float(nn.utils.clip_grad_norm_(
            [p for p in policy.language.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        metrics['critic_gradient_norm']=float(nn.utils.clip_grad_norm_(policy.value.parameters(),1.,error_if_nonfinite=True))
        check();optimizer.step();committed(metrics)
        return metrics
    return attempt_update(policy,optimizer,apply,
        lambda:rollout_kl(policy,guard,args,check),max_mean_kl=args.max_kl,
        max_individual_kl=args.max_individual_kl,record=record)
