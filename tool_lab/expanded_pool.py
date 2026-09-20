"""Fresh language-selected trajectories across old and new execution mechanisms."""
import math

import torch

from general_lab.outcome_train import normalize_advantages
from general_lab.rl import prepare
from scale_lab.common import digest,write_json
from tool_lab.mixed_runtime import Pool as ExistingPool
from tool_lab.expanded_runtime import Budget,make_episode,audit_trace


class Pool:
    def __init__(self,output,python,source,*,backend='docker',workers=4,max_seconds=14400,
                 maximum_new_episodes=1600,maximum_new_actions=12000):
        self.output=output;self.workers=workers
        self.existing=ExistingPool(output,python,source,backend=backend,workers=workers,max_seconds=max_seconds)
        self.budget=Budget(output/'expanded-attempts.jsonl',max_episodes=maximum_new_episodes,
                           max_actions=maximum_new_actions,max_seconds=max_seconds)

    @torch.no_grad()
    def collect(self,policy,tokenizer,cases,max_tokens,check,sample):
        old=[c for c in cases if c['family'] not in ('filesystem_scope','calendar','reservation')]
        new=[c for c in cases if c['family'] in ('filesystem_scope','calendar','reservation')]
        records,traces=self.existing.collect(policy,tokenizer,old,max_tokens,check,sample) if old else ([],[])
        policy.eval()
        for start in range(0,len(new),self.workers):
            episodes=[];per=[];actor=[]
            try:
                for c in new[start:start+self.workers]:episodes.append(make_episode(c,self.budget));per.append([]);actor.append([])
                for depth in range(8):
                    pending=[(i,e) for i,e in enumerate(episodes) if not e.done]
                    if not pending:break
                    check();items=[e.input() for _,e in pending]
                    rows=[prepare(tokenizer,item,digest([e.case['id'],depth,'live-policy',item]),max_tokens)
                          for item,(_,e) in zip(items,pending)]
                    for row in rows:row['task']='shell_action'
                    logits,values,_=policy(rows);dist=torch.distributions.Categorical(logits=logits)
                    choices=dist.sample() if sample else logits.argmax(-1);logp=dist.log_prob(choices).cpu().tolist()
                    probabilities=dist.probs.cpu().tolist()
                    for j,((i,ep),row,index) in enumerate(zip(pending,rows,choices.cpu().tolist())):
                        event=ep.step(row['option_ids'][index])
                        if event['input']!=items[j]:raise ValueError('Executed input changed after scoring')
                        r=dict(row=row,action=index,old_logp=logp[j],old_probabilities=probabilities[j][:len(row['option_ids'])],
                               old_value=float(values[j]),reward=event['reward'])
                        per[i].append(r);actor[i].append(dict(input=items[j],encoded_input=row,action=event['action'],
                            old_probabilities=r['old_probabilities'],sampled_log_probability=r['old_logp'],
                            value=r['old_value'],reward=event['reward'],observation=event['observation'],terminal=ep.done))
                for ep,rs,ae in zip(episodes,per,actor):
                    if not ep.done:raise ValueError('Live episode exceeded frozen horizon')
                    trace=ep.receipt();audit_trace(ep.case,trace);future=0.
                    for r in reversed(rs):future+=r['reward'];r['return']=future
                    if abs(future-trace['reward'])>1e-9:raise ValueError('Learning return differs from executed reward')
                    for event,a in zip(trace['events'],ae):
                        ids=a['encoded_input']['option_ids'];p=a['old_probabilities']
                        if abs(sum(p)-1)>1e-5 or abs(math.log(p[ids.index(event['action'])])-a['sampled_log_probability'])>1e-5:
                            raise ValueError('Sampled likelihood or action menu differs')
                    self.budget.log(dict(phase='completed_episode',attempt=ep.attempt,identity=ep.case['id'],receipt_sha256=digest(trace)))
                    trace['actor_events']=ae;records.extend(rs);traces.append(trace)
            finally:
                for ep in episodes:ep.close()
        if sample:normalize_advantages(records)
        return records,traces

    def counts(self):
        return dict(**self.existing.counts(),new_physical_episodes=self.budget.episodes,
                    new_offered_actions=self.budget.actions,**self.budget.counts)

    def close(self):
        self.existing.close();write_json(self.output/'expanded-execution-counts.json',self.counts())
