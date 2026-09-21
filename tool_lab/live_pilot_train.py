"""One identical-start mixed-tool arm. Executed rewards never come from a model."""
import argparse
import json
from pathlib import Path
import random
import signal
import time

import torch

from general_lab.rl import DeadlineReached, snapshot, _hash_trainable, parameter_audit
from general_lab.train import macro_metrics
from scale_lab.common import ROOT, file_hash, label_token_ids, read_rows, write_json, write_rows
from scale_lab.model import load_model, evaluate as evaluate_general
from tool_lab.expanded_metrics import macro_forecast_metrics, trajectory_metrics
from tool_lab.expanded_learning import gradient_diagnostic
from tool_lab.expanded_pool import Pool
from tool_lab.live_mixed import LivePolicy, collect_shell, collect_retail, learning_step
from tool_lab.live_pilot_plan import VERSION, RECIPE, SEEDS, ARMS, PARENT, INITIAL, safety, improvement
from tool_lab.retail_process import RetailProcess
from tool_lab.retail_live_qualify import FEES
from tool_lab.retail_evidence import key as retail_key


def verify(data, adapter):
    frozen = json.loads((data/'freeze.json').read_text())
    if (frozen['version'] != VERSION or frozen['recipe'] != RECIPE or frozen['seeds'] != list(SEEDS) or
            frozen['arms'] != list(ARMS) or frozen['release_eligible']):
        raise ValueError('Prospective diagnostic contract changed')
    for name, sha in frozen['sources'].items():
        if file_hash(ROOT/name) != sha:raise ValueError('Frozen source changed: '+name)
    for name, sha in frozen['files'].items():
        if file_hash(data/name) != sha:raise ValueError('Frozen data changed: '+name)
    if file_hash(adapter/'adapter_model.safetensors') != PARENT:
        raise ValueError('Original step-2742 adapter required')
    return frozen


class LivePool:
    def __init__(self, args):
        self.args=args;self.retail_resets=0;self.retail_calls=0;self.retail_turns=0
        self.shell=Pool(args.output,args.worker_python,args.worker_source,backend=args.backend,
            workers=2,max_seconds=int(args.max_hours*3600+120),maximum_new_episodes=1200,maximum_new_actions=9600)
    def collect(self, policy, tokenizer, cases, resets, check, sample):
        records,traces=collect_shell(self.shell,policy,tokenizer,cases,self.args.max_tokens,check,sample) if cases else ([],[])
        episodes=[]
        try:
            for reset in resets:
                check()
                if self.retail_resets>=144:raise ValueError('Retail prepared reset ceiling exceeded')
                stem=self.args.output/f'retail-{self.retail_resets:04d}'
                self.retail_resets+=1
                source=self.args.retail_source/(retail_key(reset['task'],reset['condition'],'stop',0)+'-private.json')
                episodes.append(RetailProcess(self.args.retail_python,self.args.retail_plan,source,FEES[reset['fee_index']],
                    stem.with_suffix('.jsonl'),stem.with_suffix('.log')))
            if episodes:
                found,executed=collect_retail(policy,tokenizer,episodes,self.args.max_tokens,check,sample)
                for reset,trace in zip(resets,executed,strict=True):
                    trace['reset_identity']=reset
                    self.retail_calls+=len(trace['receipt']['history']);self.retail_turns+=len(trace['actor_events'])
                records+=found;traces+=executed
        finally:
            for ep in episodes:ep.close()
        return records,traces
    def counts(self):
        return dict(**self.shell.counts(),retail_resets=self.retail_resets,retail_calls=self.retail_calls,retail_turns=self.retail_turns)
    def close(self):self.shell.close()


def train(args):
    from transformers import AutoTokenizer
    from torch.utils.tensorboard import SummaryWriter
    started=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or time.monotonic()-started>args.max_hours*3600:
            raise DeadlineReached('Bounded arm deadline')
    frozen=verify(args.data,args.adapter)
    for k,v in RECIPE.items():setattr(args,k,v)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(version=VERSION,status='loading',arm=args.arm,seed=args.seed,freeze_sha256=file_hash(args.data/'freeze.json'),
        parent_adapter_sha256=PARENT,model=frozen['model'],recipe=RECIPE,accepted_steps=0,physical_optimizer_attempts=0,
        updates=0,selected_update=0,release_eligible=False,stop_reason=None)
    def save():write_json(args.output/'run.json',receipt)
    save();model=policy=pool=writer=initial=None
    try:
        torch.manual_seed(args.seed);random.seed(args.seed);torch.set_float32_matmul_precision('high')
        tokenizer=AutoTokenizer.from_pretrained(frozen['model']['id'],revision=frozen['model']['revision'],token=False)
        model=load_model(frozen['model'],args.device,args.adapter,training=True)
        policy=LivePolicy(model,label_token_ids(tokenizer),tokenizer.pad_token_id,args.device,
                          exploration_floor=args.exploration_floor)
        initial=snapshot(model)
        if _hash_trainable(initial)!=INITIAL:raise ValueError('Loaded parent tensors differ')
        receipt['initial_trainable_sha256']=INITIAL
        pool=LivePool(args);writer=SummaryWriter(str(args.output/'tensorboard'))
        retention=read_rows(args.data/'retention.jsonl')
        validation_cases=read_rows(args.data/'validation-cases.jsonl')
        validation_forecasts=read_rows(args.data/'validation-forecasts.jsonl')
        def measure(tag):
            _,traces=pool.collect(policy,tokenizer,validation_cases,[],check,False)
            fm,predictions=macro_forecast_metrics(policy,validation_forecasts,args.batch_size,check)
            check()
            retained=evaluate_general(model,retention,policy.labels,policy.pad_id,args.device,args.batch_size,64)
            result=dict(**trajectory_metrics(traces,validation_cases),forecast=fm,retention=macro_metrics(retained))
            write_rows(args.output/(tag+'-trajectories.jsonl'),traces)
            write_rows(args.output/(tag+'-forecasts.jsonl'),predictions)
            write_rows(args.output/(tag+'-retention.jsonl'),retained)
            write_json(args.output/(tag+'-metrics.json'),result)
            for name,value in [('reward',result['reward']),('expected_brier',fm['macro']['expected_brier']),
                ('log_loss',fm['macro']['log_loss']),('general_accuracy',result['retention']['macro_accuracy']),
                ('general_log_loss',result['retention']['macro_log_loss'])]:
                writer.add_scalar('development/'+name,value,receipt['updates'])
            writer.flush();return result
        receipt['status']='baseline_development';save()
        baseline=measure('baseline');receipt['baseline']=baseline
        model.save_pretrained(args.output/'best');save()
        cases={c['id']:c for c in read_rows(args.data/'train-cases.jsonl')}
        forecasts={r['id']:r for r in read_rows(args.data/'train-forecasts.jsonl')}
        replay={r['id']:r for r in read_rows(args.data/'replay.jsonl')}
        probes=read_rows(args.data/'guard-probes.jsonl')
        optimizer=torch.optim.AdamW([
            dict(params=[p for p in model.parameters() if p.requires_grad],lr=args.learning_rate),
            dict(params=policy.value.parameters(),lr=args.value_learning_rate)],weight_decay=0.)
        misses=0;best_objective=float('-inf');receipt['status']='training';save()
        with (args.output/'training.jsonl').open('x') as log,(args.output/'rollouts.jsonl').open('x') as rollouts,\
                (args.output/'learning-ledger.jsonl').open('x') as ledger:
            for plan in read_rows(args.data/f'schedule-{args.seed}.jsonl'):
                check();number=plan['update'];receipt['updates']=number
                outcomes=[forecasts[x] for x in plan['forecast_ids']] if args.arm!='reward' else []
                replay_rows=[replay[x] for x in plan['replay_ids']]
                records=[];traces=[]
                if args.arm!='outcome':
                    records,traces=pool.collect(policy,tokenizer,[cases[x] for x in plan['case_ids']],plan['retail'],check,True)
                    for trace in traces:rollouts.write(json.dumps(dict(update=number,**trace),allow_nan=False)+'\n')
                    rollouts.flush()
                def record(event):
                    ledger.write(json.dumps(dict(update=number,**event),allow_nan=False)+'\n');ledger.flush()
                    if event['phase']=='optimizer_attempt':receipt['physical_optimizer_attempts']+=1
                if number==1:
                    # Diagnostic backwards are separately accounted and never step an optimizer.
                    selected=[]
                    for task in ('shell_action','retail_live_action'):
                        subset=[r for r in records if r['row']['task']==task]
                        if subset:selected.append(max(subset,key=lambda r:len(r['row']['input_ids'])))
                    diag=gradient_diagnostic(policy,selected,
                        sorted(outcomes,key=lambda r:len(r['input_ids']),reverse=True)[:2],replay_rows[:2],args,check,record)
                    write_json(args.output/'gradient-diagnostic.json',diag)
                step=learning_step(policy,optimizer,records,outcomes,replay_rows,probes,args,check,record)
                if step['accepted']:receipt['accepted_steps']+=1
                else:receipt['stop_reason']='rejected_'+step['reason']
                event=dict(update=number,step=step,episodes=len(traces),transitions=len(records),
                    forecast_presentations_scheduled=len(outcomes),replay_presentations_scheduled=len(replay_rows))
                if step['accepted'] and number%args.eval_every==0:
                    measured=measure(f'update-{number}')
                    safe=safety(measured,baseline);qualified=improvement(measured,baseline)
                    score=measured['reward']-.25*measured['forecast']['macro']['expected_brier']
                    selected=qualified and score>best_objective+1e-4
                    if selected:
                        model.save_pretrained(args.output/'best');receipt['selected_update']=number
                        best_objective=score;misses=0
                    else:misses+=1
                    event.update(metrics=measured,safe=safe,joint_development_improvement=qualified,selected=selected)
                    model.save_pretrained(args.output/'latest')
                    torch.save(policy.value.state_dict(),args.output/'latest-critic.pt')
                    if not safe:receipt['stop_reason']='development_safety_gate'
                    elif misses>=args.patience and number>=args.min_updates:receipt['stop_reason']='two_checks_without_joint_improvement'
                event.update(seconds=time.monotonic()-started,counts=pool.counts())
                log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();save()
                for k,v in step['metrics'].items():writer.add_scalar('training/'+k,v,number)
                writer.add_scalar('guard/accepted',int(step['accepted']),number);writer.flush()
                print(json.dumps(dict(arm=args.arm,seed=args.seed,update=number,accepted=step['accepted'],
                    selected=receipt['selected_update'],stop=receipt['stop_reason'])),flush=True)
                if receipt['stop_reason']:break
        model.save_pretrained(args.output/'latest');torch.save(policy.value.state_dict(),args.output/'latest-critic.pt')
        receipt.update(status='complete',parameter_audit=parameter_audit(model,initial),
            selected_adapter_sha256=file_hash(args.output/'best/adapter_model.safetensors'),
            latest_adapter_sha256=file_hash(args.output/'latest/adapter_model.safetensors'))
    except BaseException as error:
        receipt.update(status='bounded_stop' if isinstance(error,DeadlineReached) else 'failed',error=type(error).__name__,detail=str(error))
        if model is not None and receipt['accepted_steps']:model.save_pretrained(args.output/'interrupted')
        raise
    finally:
        if writer is not None:writer.close()
        if pool is not None:
            receipt['counts']=pool.counts();pool.close()
        receipt.update(seconds=time.monotonic()-started,forwards=getattr(policy,'forward_counts',{}));save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapter','output','worker-python','worker-source','retail-python','retail-plan','retail-source'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--max-hours',type=float,default=.8);p.add_argument('--device',default='cuda')
    p.add_argument('--backend',choices=('catalog','docker'),default='catalog')
    train(p.parse_args())
