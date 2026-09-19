"""Three-arm pilot on fresh executed shell trajectories and verified forecasts."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import signal
import time

import torch

from general_lab.outcome_train import (DetachedValuePolicy, gradient_diagnostic,
    normalize_advantages, unchanged_policy_check, update)
from general_lab.rl import DeadlineReached, _hash_trainable, parameter_audit, prepare, snapshot
from general_lab.train import macro_metrics
from scale_lab.common import MODELS, ROOT, digest, file_hash, label_token_ids, read_rows, write_json, write_rows
from scale_lab.model import evaluate as evaluate_general, load_model
from tool_lab.evidence_env import Episode, Executor, HORIZON, REGIMES

SEEDS = (1507, 1609)
RECIPE = dict(max_updates=40, batch_size=8, epochs_per_update=2, learning_rate=3e-6,
              value_learning_rate=1e-4, value_weight=.5, entropy_weight=.01,
              replay_weight=.5, cost_weight=0., clip=.2, max_kl=.02,
              max_tokens=1536, eval_every=10, patience=2, min_updates=20,
              episodes_per_update=8, forecast_groups_per_update=1, replay_rows=16,
              exploration_floor=.2)


class EvidencePolicy(DetachedValuePolicy):
    """An explicit actor distribution with exploration; forecasts stay native.

    Sampling, likelihood rescoring, entropy and PPO all use this same mixture.
    Greedy decisions are unchanged. The floor is not a success probability.
    """
    def forward(self, rows):
        logits,values,acceptable=super().forward(rows)
        transformed=[]
        for row,logits_row in zip(rows,logits):
            if row['task']=='shell_action':
                n=len(row['option_ids']);epsilon=RECIPE['exploration_floor']
                mixed=((1-epsilon)*logits_row[:n].softmax(-1)+epsilon/n).log()
                logits_row=torch.cat([mixed,logits_row[n:]])
            transformed.append(logits_row)
        return torch.stack(transformed),values,acceptable


def eligible(metrics, baseline):
    return (metrics['retention']['macro_accuracy'] >= baseline['retention']['macro_accuracy'] - .02
            and metrics['retention']['macro_log_loss'] <= baseline['retention']['macro_log_loss'] + .05
            and metrics['reward'] >= baseline['reward'] - .02
            and metrics['forecast']['brier'] <= baseline['forecast']['brier'] + .02)


def objective(metrics):
    return metrics['reward'] - .25 * metrics['forecast']['brier']


@torch.no_grad()
def collect(policy, tokenizer, cases, executors, max_tokens, check, sample):
    """Current policy samples commands, sees actual output, and acts again."""
    policy.eval()
    episodes, traces, records_by_episode = [], [], []
    for start in range(0, len(cases), len(executors)):
        chunk = cases[start:start+len(executors)]
        active = [Episode(c,e) for c,e in zip(chunk,executors)]
        per_episode = [[] for _ in active]
        for depth in range(HORIZON):
            check()
            pending = [(i,e) for i,e in enumerate(active) if not e.done]
            if not pending:break
            items = [e.input() for _,e in pending]
            rows = [prepare(tokenizer,item,f'{e.case["id"]}:{depth}',max_tokens) for item,(_,e) in zip(items,pending)]
            for row in rows:row['task']='shell_action'
            logits, values, _ = policy(rows)
            distribution = torch.distributions.Categorical(logits=logits)
            choices = distribution.sample() if sample else logits.argmax(-1)
            logp = distribution.log_prob(choices).cpu().tolist()
            probabilities = distribution.probs.cpu().tolist()
            for j,((i,e),row,index) in enumerate(zip(pending,rows,choices.cpu().tolist())):
                event = e.step(row['option_ids'][index])
                if event['input'] != items[j]:raise ValueError('Environment changed between scoring and acting')
                event.update(encoded_input=row, old_probabilities=probabilities[j][:len(row['option_ids'])],
                             sampled_log_probability=logp[j], value=float(values[j]))
                per_episode[i].append(dict(row=row, action=index, old_logp=logp[j],
                    old_probabilities=event['old_probabilities'], old_value=float(values[j]), reward=event['reward']))
        if any(not e.done for e in active):raise ValueError('Nonterminal trajectory')
        for e,items in zip(active,per_episode):
            future=0.
            for r in reversed(items):future+=r['reward'];r['return']=future
            records_by_episode.extend(items);traces.append(e.receipt())
    if sample:normalize_advantages(records_by_episode)
    return records_by_episode,traces


@torch.no_grad()
def forecast_metrics(policy, rows, batch_size, check):
    policy.eval(); predictions=[]
    for start in range(0,len(rows),batch_size):
        check();chunk=rows[start:start+batch_size]
        logits,_,_=policy(chunk)
        for row,p in zip(chunk,logits.softmax(-1).cpu().tolist()):
            predictions.append(dict(id=row['id'],group_id=row['group_id'],family=row['family'],regime=row['regime'],
                phase=row['phase'],outcome=row['outcome'],probability_yes=p[row['option_ids'].index('yes')]))
    def summarize(values):
        return dict(n=len(values), brier=sum((r['probability_yes']-r['outcome'])**2 for r in values)/len(values),
                    log_loss=sum(-math.log(max(1e-12,r['probability_yes'] if r['outcome'] else 1-r['probability_yes'])) for r in values)/len(values),
                    accuracy=sum((r['probability_yes']>=.5)==r['outcome'] for r in values)/len(values))
    result=summarize(predictions)
    result['by_phase']={s:summarize([r for r in predictions if r['phase']==s]) for s in sorted({r['phase'] for r in predictions})}
    return result,predictions


def verify_freeze(data, adapter):
    frozen=json.loads((data/'freeze.json').read_text())
    if frozen['recipe']!=RECIPE or frozen['seeds']!=list(SEEDS):raise ValueError('Changed pilot recipe')
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Changed pilot source: '+name)
    for name,sha in frozen['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Changed pilot input: '+name)
    if {p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()}!=frozen['adapter']:
        raise ValueError('Changed starting checkpoint')
    return frozen


def train(args):
    from transformers import AutoTokenizer
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    started=time.monotonic(); stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or time.monotonic()-started>args.max_hours*3600:
            raise DeadlineReached('Arm runtime limit reached')
    frozen=verify_freeze(args.data,args.adapter)
    for k,v in RECIPE.items():setattr(args,k,v)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='loading',arm=args.arm,seed=args.seed,recipe=RECIPE,model=MODELS['qwen35-9b'],
                 freeze_sha256=file_hash(args.data/'freeze.json'), starting_adapter=frozen['adapter'])
    write_json(args.output/'run.json',receipt)
    steps=number=selected=0;model=initial=policy=None;engines=[];halt=None
    try:
        torch.manual_seed(args.seed);random.seed(args.seed);torch.set_float32_matmul_precision('high')
        tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],token=False)
        model=load_model(MODELS['qwen35-9b'],args.device,args.adapter,training=True)
        policy=EvidencePolicy(model,label_token_ids(tokenizer),tokenizer.pad_token_id,args.device)
        initial=snapshot(model);receipt['initial_trainable_sha256']=_hash_trainable(initial)
        receipt['trainable_parameters']=sum(t.numel() for t in initial.values())
        engines=[Executor(args.backend) for _ in range(args.batch_size)]
        training_cases=read_rows(args.data/'train-cases.jsonl')
        by_regime={r:[c for c in training_cases if c['regime']==r] for r in REGIMES}
        for values in by_regime.values():random.Random(args.seed).shuffle(values)
        forecasts=defaultdict(list)
        for row in read_rows(args.data/'train-forecasts.jsonl'):
            forecasts[(row['group_id'],row['regime'])].append(row)
        groups=sorted(forecasts);random.Random(args.seed).shuffle(groups)
        if len(groups)<args.max_updates:raise ValueError('Not enough distinct forecast groups')
        replay=read_rows(args.data/'replay.jsonl');retention=read_rows(args.data/'retention.jsonl')
        def measure(split,tag):
            cases=read_rows(args.data/(split+'-cases.jsonl'))
            _,traces=collect(policy,tokenizer,cases,engines,args.max_tokens,check,False)
            fm,predictions=forecast_metrics(policy,read_rows(args.data/(split+'-forecasts.jsonl')),args.batch_size,check)
            retention_predictions=evaluate_general(model,retention,policy.labels,policy.pad_id,args.device,args.batch_size,64)
            retained=macro_metrics(retention_predictions)
            metrics=dict(episodes=len(traces),success_rate=sum(t['success'] for t in traces)/len(traces),
                         reward=sum(t['reward'] for t in traces)/len(traces),forecast=fm,retention=retained,
                         by_regime={r:dict(n=len(v),reward=sum(t['reward'] for t in v)/len(v),success_rate=sum(t['success'] for t in v)/len(v))
                            for r in REGIMES for v in [[t for t in traces if t['regime']==r]]})
            write_rows(args.output/(tag+'-trajectories.jsonl'),traces)
            write_rows(args.output/(tag+'-forecasts.jsonl'),predictions)
            write_rows(args.output/(tag+'-retention-predictions.jsonl'),retention_predictions)
            write_json(args.output/(tag+'-metrics.json'),metrics)
            return metrics
        if args.arm=='baseline':
            receipt['status']='baseline_test';write_json(args.output/'run.json',receipt)
            measure('test','selected-test')
            transfer_predictions=evaluate_general(model,read_rows(args.data/'transfer.jsonl'),policy.labels,policy.pad_id,args.device,args.batch_size,64)
            transfer=macro_metrics(transfer_predictions)
            write_rows(args.output/'selected-transfer-predictions.jsonl',transfer_predictions)
            write_json(args.output/'selected-transfer.json',transfer)
            receipt.update(status='complete',baseline_only=True)
            return
        receipt['status']='baseline_validation';write_json(args.output/'run.json',receipt)
        baseline=measure('validation','baseline-validation');best=objective(baseline)
        model.save_pretrained(args.output/'best')
        receipt.update(baseline=baseline,status='training');write_json(args.output/'run.json',receipt)
        # The richer environment must leave meaningful room to improve utility
        # or forecast quality before paying for gradient steps.
        oracle=frozen['reference_validation_reward']
        if oracle-baseline['reward']<.02 and baseline['forecast']['brier']<.04:
            raise ValueError('Insufficient policy/forecast headroom for this pilot')
        optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':args.learning_rate},
                                      {'params':policy.value.parameters(),'lr':args.value_learning_rate}],weight_decay=0.)
        misses=0
        with (args.output/'training.jsonl').open('w') as log,(args.output/'rollouts.jsonl').open('w') as rollout:
            for number in range(1,args.max_updates+1):
                check();group=groups[number-1];outcomes=forecasts[group]
                replay_rows=random.Random(args.seed*1000+number).sample(replay,args.replay_rows)
                records=[];traces=[]
                if args.arm!='outcome':
                    cases=[by_regime[r][(number-1)%len(by_regime[r])] for r in REGIMES]
                    records,traces=collect(policy,tokenizer,cases,engines,args.max_tokens,check,True)
                    for trace in traces:rollout.write(json.dumps(dict(update=number,**trace),allow_nan=False)+'\n')
                    rollout.flush()
                if number==1:
                    diagnostic=gradient_diagnostic(policy,records,outcomes if args.arm!='reward' else [],[],replay_rows,args,check)
                    write_json(args.output/'gradient-diagnostic.json',diagnostic)
                elif records:
                    unchanged_policy_check(policy,records,args,check)
                event=dict(update=number,forecast_group=list(group) if args.arm!='reward' else None,
                           forecast_rows=len(outcomes) if args.arm!='reward' else 0,
                           replay_ids=[r['id'] for r in replay_rows],episodes=len(traces),transitions=len(records),epochs=[])
                for epoch in range(args.epochs_per_update):
                    def committed(metrics):
                        nonlocal steps
                        steps+=1
                        with (args.output/'optimizer-steps.jsonl').open('a') as f:
                            f.write(json.dumps(dict(update=number,epoch=epoch+1,step=steps,metrics=metrics))+'\n')
                    result=update(policy,optimizer,records,outcomes,[],replay_rows,args,check,committed)
                    event['epochs'].append(result)
                    if records and result['post_step']['mean_full_kl']>args.max_kl:
                        halt='policy_divergence';break
                if number%args.eval_every==0 or number==args.max_updates or halt:
                    measured=measure('validation',f'update-{number}-validation')
                    ok=eligible(measured,baseline);improved=ok and objective(measured)>best+1e-4
                    event.update(validation=measured,eligible=ok,selected=improved)
                    if improved:
                        best=objective(measured);selected=number;misses=0;model.save_pretrained(args.output/'best')
                    else:misses+=1
                    model.save_pretrained(args.output/'latest')
                    if misses>=args.patience and number>=args.min_updates:halt=halt or 'validation_plateau'
                event.update(optimizer_steps=steps,seconds=time.monotonic()-started,
                             actual_commands=sum(e.count for e in engines),forwards=policy.forward_counts)
                log.write(json.dumps(event,allow_nan=False)+'\n');log.flush()
                print(json.dumps({'arm':args.arm,'seed':args.seed,'update':number,'steps':steps,
                                  'selected':selected,'halt':halt,'seconds':event['seconds']}),flush=True)
                if halt:break
        model.save_pretrained(args.output/'latest')
        receipt.update(status='final_evaluation',selected_update=selected,updates=number,optimizer_steps=steps,stop_reason=halt)
        write_json(args.output/'run.json',receipt)
        receipt['parameter_audit']=parameter_audit(model,initial)
        set_peft_model_state_dict(model,load_peft_weights(str(args.output/'best'),device=args.device))
        measure('test','selected-test')
        transfer_predictions=evaluate_general(model,read_rows(args.data/'transfer.jsonl'),policy.labels,policy.pad_id,args.device,args.batch_size,64)
        transfer=macro_metrics(transfer_predictions)
        write_rows(args.output/'selected-transfer-predictions.jsonl',transfer_predictions)
        write_json(args.output/'selected-transfer.json',transfer)
        receipt.update(status='complete',best_adapter_sha256=file_hash(args.output/'best/adapter_model.safetensors'))
    except BaseException as error:
        receipt.update(status='bounded_stop' if isinstance(error,DeadlineReached) else 'failed',error=type(error).__name__,detail=str(error))
        if model is not None and steps:model.save_pretrained(args.output/'interrupted')
        raise
    finally:
        receipt.update(optimizer_steps=steps,updates=number,selected_update=selected,seconds=time.monotonic()-started,
                       actual_commands=sum(e.count for e in engines),forwards=getattr(policy,'forward_counts',{}))
        write_json(args.output/'run.json',receipt)
        for e in engines:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('outcome','reward','hybrid','baseline'),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--max-hours',type=float,default=1.1)
    p.add_argument('--device',default='cuda')
    p.add_argument('--backend',choices=('catalog','docker'),default='catalog')
    train(p.parse_args())
