"""Bounded five-mechanism learning with a reserved calendar transfer test."""
import argparse
import json
from pathlib import Path
import random
import signal
import time

import torch

from general_lab.outcome_train import unchanged_policy_check
from general_lab.rl import DeadlineReached, snapshot, _hash_trainable, parameter_audit
from general_lab.train import macro_metrics
from scale_lab.common import ROOT, MODELS, file_hash, label_token_ids, read_rows, write_json, write_rows
from scale_lab.model import load_model, evaluate as evaluate_general
from tool_lab.expanded_metrics import macro_forecast_metrics as forecast_metrics, trajectory_metrics
from tool_lab.expanded_learning import Policy, gradient_diagnostic
from tool_lab.expanded_curriculum import VERSION, RECIPE, SEEDS, eligible, objective
from tool_lab.expanded_pool import Pool
from tool_lab.expanded_learning import learning_step

ORIGINAL_ADAPTER_SHA256='882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'


def verify_freeze(data,adapter):
    frozen=json.loads((data/'freeze.json').read_text())
    if frozen['version']!=VERSION or frozen['recipe']!=RECIPE or frozen['seeds']!=list(SEEDS):
        raise ValueError('Prospective recipe changed')
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Frozen source changed: '+name)
    for name,sha in frozen['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Frozen data changed: '+name)
    actual={p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()}
    if actual!=frozen['adapter'] or actual['adapter_model.safetensors']!=ORIGINAL_ADAPTER_SHA256:
        raise ValueError('Not the original supervised step-2742 start')
    if MODELS['qwen35-9b']!=frozen['model']:raise ValueError('Foundation revision changed')
    return frozen


def train(args):
    from transformers import AutoTokenizer
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    started=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or time.monotonic()-started>args.max_hours*3600:raise DeadlineReached('Arm runtime limit reached')
    frozen=verify_freeze(args.data,args.adapter)
    for key,value in RECIPE.items():setattr(args,key,value)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='loading',version=VERSION,arm=args.arm,seed=args.seed,recipe=RECIPE,
        model=frozen['model'],freeze_sha256=file_hash(args.data/'freeze.json'),
        starting_adapter=frozen['adapter'],accepted_steps=0,physical_optimizer_attempts=0,
        selected_update=0,updates=0,stop_reason=None)
    def save():write_json(args.output/'run.json',receipt)
    save();model=policy=initial=pool=writer=None;best=None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(args.output/'tensorboard'))
        torch.manual_seed(args.seed);random.seed(args.seed);torch.set_float32_matmul_precision('high')
        tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],token=False)
        model=load_model(MODELS['qwen35-9b'],args.device,args.adapter,training=True)
        policy=Policy(model,label_token_ids(tokenizer),tokenizer.pad_token_id,args.device,exploration_floor=RECIPE["exploration_floor"])
        initial=snapshot(model);receipt['initial_trainable_sha256']=_hash_trainable(initial)
        if receipt['initial_trainable_sha256']!=frozen['initial_trainable_sha256']:
            raise ValueError('Loaded language tensors differ from identical-start contract')
        pool=Pool(args.output,args.worker_python,args.worker_source,backend=args.backend,
                  workers=args.batch_size,max_seconds=min(14400,int(args.max_hours*3600+180)))
        retention=read_rows(args.data/'retention.jsonl')
        def measure(split,tag):
            cases_for_measurement=read_rows(args.data/(split+'-cases.jsonl'))
            _,traces=pool.collect(policy,tokenizer,cases_for_measurement,args.max_tokens,check,False)
            fm,predictions=forecast_metrics(policy,read_rows(args.data/(split+'-forecasts.jsonl')),args.batch_size,check)
            check();retained_predictions=evaluate_general(model,retention,policy.labels,policy.pad_id,args.device,args.batch_size,64)
            result=dict(**trajectory_metrics(traces,cases_for_measurement),forecast=fm,
                        retention=macro_metrics(retained_predictions))
            write_rows(args.output/(tag+'-trajectories.jsonl'),traces)
            write_rows(args.output/(tag+'-forecasts.jsonl'),predictions)
            write_rows(args.output/(tag+'-retention-predictions.jsonl'),retained_predictions)
            write_json(args.output/(tag+'-metrics.json'),result)
            prefix='transfer' if split=='transfer' else 'validation'
            for name,value in [('reward',result['reward']),('success_rate',result['success_rate']),
                    ('forecast_brier',fm['macro']['expected_brier']),('forecast_log_loss',fm['macro']['log_loss']),
                    ('general_accuracy',result['retention']['macro_accuracy']),
                    ('general_log_loss',result['retention']['macro_log_loss'])]:
                writer.add_scalar(prefix+'/'+name,value,receipt['updates'])
            writer.flush()
            return result
        if args.arm=='baseline':
            receipt['status']='original_test';save();measure('transfer','selected-test')
        else:
            cases={c['id']:c for c in read_rows(args.data/'train-cases.jsonl')}
            forecasts={r['id']:r for r in read_rows(args.data/'train-forecasts.jsonl')}
            replay={r['id']:r for r in read_rows(args.data/'replay.jsonl')}
            probes=read_rows(args.data/'guard-probes.jsonl')
            schedule=read_rows(args.data/f'schedule-{args.seed}.jsonl')
            receipt['status']='baseline_validation';save()
            baseline=measure('validation','baseline-validation');best=objective(baseline)
            receipt.update(baseline=baseline,status='training');model.save_pretrained(args.output/'best');save()
            if frozen['reference_validation_reward']-baseline['reward']<.02 and baseline['forecast']['macro']['excess_brier']<.01:
                raise ValueError('Insufficient declared control/forecast headroom')
            optimizer=torch.optim.AdamW([
                dict(params=[p for p in model.parameters() if p.requires_grad],lr=args.learning_rate),
                dict(params=policy.value.parameters(),lr=args.value_learning_rate)],weight_decay=0.)
            misses=0
            with (args.output/'training.jsonl').open('x') as log,(args.output/'rollouts.jsonl').open('x') as rollouts, \
                 (args.output/'learning-ledger.jsonl').open('x') as ledger:
                for plan in schedule:
                    check();number=plan['update'];receipt['updates']=number
                    outcomes=[forecasts[i] for i in plan['forecast_ids']] if args.arm!='reward' else []
                    replay_rows=[replay[i] for i in plan['replay_ids']]
                    records=[];traces=[]
                    if args.arm!='outcome':
                        records,traces=pool.collect(policy,tokenizer,[cases[i] for i in plan['case_ids']],
                            args.max_tokens,check,True)
                        for trace in traces:rollouts.write(json.dumps(dict(update=number,**trace),allow_nan=False)+'\n')
                        rollouts.flush()
                    def record(event):
                        ledger.write(json.dumps(dict(update=number,**event),allow_nan=False)+'\n');ledger.flush()
                        if event['phase']=='optimizer_attempt':receipt['physical_optimizer_attempts']+=1
                    unchanged=unchanged_policy_check(policy,records,args,check) if records else None
                    if number==1:
                        diagnostic=gradient_diagnostic(policy,records,outcomes,replay_rows,args,check,record)
                        diagnostic['unchanged_policy']=unchanged
                        write_json(args.output/'gradient-diagnostic.json',diagnostic)
                    step=learning_step(policy,optimizer,records,outcomes,replay_rows,probes,args,check,record)
                    if step['accepted']:receipt['accepted_steps']+=1
                    else:receipt['stop_reason']='rejected_'+step['reason']
                    event=dict(update=number,step=step,episodes=len(traces),transitions=len(records),
                               scheduled_forecast_rows=len(plan['forecast_ids']),scheduled_replay_rows=len(replay_rows))
                    # A rejected transaction may never be considered for selection.
                    if step['accepted'] and (number%args.eval_every==0 or number==args.max_updates):
                        measured=measure('validation',f'update-{number}-validation')
                        ok=eligible(measured,baseline);improved=ok and objective(measured)>best+1e-4
                        event.update(validation=measured,eligible=ok,selected=improved)
                        if improved:
                            best=objective(measured);receipt['selected_update']=number;misses=0
                            model.save_pretrained(args.output/'best')
                        else:misses+=1
                        model.save_pretrained(args.output/'latest')
                        if not ok:receipt['stop_reason']='validation_safety_gate'
                        elif misses>=args.patience and number>=args.min_updates:receipt['stop_reason']='validation_plateau'
                    event.update(seconds=time.monotonic()-started,**pool.counts(),forwards=policy.forward_counts)
                    log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();save()
                    for key,value in step['metrics'].items():writer.add_scalar('training/'+key,value,number)
                    writer.add_scalar('guard/accepted',int(step['accepted']),number)
                    writer.add_scalar('guard/mean_divergence',step['diagnostic']['mean_full_kl'],number)
                    writer.add_scalar('guard/maximum_divergence',step['diagnostic']['max_full_kl'],number)
                    writer.flush()
                    print(json.dumps(dict(arm=args.arm,seed=args.seed,update=number,
                        accepted=step['accepted'],selected=receipt['selected_update'],halt=receipt['stop_reason'])),flush=True)
                    if receipt['stop_reason']:break
            model.save_pretrained(args.output/'latest')
            receipt['latest_parameter_audit']=parameter_audit(model,initial)
            receipt['status']='selected_final_test';save()
            set_peft_model_state_dict(model,load_peft_weights(str(args.output/'best'),device=args.device))
            receipt['selected_parameter_audit']=parameter_audit(model,initial)
            receipt['selected_trainable_sha256']=_hash_trainable(snapshot(model))
            measure('transfer','selected-test')
            receipt['best_adapter_sha256']=file_hash(args.output/'best/adapter_model.safetensors')
        check();transfer_predictions=evaluate_general(model,read_rows(args.data/'transfer.jsonl'),
            policy.labels,policy.pad_id,args.device,args.batch_size,64)
        write_rows(args.output/'selected-general-transfer-predictions.jsonl',transfer_predictions)
        write_json(args.output/'selected-general-transfer.json',macro_metrics(transfer_predictions))
        receipt['status']='complete'
    except BaseException as error:
        receipt.update(status='bounded_stop' if isinstance(error,DeadlineReached) else 'failed',
                       error=type(error).__name__,detail=str(error))
        if model is not None and receipt['accepted_steps']:model.save_pretrained(args.output/'interrupted')
        raise
    finally:
        if writer is not None:writer.close()
        if pool is not None:
            receipt.update(pool.counts());pool.close()
        receipt.update(seconds=time.monotonic()-started,forwards=getattr(policy,'forward_counts',{}));save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapter','output','worker-python','worker-source'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('outcome','reward','hybrid','baseline'),required=True)
    parser.add_argument('--seed',choices=SEEDS,type=int,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--backend',choices=('catalog','docker'),default='catalog')
    parser.add_argument('--max-hours',type=float,default=.7)
    train(parser.parse_args())
