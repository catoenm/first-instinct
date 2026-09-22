"""Coverage-balanced follow-up; same original parent and release acceptance gates."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import signal
import time

from scale_lab.common import ROOT, file_hash, read_rows, write_json, write_rows
from .balanced_plan import RECIPES,PARENT,INITIAL
from .pilot_metrics import summarize,gates,better


def learning_progress(metrics,baseline,config,best_loss,stale,step):
    """Stopping for the longer recipe is distinct from release qualification."""
    checks=gates(metrics,baseline)
    loss=metrics['tools']['macro']['log_loss']
    improved=loss<best_loss-config['progress_log_loss_delta']
    best_loss,stale=(loss,0) if improved else (best_loss,stale+1)
    reason=None
    if not checks['retention'] or not checks['product_slices']:reason='retention_breach'
    elif step>=config['min_steps'] and stale>=config['patience']:reason='tool_log_loss_plateau_after_minimum_dose'
    return dict(best_loss=best_loss,stale=stale,improved=improved,stop_reason=reason)


def verify(data,adapter):
    frozen=json.loads((data/'freeze.json').read_text())
    config=RECIPES.get(frozen.get('version'))
    if config is None or frozen['config']!=config or frozen['parent_adapter_sha256']!=PARENT:
        raise ValueError('Changed pilot specification')
    if file_hash(adapter/'adapter_model.safetensors')!=PARENT:
        raise ValueError('Original supervised parent required')
    for name,sha in frozen['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Changed data: '+name)
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Changed code: '+name)
    rows=read_rows(data/'train.jsonl'); development=read_rows(data/'development.jsonl')
    steps=json.loads((data/'schedule.json').read_text()); index={r['id']:r for r in rows}
    if len(index)!=len(rows) or len(steps)!=config['max_steps']:
        raise ValueError('Invalid schedule identity')
    counts=Counter(x for step in steps for x in step)
    for step in steps:
        if Counter(index[x]['learning_pool'] for x in step)!=Counter(config['per_step']):
            raise ValueError('Changed update weighting')
    for r in rows+development:
        if not 0<len(r['input_ids'])<=config['max_tokens']:
            raise ValueError('Unsupported context')
    for r in rows:
        ceiling=config.get('max_'+r['learning_pool'].rstrip('s')+'_visits',1)
        if r['role']!='train' or counts[r['id']]>ceiling:
            raise ValueError('Role or repetition breach')
    if any(r['role']!='development' for r in development) or set(index)&{r['id'] for r in development}:
        raise ValueError('Evaluation overlap or role breach')
    return frozen,index,development,steps


def train(data,adapter,output,resume=None,hard_stop=None):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from scale_lab.model import batch,load_model,score
    from .objectives import target_tensors,mixed_decision_loss
    from .pilot_state import checkpoint,restore,parameter_hash
    frozen,index,development,steps=verify(data,adapter)
    config=frozen['config']
    freeze_sha=file_hash(data/'freeze.json')
    if resume is None:
        output.mkdir(parents=True,exist_ok=False)
        receipt=dict(status='loading',started_at=time.time(),completed_steps=0,freeze_sha256=freeze_sha,
                     parent_adapter_sha256=PARENT,model=frozen['model'],config=config,training_presentations=0)
    else:
        receipt=json.loads((output/'run.json').read_text())
        if receipt['freeze_sha256']!=freeze_sha:raise ValueError('Resume freeze changed')
    deadline=receipt['started_at']+config['max_seconds']
    if config.get('final_reserve_seconds'):
        if hard_stop is None or not math.isfinite(hard_stop) or not 0<hard_stop-time.time()<=36*3600:
            raise ValueError('A bounded independent provider deadline is required')
        if resume and receipt.get('hard_stop_at')!=hard_stop:raise ValueError('Restart provider deadline changed')
        receipt['hard_stop_at']=hard_stop
        deadline=min(deadline,hard_stop-1800)
    def save(**kw):
        receipt.update(kw)
        temporary=output/'run.json.tmp';write_json(temporary,receipt);temporary.replace(output/'run.json')
        print(json.dumps(kw,allow_nan=False),flush=True)
    def guard():
        if time.time()>deadline:
            raise TimeoutError('Pilot training deadline')
    def stop(*_):raise InterruptedError('Pilot termination requested')
    signal.signal(signal.SIGTERM,stop)
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    writer=None
    try:
        save(status='loading')
        model=load_model(frozen['model'],'cuda',resume/'adapter' if resume else adapter,training=True)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=config['weight_decay'])
        labels,pad=frozen['label_token_ids'],frozen['pad_id']
        def forward(rows,padding=64):
            inputs,label_tensor,mask,_=batch(rows,labels,pad,'cuda',padding)
            return score(model,inputs,label_tensor,mask)
        def losses(rows):
            logits=forward(rows)
            return mixed_decision_loss(logits,*target_tensors(rows,logits.shape[1],'cuda'))
        suites=defaultdict(list)
        for r in development:suites[r['suite']].append(r)
        writer=SummaryWriter(str(output/'events'),purge_step=receipt['completed_steps']+1 if resume else None)
        @torch.no_grad()
        def evaluate(step):
            model.eval();result={};started=time.time()
            for name,rs in sorted(suites.items()):
                rs=sorted(rs,key=lambda r:(len(r['input_ids']),r['id']))
                predictions=[]
                for i in range(0,len(rs),config['eval_batch']):
                    guard();chunk=rs[i:i+config['eval_batch']]
                    p=forward(chunk).softmax(-1).cpu().tolist()
                    predictions.extend([v[:len(r['option_ids'])] for r,v in zip(chunk,p,strict=True)])
                write_rows(output/f'{step}-{name}-predictions.jsonl',
                           [dict(id=r['id'],probabilities=p) for r,p in zip(rs,predictions,strict=True)])
                result[name]=summarize(rs,predictions)
                for key,value in result[name]['macro'].items():writer.add_scalar(name+'/'+key,value,step)
                print(json.dumps(dict(evaluation_step=step,suite=name,macro=result[name]['macro'],n=len(rs))),flush=True)
            write_json(output/f'{step}-metrics.json',result);model.train()
            writer.add_scalar('runtime/evaluation_seconds',time.time()-started,step);writer.flush()
            return result
        if resume:
            latest=json.loads((output/'latest.json').read_text())
            if resume.resolve()!=(output/latest['path']).resolve() or file_hash(resume/'state.pt')!=latest['sha256']:
                raise ValueError('Restart checkpoint digest differs')
            state=restore(model,optimizer,resume)
            if state['freeze_sha256']!=freeze_sha or state['completed_steps']!=receipt['completed_steps']:
                raise ValueError('Restart cursor differs')
            baseline,best,best_step,stale=state['baseline'],state['best'],state['best_step'],state['stale']
            best_loss=state.get('best_loss',best['tools']['macro']['log_loss'])
            write_json(output/'restart-qualification.json',dict(status='passed',separate_process=True,
                completed_steps=state['completed_steps'],restored_trainable=state['trainable'],
                optimizer_states=len(optimizer.state),schedule_cursor=state['completed_steps'],
                optimizer_state_restored=True,rng_state_restored=True))
            save(status='training',restart_verified=True)
        else:
            initial=parameter_hash(model)
            if initial['sha256']!=INITIAL:raise ValueError('Starting trainable tensor identity differs')
            save(initial_trainable=initial)
            # Test actual long contexts and padding in forward AND backward before updates.
            ordered=sorted(index.values(),key=lambda r:len(r['input_ids']))
            probes=[ordered[0],ordered[-1]]
            model.eval()
            with torch.no_grad():
                individual=[forward([r],1).softmax(-1)[0,:len(r['option_ids'])] for r in probes]
                together=forward(probes).softmax(-1)
                difference=max((p-together[i,:len(p)]).abs().max().item() for i,p in enumerate(individual))
                choices_equal=all(p.argmax().item()==together[i,:len(p)].argmax().item() for i,p in enumerate(individual))
            if difference>.025 or not choices_equal:raise ValueError('Padding qualification failed: '+str(difference))
            model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            diagnostic=losses(probes).mean();diagnostic.backward()
            if not torch.isfinite(diagnostic) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
                raise ValueError('Nonfinite diagnostic loss/gradient')
            if not any(p.grad is not None and p.grad.abs().max()>0 for p in trainable):raise ValueError('No adapter gradient')
            optimizer.zero_grad(set_to_none=True)
            if parameter_hash(model)!=initial:raise ValueError('Diagnostic mutated weights')
            write_json(output/'runtime-qualification.json',dict(status='passed',probe_lengths=[len(r['input_ids']) for r in probes],
                max_probability_difference=difference,choices_equal=choices_equal,backward_presentations=2,optimizer_steps=0,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name()))
            baseline=evaluate(0);best=baseline;best_step=0;stale=0
            best_loss=baseline['tools']['macro']['log_loss']
            write_json(output/'selected.json',dict(step=0,source='original_parent',qualifies=False))
            save(status='training',baseline={k:v['macro'] for k,v in baseline.items()},best_step=0)
        first=receipt['completed_steps']+1
        total_per_step=sum(config['per_step'].values())
        evaluated_step=receipt.get('last_evaluated_step',0)
        def store_checkpoint(step):
            path=output/f'checkpoint-{step:04d}'
            checkpoint(model,optimizer,path,dict(completed_steps=step,freeze_sha256=freeze_sha,
                baseline=baseline,best=best,best_step=best_step,stale=stale,best_loss=best_loss))
            write_json(output/'latest.json',dict(path=path.name,step=step,sha256=file_hash(path/'state.pt')))
            if best_step==step:
                write_json(output/'selected.json',dict(step=step,source=str(path.name+'/adapter'),
                    qualifies=gates(best,baseline)['qualifies'],adapter_sha256=file_hash(path/'adapter/adapter_model.safetensors')))
        for step in range(first,config['max_steps']+1):
            remaining=deadline-time.time()
            reserve=config.get('final_reserve_seconds',0)
            if reserve and remaining<reserve+max(180.,2*receipt.get('step_seconds',90.)):
                save(status='stopped',stop_reason='reserved_final_evaluation');break
            guard();started=time.time();model.train();optimizer.zero_grad(set_to_none=True)
            rate=config['learning_rate']*min(1.,step/max(1,config.get('warmup_steps',0)))
            for group in optimizer.param_groups:group['lr']=rate
            scheduled=sorted((index[x] for x in steps[step-1]),key=lambda r:(len(r['input_ids']),r['id']))
            total_loss=0.
            with (output/'consumption.jsonl').open('a') as consumed:
                for i in range(0,len(scheduled),config['micro_batch']):
                    guard();chunk=scheduled[i:i+config['micro_batch']]
                    loss=losses(chunk).sum()/total_per_step
                    if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
                    loss.backward();total_loss+=loss.item()
                    for r in chunk:
                        consumed.write(json.dumps(dict(step=step,id=r['id'],pool=r['learning_pool'],tokens=len(r['input_ids'])))+'\n')
                    consumed.flush()
            norm=torch.nn.utils.clip_grad_norm_(trainable,config['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            seconds=time.time()-started
            save(status='training',completed_steps=step,training_presentations=step*total_per_step,
                 last_loss=total_loss,step_seconds=seconds,gradient_norm=float(norm),best_step=best_step,learning_rate=rate)
            for k,v in {'loss':total_loss,'step_seconds':seconds,'gradient_norm':float(norm)}.items():writer.add_scalar('train/'+k,v,step)
            stop_reason=None;metrics=None
            if step%config['eval_every']==0 or step==config['max_steps']:
                metrics=evaluate(step);checks=gates(metrics,baseline)
                improved=checks['eligible'] and better(metrics,best)
                if improved:best,best_step=metrics,step
                write_json(output/f'{step}-gates.json',checks)
                if 'min_steps' in config:
                    progress=learning_progress(metrics,baseline,config,best_loss,stale,step)
                    best_loss,stale=progress['best_loss'],progress['stale'];stop_reason=progress['stop_reason']
                    write_json(output/f'{step}-progress.json',progress)
                else:
                    stale=0 if improved else stale+1
                    if not checks['retention'] or not checks['product_slices']:stop_reason='retention_breach'
                    elif stale>=config['patience']:stop_reason='two_checks_without_eligible_improvement'
                evaluated_step=step
                save(best_step=best_step,last_gates=checks,stale=stale,last_evaluated_step=step)
            if step==config['restart_after_step'] or metrics is not None:
                store_checkpoint(step)
                if step==config['restart_after_step'] and resume is None:
                    save(status='restart_pending');return 75
            if stop_reason:
                save(status='stopped',stop_reason=stop_reason);break
        else:save(status='completed',stop_reason='scheduled_limit')
        # A time boundary may fall between periodic checks. Measure and preserve
        # the actual final weights, rather than reporting the earlier checkpoint.
        final_step=receipt['completed_steps']
        if config.get('final_reserve_seconds') and final_step>evaluated_step:
            metrics=evaluate(final_step);checks=gates(metrics,baseline)
            if checks['eligible'] and better(metrics,best):best,best_step=metrics,final_step
            write_json(output/f'{final_step}-gates.json',checks)
            if not (output/f'checkpoint-{final_step:04d}').exists():store_checkpoint(final_step)
            save(best_step=best_step,last_gates=checks,last_evaluated_step=final_step)
        consumed_ids=[x for s in steps[:receipt['completed_steps']] for x in s]
        final=dict(status=receipt['status'],completed_steps=receipt['completed_steps'],selected_step=best_step,
                   gates=gates(best,baseline),unique_consumed_questions=len(set(consumed_ids)),
                   consumed_presentations=len(consumed_ids),consumed_tokens=sum(len(index[x]['input_ids']) for x in consumed_ids),
                   by_pool=dict(Counter(index[x]['learning_pool'] for x in consumed_ids)),
                   freeze_sha256=freeze_sha,parent_adapter_sha256=PARENT)
        write_json(output/'result.json',final);save(finished_at=time.time(),result=final)
        return 0
    except BaseException as error:
        save(status='failed',error_type=type(error).__name__,error=str(error),finished_at=time.time())
        raise
    finally:
        if writer:writer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--hard-stop-epoch',type=float)
    a=p.parse_args();raise SystemExit(train(a.data,a.adapter,a.output,a.resume,a.hard_stop_epoch))
