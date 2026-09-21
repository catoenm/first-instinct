"""Prospective history pilot: supervised targets, reference regularization, rollback."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import signal
import time

from scale_lab.common import ROOT, file_hash, read_rows, write_json, write_rows
from .history_plan import CONFIG, PARENT, INITIAL
from .pilot_metrics import summarize, gates as original_gates, better


def gates(metrics, baseline):
    result=original_gates(metrics,baseline)
    a,b=metrics['history']['macro'],baseline['history']['macro']
    good=a['accuracy']>=b['accuracy']-CONFIG['history_accuracy_drop'] and a['log_loss']<=b['log_loss']+CONFIG['history_log_loss_increase']
    result.update(history_retention=good,eligible=result['eligible'] and good,qualifies=result['qualifies'] and good)
    return result


def divergence(before, after):
    values=[]
    for p,q in zip(before,after,strict=True):
        if len(p)!=len(q):raise ValueError('Probe menu changed')
        values.append(math.fsum(a*(math.log(max(a,1e-30))-math.log(max(b,1e-30))) for a,b in zip(p,q) if a>0))
    return dict(mean_full_kl=sum(values)/len(values),max_full_kl=max(values))


def verify(data,adapter):
    freeze=json.loads((data/'freeze.json').read_text())
    if freeze['version']!='history-pilot-v1' or freeze['config']!=CONFIG or freeze['parent_adapter_sha256']!=PARENT:
        raise ValueError('Changed prospective recipe')
    if file_hash(adapter/'adapter_model.safetensors')!=PARENT:raise ValueError('Original supervised parent required')
    for name,sha in freeze['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Changed frozen data: '+name)
    for name,sha in freeze['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Changed frozen source: '+name)
    rows=read_rows(data/'train.jsonl');dev=read_rows(data/'development.jsonl');steps=json.loads((data/'schedule.json').read_text())
    index={r['id']:r for r in rows};counts=Counter(x for step in steps for x in step)
    if len(index)!=len(rows) or len(steps)!=CONFIG['max_steps'] or set(counts)!=set(index):raise ValueError('Schedule coverage differs')
    requests=Counter();history=Counter()
    for step in steps:
        if Counter(index[x]['learning_pool'] for x in step)!=Counter(CONFIG['per_step']):raise ValueError('Update mixture differs')
    for r in rows:
        cap=CONFIG['max_verified_visits'] if r['learning_pool']=='verified' else 1
        if r['role']!='train' or counts[r['id']]>cap:raise ValueError('Role/repetition violation')
        if r['learning_pool'] in ('tools','history'):requests[r['request_key']]+=counts[r['id']]
        if r['learning_pool']=='history':history[r['request_key']]+=counts[r['id']]
    if max(requests.values())>CONFIG['max_tools_per_request'] or max(history.values())>CONFIG['max_history_per_request']:
        raise ValueError('Request concentration exceeded')
    if any(not 0<len(r['input_ids'])<=4096 for r in rows+dev):raise ValueError('Invalid context')
    if any(r['role']!='development' for r in dev) or set(index)&{r['id'] for r in dev}:raise ValueError('Evaluation overlap')
    return freeze,index,dev,steps


def train(data,adapter,output,resume=None):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from scale_lab.model import batch, load_model, score
    from tool_lab.guarded_update import attempt_update
    from .objectives import mixed_decision_loss,target_tensors
    from .history_objectives import reference_kl
    from .pilot_state import checkpoint,restore,parameter_hash
    if not torch.cuda.is_available():raise ValueError('Remote CUDA runtime required; local model jobs remain paused')
    freeze,index,development,steps=verify(data,adapter);freeze_sha=file_hash(data/'freeze.json')
    if resume is None:
        output.mkdir(parents=True,exist_ok=False)
        receipt=dict(status='loading',started_at=time.time(),completed_steps=0,freeze_sha256=freeze_sha,
                     parent_adapter_sha256=PARENT,config=CONFIG,training_presentations=0)
    else:
        receipt=json.loads((output/'run.json').read_text())
        if receipt['freeze_sha256']!=freeze_sha:raise ValueError('Resume freeze changed')
    def save(**values):
        receipt.update(values);temporary=output/'run.json.tmp';write_json(temporary,receipt);temporary.replace(output/'run.json')
        print(json.dumps(values,allow_nan=False),flush=True)
    def guard():
        if time.time()-receipt['started_at']>CONFIG['max_seconds']:raise TimeoutError('Bounded history pilot deadline')
    def stop(*_):raise InterruptedError('Remote termination requested')
    signal.signal(signal.SIGTERM,stop);random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed']);writer=None
    try:
        model=load_model(freeze['model'],'cuda',resume/'adapter' if resume else adapter,training=True)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        labels,pad=freeze['label_token_ids'],freeze['pad_id']
        def forward(rows,padding=64):
            inputs,label_tensor,mask,_=batch(rows,labels,pad,'cuda',padding)
            return score(model,inputs,label_tensor,mask)
        @torch.no_grad()
        def probabilities(rows):
            mode=model.training;model.eval();result=[]
            try:
                for i in range(0,len(rows),CONFIG['eval_batch']):
                    guard();chunk=rows[i:i+CONFIG['eval_batch']];p=forward(chunk).softmax(-1).cpu().tolist()
                    result.extend(v[:len(r['option_ids'])] for r,v in zip(chunk,p,strict=True))
            finally:model.train(mode)
            return result
        def supervised(rows,logits=None):
            logits=forward(rows) if logits is None else logits
            return mixed_decision_loss(logits,*target_tensors(rows,logits.shape[1],'cuda'))
        suites=defaultdict(list)
        for r in development:suites[r['suite']].append(r)
        writer=SummaryWriter(str(output/'events'),purge_step=receipt['completed_steps']+1 if resume else None)
        def evaluate(step):
            result={};started=time.time()
            for name,rs in sorted(suites.items()):
                rs=sorted(rs,key=lambda r:(len(r['input_ids']),r['id']));p=probabilities(rs)
                write_rows(output/f'{step}-{name}-predictions.jsonl',[dict(id=r['id'],probabilities=v) for r,v in zip(rs,p,strict=True)])
                result[name]=summarize(rs,p)
                for key,value in result[name]['macro'].items():writer.add_scalar(name+'/'+key,value,step)
                print(json.dumps(dict(evaluation_step=step,suite=name,macro=result[name]['macro'],n=len(rs))),flush=True)
            write_json(output/f'{step}-metrics.json',result)
            writer.add_scalar('runtime/evaluation_seconds',time.time()-started,step)
            return result
        reference_path=output/'reference-general.jsonl'
        if resume:
            state=restore(model,optimizer,resume)
            if state['freeze_sha256']!=freeze_sha or state['completed_steps']!=receipt['completed_steps']:
                raise ValueError('Restart schedule differs')
            if file_hash(reference_path)!=state['reference_sha256']:raise ValueError('Restart reference cache changed')
            baseline,best,best_step,stale=state['baseline'],state['best'],state['best_step'],state['stale']
            write_json(output/'restart-qualification.json',dict(status='passed',separate_process=True,
                completed_steps=state['completed_steps'],restored_trainable=state['trainable'],optimizer_states=len(optimizer.state),
                optimizer_state_restored=True,rng_state_restored=True,reference_sha256=state['reference_sha256']))
            save(status='training',restart_verified=True)
        else:
            initial=parameter_hash(model)
            if initial['sha256']!=INITIAL:raise ValueError('Wrong starting trainable weights')
            ordered=sorted(index.values(),key=lambda r:len(r['input_ids']));probes=[ordered[0],ordered[-1]]
            model.eval()
            with torch.no_grad():
                individual=[forward([r],1).softmax(-1)[0,:len(r['option_ids'])] for r in probes]
                together=forward(probes).softmax(-1)
                difference=max((p-together[i,:len(p)]).abs().max().item() for i,p in enumerate(individual))
                choices=all(p.argmax().item()==together[i,:len(p)].argmax().item() for i,p in enumerate(individual))
            if difference>.025 or not choices:raise ValueError('Long-context padding qualification failed')
            model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            loss=supervised(probes).mean();loss.backward()
            def gradient_ok():
                return all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable) and any(p.grad is not None and p.grad.abs().max()>0 for p in trainable)
            if not torch.isfinite(loss) or not gradient_ok():raise ValueError('Supervised gradient qualification failed')
            optimizer.zero_grad(set_to_none=True)
            # A controlled distribution perturbation tests the regularizer's gradient.
            # It is a no-update numerical probe, never an outcome or training label.
            artificial={r['id']:(p*.9+.1/len(p)).cpu().tolist() for r,p in zip(probes,individual,strict=True)}
            regularizer=reference_kl(forward(probes),probes,artificial).mean();regularizer.backward()
            if not torch.isfinite(regularizer) or not gradient_ok():raise ValueError('Reference-gradient qualification failed')
            optimizer.zero_grad(set_to_none=True)
            if parameter_hash(model)!=initial:raise ValueError('Qualification mutated weights')
            write_json(output/'runtime-qualification.json',dict(status='passed',probe_lengths=[len(r['input_ids']) for r in probes],
                max_probability_difference=difference,choices_equal=choices,backward_presentations=4,optimizer_steps=0,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),initial_trainable=initial))
            general=sorted((r for r in index.values() if r['learning_pool']=='general'),key=lambda r:(len(r['input_ids']),r['id']))
            save(status='caching_parent_general_reference',reference_questions=len(general))
            reference_values=probabilities(general)
            write_rows(reference_path,[dict(id=r['id'],probabilities=p) for r,p in zip(general,reference_values,strict=True)])
            save(reference_forward_questions=len(general),reference_forward_tokens=sum(len(r['input_ids']) for r in general),reference_sha256=file_hash(reference_path))
            baseline=evaluate(0);best=baseline;best_step=0;stale=0
            write_json(output/'selected.json',dict(step=0,source='original_parent',qualifies=False))
        references={r['id']:r['probabilities'] for r in read_rows(reference_path)}
        if set(references)!={r['id'] for r in index.values() if r['learning_pool']=='general'}:raise ValueError('Reference coverage differs')
        reference_sha=file_hash(reference_path)
        def save_checkpoint(step,suffix=''):
            path=output/f'checkpoint-{step:04d}{suffix}'
            checkpoint(model,optimizer,path,dict(completed_steps=step,freeze_sha256=freeze_sha,reference_sha256=reference_sha,
                baseline=baseline,best=best,best_step=best_step,stale=stale))
            write_json(output/'latest.json',dict(path=path.name,step=step,sha256=file_hash(path/'state.pt')))
            return path
        total=sum(CONFIG['per_step'].values());save(status='training')
        for step in range(receipt['completed_steps']+1,CONFIG['max_steps']+1):
            guard();started=time.time();model.train()
            scheduled=sorted((index[x] for x in steps[step-1]),key=lambda r:(len(r['input_ids']),r['id']))
            probes=[r for kind in CONFIG['per_step'] for r in sorted((r for r in scheduled if r['learning_pool']==kind),key=lambda r:r['id'])[:CONFIG['probes_per_pool']]]
            before=probabilities(probes)
            def update(mutated):
                optimizer.zero_grad(set_to_none=True);supervised_sum=0.;reference_sum=0.
                with (output/'consumption.jsonl').open('a') as ledger:
                    for i in range(0,len(scheduled),CONFIG['micro_batch']):
                        guard();chunk=scheduled[i:i+CONFIG['micro_batch']];logits=forward(chunk)
                        main_loss=supervised(chunk,logits).sum()/total;loss=main_loss
                        positions=[j for j,r in enumerate(chunk) if r['learning_pool']=='general']
                        if positions:
                            anchor=reference_kl(logits[positions],[chunk[j] for j in positions],references).sum()/CONFIG['per_step']['general']
                            loss=loss+CONFIG['reference_weight']*anchor;reference_sum+=float(anchor.detach())
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
                        loss.backward();supervised_sum+=float(main_loss.detach())
                        for r in chunk:ledger.write(json.dumps(dict(step=step,id=r['id'],pool=r['learning_pool'],tokens=len(r['input_ids'])))+'\n')
                        ledger.flush()
                norm=torch.nn.utils.clip_grad_norm_(trainable,CONFIG['grad_clip'],error_if_nonfinite=True)
                optimizer.step();metrics=dict(supervised_loss=supervised_sum,reference_kl=reference_sum,gradient_norm=float(norm))
                mutated(metrics);return metrics
            def measure():
                after=probabilities(probes);diagnostic=divergence(before,after)
                with (output/'guard-probes.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=step,ids=[r['id'] for r in probes],before=before,after=after,**diagnostic))+'\n')
                return diagnostic
            def record(entry):
                with (output/'optimizer-attempts.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,**entry),allow_nan=False)+'\n')
            attempt=attempt_update(model,optimizer,update,measure,max_mean_kl=CONFIG['max_mean_update_kl'],
                                   max_individual_kl=CONFIG['max_individual_update_kl'],record=record)
            if not attempt['accepted']:
                save_checkpoint(receipt['completed_steps'],'-restored-stop')
                save(status='stopped',stop_reason='rejected_update_'+attempt['reason'],rejected_step=step);break
            save(completed_steps=step,training_presentations=step*total,step_seconds=time.time()-started,
                 best_step=best_step,last_guard=attempt['diagnostic'],**attempt['metrics'])
            for key,value in attempt['metrics'].items():writer.add_scalar('train/'+key,value,step)
            writer.add_scalar('guard/mean_kl',attempt['diagnostic']['mean_full_kl'],step)
            metrics=None;reason=None
            if step%CONFIG['eval_every']==0 or step==CONFIG['max_steps']:
                metrics=evaluate(step);checks=gates(metrics,baseline)
                if checks['eligible'] and better(metrics,best):best,best_step,stale=metrics,step,0
                else:stale+=1
                write_json(output/f'{step}-gates.json',checks)
                if not checks['retention'] or not checks['product_slices'] or not checks['history_retention']:reason='retention_breach'
                elif stale>=CONFIG['patience']:reason='two_checks_without_eligible_improvement'
                save(best_step=best_step,last_gates=checks,stale=stale)
            if step==CONFIG['restart_after_step'] or metrics is not None:
                path=save_checkpoint(step)
                if best_step==step:write_json(output/'selected.json',dict(step=step,source=str(path.name+'/adapter'),qualifies=gates(best,baseline)['qualifies'],adapter_sha256=file_hash(path/'adapter/adapter_model.safetensors')))
                if step==CONFIG['restart_after_step'] and resume is None:save(status='restart_pending');return 75
            if reason:save(status='stopped',stop_reason=reason);break
        else:save(status='completed',stop_reason='scheduled_limit')
        consumed=[x for s in steps[:receipt['completed_steps']] for x in s]
        all_backward=read_rows(output/'consumption.jsonl') if (output/'consumption.jsonl').exists() else []
        final=dict(status=receipt['status'],completed_steps=receipt['completed_steps'],selected_step=best_step,gates=gates(best,baseline),
            unique_consumed_questions=len(set(consumed)),consumed_presentations=len(consumed),
            uncommitted_backward_presentations=len(all_backward)-len(consumed),consumed_tokens=sum(len(index[x]['input_ids']) for x in consumed),
            by_pool=dict(Counter(index[x]['learning_pool'] for x in consumed)),freeze_sha256=freeze_sha,parent_adapter_sha256=PARENT,
            reference_sha256=reference_sha,reference_forward_questions=len(references))
        write_json(output/'result.json',final);save(finished_at=time.time(),result=final);return 0
    except BaseException as error:
        save(status='failed',error_type=type(error).__name__,error=str(error),finished_at=time.time());raise
    finally:
        if writer:writer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    a=p.parse_args();raise SystemExit(train(a.data,a.adapter,a.output,a.resume))
