"""Frozen paired supervised shell-data experiment; no reinforcement updates."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from scale_lab.common import ROOT, MODELS, file_hash, read_rows, write_json, write_rows

SEEDS = (907, 1709)
RECIPE = dict(epochs=2, batch_size=8, accumulation=8, eval_every=100,
              validation_per_task=100000, learning_rate=1e-5, max_hours=2.)
GUARD = dict(accuracy_drop=.015, log_loss_increase=.04, improvement=1e-4, patience=3)


def select_rows(rows, per_task, seed):
    counts, result = {}, []
    ordered = list(rows); random.Random(seed).shuffle(ordered)
    for row in ordered:
        task = row['task']
        if counts.get(task, 0) < per_task:
            result.append(row); counts[task] = counts.get(task, 0) + 1
    return result


def prepare(shell, output, adapter):
    output.mkdir(parents=True, exist_ok=False)
    general = ROOT/'output/general-qwen35-9b-v2'
    shell_manifest = json.loads((shell/'manifest.json').read_text())
    general_manifest = json.loads((general/'manifest.json').read_text())
    for folder, manifest, splits in [(general,general_manifest,('train','validation')),
                                     (shell,shell_manifest,('train','validation','test'))]:
        for split in splits:
            if file_hash(folder/(split+'.jsonl'))!=manifest['outputs'][split+'.jsonl']:
                raise ValueError('Source data checksum mismatch')
    if shell_manifest['exclusions']:
        raise ValueError('Review token exclusions before freezing a training experiment')
    shell_rows = read_rows(shell/'train.jsonl')
    pool = read_rows(general/'train.jsonl'); random.Random(907).shuffle(pool)
    replay = pool[:30000]
    control = pool[:len(replay)+len(shell_rows)]
    if len(control) != len(replay)+len(shell_rows): raise ValueError('Insufficient control rows')
    retention = select_rows(read_rows(general/'validation.jsonl'), 12, 907)
    # One general aggregate and six shell task aggregates. Guard the former
    # separately; never rely on the aggregate score to preserve general behavior.
    validation = [dict(r, task='general-retention') for r in retention] + read_rows(shell/'validation.jsonl')
    for name, rows in [('control', control), ('shell', replay+shell_rows)]:
        folder = output/name; folder.mkdir()
        random.Random(907).shuffle(rows)
        write_rows(folder/'train.jsonl', rows); write_rows(folder/'validation.jsonl', validation)
        write_json(folder/'manifest.json', dict(model=MODELS['qwen35-9b'],
                   label_token_ids=shell_manifest['label_token_ids'], counts={'train':len(rows),'validation':len(validation)},
                   outputs={p.name:file_hash(p) for p in folder.glob('*.jsonl')}))
    shutil.copyfile(shell/'test.jsonl', output/'shell-test.jsonl')
    shutil.copyfile(ROOT/'output/mixed-game-training-v1b-data/transfer.jsonl', output/'general-transfer.jsonl')
    write_json(output/'shell-source-manifest.json', shell_manifest)
    train_ids={r['id'] for r in control+shell_rows}
    if train_ids & {r['id'] for r in validation}: raise ValueError('Training/validation identifier overlap')
    for name in ('shell-test.jsonl','general-transfer.jsonl'):
        if train_ids & {r['id'] for r in read_rows(output/name)}: raise ValueError('Training/test identifier overlap')
    source_paths = [p for package in ('scale_lab','general_lab','tool_lab') for p in (ROOT/package).glob('*.py')]
    source_paths += [ROOT/'requirements-scale-cuda.txt',ROOT/'test_shell_supervision.py',ROOT/'test_shell_experiment.py',ROOT/'docs/shell-supervised-v1-protocol.md']
    freeze = dict(schema='shell-supervised-v1', model=MODELS['qwen35-9b'], seeds=list(SEEDS),recipe=RECIPE,guard=GUARD,
                  starting_adapter_sha256=file_hash(adapter/'adapter_model.safetensors'),
                  files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()},
                  sources={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths},
                  mixture={'general_replay':len(replay),'new_shell':len(shell_rows),'control_general':len(control)},
                  scope='Supervised execution-backed decisions and deterministic completion forecasts; not PPO or live Harbor trajectories.')
    write_json(output/'freeze.json',freeze)
    return freeze


def verify(data):
    frozen=json.loads((data/'freeze.json').read_text())
    if frozen['seeds']!=list(SEEDS) or frozen['recipe']!=RECIPE or frozen['guard']!=GUARD:
        raise ValueError('Changed experiment recipe')
    for name,sha in frozen['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Changed frozen data: '+name)
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Changed frozen source: '+name)
    return frozen


def eligible(candidate, baseline):
    current=candidate['by_task']['general-retention']; before=baseline['by_task']['general-retention']
    return (current['accuracy']>=before['accuracy']-GUARD['accuracy_drop']
            and current['acceptable_set_log_loss']<=before['acceptable_set_log_loss']+GUARD['log_loss_increase'])


def evaluate(data, adapter, output):
    import torch
    from transformers import AutoTokenizer
    from general_lab.train import macro_metrics
    from scale_lab.model import load_model, evaluate as predict
    frozen=verify(data); output.mkdir(parents=True,exist_ok=False)
    torch.set_float32_matmul_precision('high')
    tokenizer=AutoTokenizer.from_pretrained(frozen['model']['id'],revision=frozen['model']['revision'],token=False)
    from scale_lab.common import label_token_ids
    model=load_model(frozen['model'],'cuda',adapter,training=False)
    for name in ('shell-test','general-transfer'):
        rows=read_rows(data/(name+'.jsonl'))
        predictions=predict(model,rows,label_token_ids(tokenizer),tokenizer.pad_token_id or tokenizer.eos_token_id,'cuda',8,64)
        write_rows(output/(name+'-predictions.jsonl'),predictions)
        metrics=macro_metrics(predictions)
        write_json(output/(name+'-metrics.json'),dict(metrics,adapter_sha256=file_hash(adapter/'adapter_model.safetensors'),data_sha256=file_hash(data/(name+'.jsonl'))))


def train_arm(data, adapter, output, seed, name, deadline):
    arguments=[sys.executable,'-m','tool_lab.shell_train','--data',str(data/name),'--adapter',str(adapter),'--output',str(output),'--device','cuda','--seed',str(seed)]
    for key,value in RECIPE.items():arguments += ['--'+key.replace('_','-'),str(value)]
    selected=output.parent/(output.name+'-selected'); shutil.copytree(adapter,selected)
    selection=dict(step=0,adapter_sha256=file_hash(selected/'adapter_model.safetensors'),checks=[],status='running')
    selection_path=output.parent/(output.name+'-selection.json');write_json(selection_path,selection)
    baseline=None; best=float('inf'); misses=0; consumed=set(); stopped=None
    with (output.parent/(output.name+'.log')).open('w') as log:
        child=subprocess.Popen(arguments,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        def examine():
            nonlocal baseline,best,misses
            receipt=output/'run.json';trace=output/'training.jsonl'
            if not receipt.exists():return
            try:current=json.loads(receipt.read_text())
            except ValueError:return
            if baseline is None and current.get('baseline'):
                baseline=current['baseline'];best=baseline['macro_log_loss'];selection['baseline']=baseline;write_json(selection_path,selection)
            if baseline is None or not trace.exists():return
            for line in trace.read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                measured=event.get('validation',{})
                if 'macro_log_loss' not in measured or event['step'] in consumed:continue
                consumed.add(event['step']);ok=eligible(measured,baseline)
                checkpoint=output/'validation-checkpoints'/str(event['step'])
                binding=json.loads((output/f"validation-checkpoint-{event['step']}.json").read_text())
                if (file_hash(checkpoint/'adapter_model.safetensors')!=binding['adapter_sha256'] or
                        file_hash(output/f"validation-step-{event['step']}.jsonl")!=binding['predictions_sha256']):
                    raise ValueError('Validation/checkpoint binding differs')
                improved=ok and measured['macro_log_loss']<best-GUARD['improvement']
                if improved:
                    # The trainer snapshots these exact weights synchronously
                    # before continuing optimization, so monitoring has no race.
                    temporary=selected.with_name(selected.name+'-copy')
                    if temporary.exists():shutil.rmtree(temporary)
                    before=binding['adapter_sha256']
                    shutil.copytree(checkpoint,temporary)
                    if file_hash(temporary/'adapter_model.safetensors')!=before:
                        raise ValueError('Checkpoint changed while snapshotting')
                    shutil.rmtree(selected);temporary.rename(selected)
                    best=measured['macro_log_loss'];misses=0
                    selection.update(step=event['step'],adapter_sha256=before,selected_validation=measured)
                else:misses+=1
                selection['checks'].append(dict(step=event['step'],eligible=ok,selected=improved,macro_log_loss=measured['macro_log_loss'],general=measured['by_task']['general-retention']))
                write_json(selection_path,selection)
                shutil.rmtree(checkpoint)
        try:
            while child.poll() is None:
                examine()
                reason=('validation_patience' if misses>=GUARD['patience'] else
                        'provider_reserve' if time.time()>deadline-1800 else None)
                if reason and stopped is None:
                    os.killpg(child.pid,signal.SIGTERM);stopped=time.time();selection['stop_reason']=reason;write_json(selection_path,selection)
                if stopped and time.time()-stopped>150:os.killpg(child.pid,signal.SIGKILL)
                time.sleep(2)
            # Early termination may add an unscheduled final validation; it
            # cannot select a checkpoint because it was not a scheduled check.
            examine()
            selection.update(status='complete' if child.returncode==0 else 'failed',exit_code=child.returncode)
            write_json(selection_path,selection)
            if child.returncode:raise RuntimeError('Training arm failed')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=150)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return selected


def pipeline(data,adapter,output,deadline):
    frozen=verify(data)
    if file_hash(adapter/'adapter_model.safetensors')!=frozen['starting_adapter_sha256']:raise ValueError('Wrong starting checkpoint')
    output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='running',started_at=time.time(),stages=[],freeze_sha256=file_hash(data/'freeze.json'))
    def save():write_json(output/'pipeline.json',receipt)
    save()
    def run_evaluation(name,path):
        entry=dict(name=name,status='running',started_at=time.time());receipt['stages'].append(entry);save()
        subprocess.run([sys.executable,'-m','tool_lab.shell_experiment','evaluate','--data',str(data),'--adapter',str(path),'--output',str(output/name)],check=True,timeout=min(1200,max(1,deadline-time.time()-1200)))
        entry.update(status='complete',completed_at=time.time());save()
    try:
        # Test outputs do not influence any recipe, stopping or checkpoint choice.
        run_evaluation('evaluate-original',adapter)
        for seed in SEEDS:
            for arm in ('control','shell'):
                if deadline-time.time()<RECIPE['max_hours']*3600+1800:raise TimeoutError('Insufficient time for another arm and recovery')
                name=f'{arm}-{seed}';entry=dict(name=name,status='running',started_at=time.time());receipt['stages'].append(entry);save()
                selected=train_arm(data,adapter,output/name,seed,arm,deadline)
                entry.update(status='complete',completed_at=time.time());save()
                run_evaluation('evaluate-'+name,selected)
        receipt.update(status='complete',completed_at=time.time());save()
    except BaseException as error:
        receipt.update(status='failed',completed_at=time.time(),error={'type':type(error).__name__,'detail':str(error)});save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify','evaluate','pipeline'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--output',type=Path);p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.data,a.output,a.adapter),indent=2))
    elif a.mode=='verify':print(json.dumps({'verified':verify(a.data)['schema']}))
    elif a.mode=='evaluate':evaluate(a.data,a.adapter,a.output)
    else:pipeline(a.data,a.adapter,a.output,a.deadline)
