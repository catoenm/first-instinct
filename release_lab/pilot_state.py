"""Atomic checkpoints for trainable weights, optimizer, schedule cursor and RNG."""
import hashlib
import os
from pathlib import Path
import random
import torch


def parameter_hash(model):
    digest=hashlib.sha256(); count=0
    for name,p in sorted(model.named_parameters()):
        if p.requires_grad:
            digest.update(name.encode());digest.update(p.detach().float().cpu().numpy().tobytes());count+=p.numel()
    return dict(sha256=digest.hexdigest(),parameters=count)


def rng_state():
    return dict(python=random.getstate(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def atomic_save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint(model,optimizer,path,state):
    path.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(path/'adapter')
    payload=dict(state,optimizer=optimizer.state_dict(),rng=rng_state(),trainable=parameter_hash(model))
    atomic_save(path/'state.pt',payload)
    return payload


def restore(model,optimizer,path):
    # Only an owned, digest-bound local checkpoint, never an untrusted pickle.
    state=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
    if parameter_hash(model)!=state['trainable']:
        raise ValueError('Restart model tensor identity differs')
    optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    return state
