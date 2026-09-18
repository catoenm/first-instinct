"""Prospectively bounded CPU PPO check; this is not native CUDA PuffeRL training."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .contract import ACTIONS, OBS_SIZE, canonical, continuation
from .native import NativeEpisode, library


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(OBS_SIZE,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor=nn.Linear(64,len(ACTIONS));self.value=nn.Linear(64,1)
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,.01);nn.init.orthogonal_(self.value.weight,1.)

    def forward(self,obs):
        features=self.body(obs)
        return Categorical(logits=self.actor(features)),self.value(features).squeeze(-1)


def advantages(rewards,values,dones,bootstrap):
    """Episode boundaries stop both bootstrapping and advantage propagation."""
    result=np.zeros_like(rewards);running=np.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_value=bootstrap if t==len(rewards)-1 else values[t+1]
        active=1.-dones[t]
        delta=rewards[t]+active*next_value-values[t]
        running=delta+.95*active*running
        result[t]=running
    return result,result+values


def profiles(kind):
    if kind=='train':
        priors=([1,1,1,1,0,0],[7,1,1,1,0,0],[1,7,1,1,0,0],[1,1,1,7,0,0])
        inspections,atomic=(1,8),(4,16)
    elif kind=='validation':
        priors=([1,1,1,1,0,0],);inspections,atomic=(2,6),(5,14)
    else:
        priors=([1,1,1,1,0,0],) if kind=='familiar' else ([0,0,0,0,1,1],)
        inspections,atomic=(1,8),(4,16)
    return [{'costs':[i,i,a,4,12,12,12,8,0],'horizon':h,'prior':list(p)}
            for i,a,h,p in itertools.product(inspections,atomic,(3,6),priors)]


def weight_hash(policy):
    h=hashlib.sha256()
    for name,value in sorted(policy.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Limits:
    def __init__(self):self.transitions=0;self.started=time.monotonic()
    def step(self,episode,action):
        if self.transitions>=160000 or time.monotonic()-self.started>=900:
            raise RuntimeError('Small-policy time/transition cap reached')
        self.transitions+=1
        return episode.step(action)


@torch.no_grad()
def evaluate(policy,lib,kind,limits,mode='policy',repetitions=1):
    rng=random.Random(2026);rows=[];average_return=average_success=0.
    contexts=profiles(kind)
    for context_id,profile in enumerate(contexts):
        for world,weight in enumerate(profile['prior']):
            if not weight:continue
            for repeat in range(repetitions):
                with NativeEpisode(lib,world,profile) as episode:
                    actions=[];total=0.
                    while not episode.state()['done']:
                        if mode=='policy':
                            distribution,_=policy(torch.tensor([episode.observe()],dtype=torch.float32))
                            action=ACTIONS[int(distribution.probs.argmax(-1))]
                        elif mode=='random':action=rng.choice(ACTIONS)
                        elif mode=='public':action=continuation(episode.public())
                        else:action='finish'
                        total+=limits.step(episode,action);actions.append(action)
                    success=int(episode.state()['outcome']==1)
                    probability=weight/sum(profile['prior'])/len(contexts)/repetitions
                    average_return+=probability*total;average_success+=probability*success
                    rows.append({'context':context_id,'profile':profile,'world':world,'repeat':repeat,
                                 'actions':actions,'return':total,'success':success,
                                 'outcome':episode.state()['outcome'],'aggregation_weight':probability})
    return {'kind':kind,'mode':mode,'mean_return':average_return,'success_rate':average_success,
            'episodes':len(rows),'rows':rows}


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def train(seed,lib_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'rollouts').mkdir()
    torch.set_num_threads(2);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng=random.Random(seed);permutations=np.random.default_rng(seed+987)
    lib=library(lib_path);limits=Limits();policy=Policy();initial=weight_hash(policy)
    torch.save(policy.state_dict(),output/'initial.pt')
    optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
    train_profiles=profiles('train');selection=[];updates=[];episodes=[]
    config={'seed':seed,'algorithm':'local PyTorch PPO, not native CUDA PuffeRL',
            'device':'cpu','parameters':sum(p.numel() for p in policy.parameters()),
            'updates':64,'environments':64,'rollout_steps':32,'training_transitions':131072,
            'epochs':4,'minibatch':256,'learning_rate':.0003,'clip':.2,'entropy':.02,
            'value_coefficient':.5,'gradient_norm':.5,'gamma':1.,'lambda':.95,
            'max_transitions_with_evaluation':160000,'max_seconds':900,
            'torch_version':torch.__version__,'initial_weight_sha256':initial,
            'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__),Path(__file__).with_name('reservation_core.h'),
                                Path(__file__).with_name('contract.py'),Path(__file__).with_name('native.py')]}}
    # Relative source names keep the receipt suitable for public release.
    config['sources_sha256']={Path(k).name:v for k,v in config['sources_sha256'].items()}
    write(output/'config.json',config)
    best=-float('inf');best_update=0;status='running';error=None;optimizer_steps=0

    def new_episode():
        profile=rng.choice(train_profiles)
        world=rng.choices(range(6),weights=profile['prior'],k=1)[0]
        return NativeEpisode(lib,world,profile)

    def validate(update):
        nonlocal best,best_update
        report=evaluate(policy,lib,'validation',limits)
        write(output/f'validation-{update:03d}.json',report)
        selection.append({'update':update,'mean_return':report['mean_return'],'success_rate':report['success_rate']})
        if report['mean_return']>best+1e-10:
            best=report['mean_return'];best_update=update
            torch.save(policy.state_dict(),output/'selected.pt')

    try:
        validate(0)
        episodes=[new_episode() for _ in range(64)]
        current=np.asarray([e.observe() for e in episodes],dtype=np.float32)
        log=(output/'training.jsonl').open('w')
        try:
            for update in range(1,65):
                obs=np.empty((32,64,OBS_SIZE),dtype=np.float32)
                actions=np.empty((32,64),dtype=np.int64)
                old_logp=np.empty((32,64),dtype=np.float32)
                probabilities=np.empty((32,64,9),dtype=np.float32)
                rewards=np.empty((32,64),dtype=np.float32)
                dones=np.empty((32,64),dtype=np.float32)
                values=np.empty((32,64),dtype=np.float32)
                terminals=[]
                for t in range(32):
                    obs[t]=current
                    with torch.no_grad():
                        distribution,value=policy(torch.from_numpy(current))
                        action=distribution.sample()
                        actions[t]=action.numpy();old_logp[t]=distribution.log_prob(action).numpy()
                        probabilities[t]=distribution.probs.numpy();values[t]=value.numpy()
                    for i,episode in enumerate(episodes):
                        rewards[t,i]=limits.step(episode,ACTIONS[actions[t,i]])
                        ended=episode.state()['done'];dones[t,i]=ended
                        if ended:
                            terminals.append(episode.state()['outcome']);episode.close();episodes[i]=new_episode()
                        current[i]=episodes[i].observe()
                with torch.no_grad():_,bootstrap=policy(torch.from_numpy(current))
                adv,returns=advantages(rewards,values,dones,bootstrap.numpy())
                np.savez_compressed(output/'rollouts'/f'update-{update:03d}.npz',
                    observations=obs,actions=actions,old_logp=old_logp,probabilities=probabilities,
                    rewards=rewards,dones=dones,values=values,bootstrap=bootstrap.numpy(),
                    advantages=adv,returns=returns)
                x=torch.from_numpy(obs.reshape(-1,OBS_SIZE));a=torch.from_numpy(actions.reshape(-1))
                old=torch.from_numpy(old_logp.reshape(-1));target=torch.from_numpy(returns.reshape(-1))
                advantage=torch.from_numpy(adv.reshape(-1));advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
                statistics=[];stopped_early=False
                for epoch in range(4):
                    epoch_kl=[]
                    for indices in np.array_split(permutations.permutation(2048),8):
                        distribution,value=policy(x[indices]);new_logp=distribution.log_prob(a[indices])
                        log_ratio=new_logp-old[indices];ratio=log_ratio.exp()
                        actor=-torch.minimum(ratio*advantage[indices],ratio.clamp(.8,1.2)*advantage[indices]).mean()
                        critic=(value-target[indices]).square().mean();entropy=distribution.entropy().mean()
                        loss=actor+.5*critic-.02*entropy
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                        optimizer.zero_grad();loss.backward()
                        norm=nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
                        optimizer.step();optimizer_steps+=1
                        kl=float(((ratio-1)-log_ratio).detach().mean());epoch_kl.append(kl)
                        statistics.append([float(actor.detach()),float(critic.detach()),float(entropy.detach()),kl,float(norm)])
                    if np.mean(epoch_kl)>.05:stopped_early=True;break
                means=np.mean(statistics,axis=0).tolist()
                row={'update':update,'training_transitions':update*2048,'completed_episodes':len(terminals),
                     'sampled_success_rate':sum(t==1 for t in terminals)/max(1,len(terminals)),
                     'policy_loss':means[0],'value_loss':means[1],'entropy':means[2],
                     'approx_kl':means[3],'gradient_norm_before_clip':means[4],
                     'early_epoch_stop':stopped_early,'elapsed_seconds':time.monotonic()-limits.started}
                updates.append(row);log.write(canonical(row)+'\n');log.flush()
                if update%8==0:
                    validate(update)
                    print(canonical({'seed':seed,**selection[-1],'elapsed_seconds':row['elapsed_seconds']}),flush=True)
        finally:log.close()
        torch.save(policy.state_dict(),output/'latest.pt')
        final_weight=weight_hash(policy)
        selection_weight=None;diagnostics={}
        # Fixed final diagnostics are read only after the validation selection.
        for identity in ('initial','selected'):
            policy.load_state_dict(torch.load(output/f'{identity}.pt',weights_only=True))
            if identity=='selected':selection_weight=weight_hash(policy)
            diagnostics[identity]={}
            for kind in ('familiar','combined'):
                report=evaluate(policy,lib,kind,limits)
                write(output/f'{identity}-{kind}.json',report)
                diagnostics[identity][kind]={k:report[k] for k in ('mean_return','success_rate','episodes')}
        for mode in ('finish','public','random'):
            diagnostics[mode]={}
            for kind in ('familiar','combined'):
                report=evaluate(policy,lib,kind,limits,mode,16 if mode=='random' else 1)
                write(output/f'{mode}-{kind}.json',report)
                diagnostics[mode][kind]={k:report[k] for k in ('mean_return','success_rate','episodes')}
        status='complete'
        write(output/'diagnostics.json',diagnostics)
        write(output/'weights.json',{'initial':initial,'latest':final_weight,'selected':selection_weight})
    except BaseException as exc:
        status='failed';error={'type':type(exc).__name__,'detail':str(exc)};raise
    finally:
        for episode in episodes:episode.close()
        write(output/'run.json',{'status':status,'error':error,'seed':seed,'updates_completed':len(updates),
            'training_transitions':len(updates)*2048,'all_transitions':limits.transitions,
            'optimizer_steps':optimizer_steps,'selected_update':best_update,'validation_selection':selection,
            'elapsed_seconds':time.monotonic()-limits.started,'language_model_updated':False,
            'native_puffer_trainer_executed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for seed in (41,73):train(seed,args.library,args.output/f'seed-{seed}')
