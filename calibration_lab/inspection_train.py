"""A two-step, outcome-reward experiment with a learned inspection policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from .inspection_environment import InspectionEnvironment, generate, report_grid

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('inspection_environment.py', 'inspection_train.py', 'inspection_evaluate.py')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Features(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8,width), nn.Tanh(), nn.Linear(width,width), nn.Tanh())

    def forward(self, x):
        # Fixed unit scaling, without computing any belief or planning feature.
        scale = x.new_tensor([1,1,1,1,1,10,1,1])
        return self.layers(x*scale*2-1)


class Policy(nn.Module):
    def __init__(self, objective, width=64):
        super().__init__()
        self.objective = objective
        self.features = Features(width)
        self.report = nn.Linear(width,len(report_grid(objective)))
        self.buy = nn.Linear(width,1)
        self.register_buffer('grid',torch.from_numpy(report_grid(objective)))

    def heads(self, x):
        h = self.features(x)
        return self.report(h), self.buy(h).squeeze(-1)

    def distribution(self, x, exploration_buy_probability=None):
        report, buy = self.heads(x)
        acquired = x[:,6]>0
        log_report = report.log_softmax(-1)
        if exploration_buy_probability is None:
            stop_logp, buy_logp = F.logsigmoid(-buy),F.logsigmoid(buy)
        else:
            if not 0<exploration_buy_probability<1:
                raise ValueError('Exploration probability must be strictly between zero and one')
            stop_logp = torch.full_like(buy,np.log1p(-exploration_buy_probability))
            buy_logp = torch.full_like(buy,np.log(exploration_buy_probability))
        stop_logp = torch.where(acquired,0.,stop_logp)
        buy_logp = torch.where(acquired, -1e9, buy_logp)
        return torch.distributions.Categorical(logits=torch.cat(
            [log_report+stop_logp[:,None],buy_logp[:,None]],dim=1))


class ScalarNetwork(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.features = Features(width)
        self.output = nn.Linear(width,1)

    def forward(self, x):
        return self.output(self.features(x)).squeeze(-1)


@torch.no_grad()
def predict(model, observations):
    chunks = []
    for start in range(0,len(observations),4096):
        x = torch.from_numpy(observations[start:start+4096])
        if isinstance(model,Policy):
            report, buy = model.heads(x)
            distribution = report.softmax(-1)
            chunks.append({'mean':(distribution@model.grid).numpy(),
                           'second':(distribution@model.grid.square()).numpy(),
                           'buy':torch.where(x[:,6]>0,0.,buy.sigmoid()).numpy()})
        else:
            q = model(x).sigmoid().numpy()
            chunks.append({'mean':q,'second':q*q})
    return {key:np.concatenate([part[key] for part in chunks]).astype(float) for key in chunks[0]}


def selection_loss(model, world):
    """Integrate policy actions, but use only observed validation outcomes."""
    first, last = [predict(model,world.observations(revealed)) for revealed in (False,True)]
    y = world.outcomes.astype(float)
    if isinstance(model,Policy):
        risk0, risk1 = [v['second']-2*y*v['mean']+y for v in (first,last)]
        b = first['buy']
        return float(np.mean((1-b)*risk0+b*(world.data[:,5]+risk1)))
    return float(np.mean([-(y*np.log(np.clip(v['mean'],1e-7,1-1e-7))+
                             (1-y)*np.log1p(-np.clip(v['mean'],1e-7,1-1e-7))) for v in (first,last)]))


def episode_returns(first_reward, acquired_ids, last_reward):
    first = first_reward.detach().clone()
    first[acquired_ids] += last_reward.detach()
    return torch.cat([first,last_reward.detach()])


@torch.no_grad()
def collect(policy, critic, world, exploration_buy_probability=None):
    environment = InspectionEnvironment(world,policy.objective)
    x0 = torch.from_numpy(environment.observe())
    d0 = policy.distribution(x0,exploration_buy_probability)
    a0 = d0.sample()
    r0, terminal = environment.step(a0.numpy())
    ids = torch.from_numpy(environment.active.copy())
    x1 = torch.from_numpy(environment.observe())
    if len(ids):
        d1 = policy.distribution(x1,exploration_buy_probability)
        a1 = d1.sample()
        r1, done = environment.step(a1.numpy())
        assert done.all() and not len(environment.active)
        logp = torch.cat([d0.log_prob(a0),d1.log_prob(a1)])
    else:
        a1, r1 = torch.empty(0,dtype=torch.long),np.empty(0,dtype=np.float32)
        logp = d0.log_prob(a0)
    x, actions = torch.cat([x0,x1]),torch.cat([a0,a1])
    rewards = torch.from_numpy(np.concatenate([r0,r1]))
    returns = episode_returns(torch.from_numpy(r0),ids,torch.from_numpy(r1))
    return {'x':x,'actions':actions,'old_logp':logp,'old_values':critic(x),
            'returns':returns,'rewards':rewards,'acquired_ids':ids,
            'episode_ids':torch.cat([torch.arange(len(x0)),ids]),
            'terminal':torch.from_numpy(np.concatenate([terminal,np.ones(len(ids),dtype=bool)]))}


def ppo_update(policy,critic,actor_optimizer,value_optimizer,batch,entropy_coefficient,epochs=4,
               exploration_buy_probability=None):
    x, actions = batch['x'],batch['actions']
    old_logp, returns, old_values = [batch[k].detach() for k in ('old_logp','returns','old_values')]
    advantages = returns-old_values
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)
    updates, peak_divergence, peak_clip = 0,0.,0.
    for _ in range(epochs):
        distribution = policy.distribution(x,exploration_buy_probability)
        log_ratio = distribution.log_prob(actions)-old_logp
        ratio = log_ratio.exp()
        divergence = ((ratio-1)-log_ratio).mean().detach().item()
        clip_fraction = ((ratio-1).abs()>.2).float().mean().detach().item()
        peak_divergence, peak_clip = max(peak_divergence,divergence),max(peak_clip,clip_fraction)
        if divergence > .03:
            break
        loss = -torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages).mean()
        loss = loss-entropy_coefficient*distribution.entropy().mean()
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(),5,error_if_nonfinite=True)
        actor_optimizer.step()
        updates += 1
    for _ in range(epochs):
        value_loss = .5*(critic(x)-returns).square().mean()
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(),5,error_if_nonfinite=True)
        value_optimizer.step()
    with torch.no_grad():
        log_ratio = policy.distribution(x,exploration_buy_probability).log_prob(actions)-old_logp
        final_divergence = ((log_ratio.exp()-1)-log_ratio).mean().item()
    return {'policy_updates':updates,'value_updates':epochs,'loss':loss.item(),
            'value_loss':value_loss.item(),'peak_divergence':max(peak_divergence,final_divergence),
            'peak_clip_fraction':peak_clip,'entropy_coefficient':entropy_coefficient}


def train_one(root, recipe, seed, validation, rollouts=4000, batch_size=512, evaluate_every=200,
              exploration_fraction=.25):
    folder = root/f'{recipe}-s{seed}'
    folder.mkdir()
    torch.manual_seed(seed)
    objective = 'forecast' if recipe=='forecast_no_exploration' else recipe
    if recipe=='forecast_no_exploration':
        exploration_fraction = 0.
    model = ScalarNetwork() if recipe=='supervised' else Policy(objective)
    save_file(model.state_dict(),folder/'initial.safetensors')
    optimizer = torch.optim.Adam(model.parameters(),lr=.001 if recipe=='supervised' else .0005)
    critic = None
    if recipe!='supervised':
        torch.manual_seed(seed+18000)
        critic = ScalarNetwork()
        save_file(critic.state_dict(),folder/'initial_critic.safetensors')
        value_optimizer = torch.optim.Adam(critic.parameters(),lr=.001)
    torch.manual_seed(seed+40000)
    rng = np.random.default_rng(8100000+seed)
    subset_rng = np.random.default_rng(8400000+seed)
    history, best, selected, transitions, actor_updates, value_updates = [],float('inf'),-1,0,0,0
    started = time.perf_counter()

    def select(step):
        nonlocal best,selected
        value = selection_loss(model,validation)
        history.append({'rollout':step,'selection_loss':value})
        if value<best:
            best,selected = value,step
            save_file(model.state_dict(),folder/'model.safetensors')
            if critic is not None:
                save_file(critic.state_dict(),folder/'critic.safetensors')
        print(f'{folder.name}: {step}/{rollouts}; validation {value:.6f}; selected {selected}',flush=True)

    select(0)
    with (folder/'trace.jsonl').open('w') as trace:
        for step in range(1,rollouts+1):
            world = generate(rng,batch_size)
            if recipe=='supervised':
                subset = subset_rng.random(batch_size)<.5
                x = torch.from_numpy(np.concatenate([world.observations(),world.observations(True)[subset]]))
                y = torch.from_numpy(np.concatenate([world.outcomes,world.outcomes[subset]]))
                loss = F.binary_cross_entropy_with_logits(model(x),y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                optimizer.step()
                count = len(x)
                row = {'loss':loss.item(),'policy_updates':1,'value_updates':0,
                       'observations_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
            else:
                exploration = .5 if step<=int(rollouts*exploration_fraction) else None
                batch = collect(model,critic,world,exploration)
                coefficient = .01*max(0.,1-step/(.75*rollouts))
                row = ppo_update(model,critic,optimizer,value_optimizer,batch,coefficient,
                                 exploration_buy_probability=exploration)
                row['exploration_buy_probability'] = exploration
                count = len(batch['x'])
                ids = batch['acquired_ids'].numpy()
                duplicate_count = (world.data[:,4]==1).sum()
                row.update(mean_return=float(batch['returns'][:batch_size].mean()),
                           purchases=len(ids),duplicate_purchases=int(world.data[ids,4].sum()),
                           duplicates=int(duplicate_count))
                for key in ('actions','rewards','returns'):
                    row[key+'_sha256'] = hashlib.sha256(batch[key].numpy().tobytes()).hexdigest()
                if step in (1,rollouts):
                    np.savez_compressed(folder/f'rollout-{step}.npz',world=world.data,
                                        **{key:value.numpy() for key,value in batch.items()})
            transitions += count
            actor_updates += row['policy_updates']
            value_updates += row['value_updates']
            row.update(rollout=step,root_episodes=batch_size,transitions=count,world_sha256=world.digest())
            if not all(np.isfinite(v) for v in row.values() if isinstance(v,float)):
                raise FloatingPointError('Non-finite training statistic')
            trace.write(json.dumps(row,allow_nan=False)+'\n')
            if step%evaluate_every==0 or step==rollouts:
                select(step)
    write_json(folder/'history.json',history)
    manifest = {'recipe':recipe,'objective':objective,'seed':seed,'width':64,'rollouts':rollouts,'batch_size':batch_size,
                'root_episodes':rollouts*batch_size,'transitions_or_labeled_states':transitions,
                'policy_updates':actor_updates,'value_updates':value_updates,
                'parameters':sum(p.numel() for p in model.parameters()),
                'critic_parameters':sum(p.numel() for p in critic.parameters()) if critic is not None else 0,
                'exploration_rollouts':int(rollouts*exploration_fraction) if critic is not None else 0,
                'training_world_seed':8100000+seed,'selected_rollout':selected,'selection_loss':best,
                'validation_world_sha256':validation.digest(),'seconds':time.perf_counter()-started,
                'files':{p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    write_json(folder/'manifest.json',manifest)
    return manifest


def load_model(folder):
    info = json.loads((folder/'manifest.json').read_text())
    model = ScalarNetwork(info['width']) if info['recipe']=='supervised' else Policy(info.get('objective',info['recipe']),info['width'])
    model.load_state_dict(load_file(folder/'model.safetensors'))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--rollouts',type=int,default=4000)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--evaluate-every',type=int,default=200)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--exploration-fraction',type=float,default=.25)
    args = parser.parse_args()
    if not 0<=args.exploration_fraction<1 or min(args.rollouts,args.batch_size,args.evaluate_every,args.threads)<1:
        parser.error('Positive sizes and an exploration fraction in [0,1) are required')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = args.output or ROOT/'output'/'inspection_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root.mkdir(parents=True,exist_ok=False)
    source = root/'source'
    source.mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent/name,source/name)
    for name in ('tests/test_inspection.py','requirements-calibration.txt'):
        (source/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,source/name)
    shutil.copy2(ROOT/'docs'/'learned-inspection-protocol.md',source/'protocol.md')
    write_json(root/'run.json',{'pilot':args.pilot,'rollouts':args.rollouts,'batch_size':args.batch_size,
               'evaluate_every':args.evaluate_every,'threads':args.threads,'torch':torch.__version__,
               'exploration_fraction':args.exploration_fraction,
               'numpy':np.__version__,'python':platform.python_version(),'platform':platform.platform(),
               'source_sha256':{p.name:digest(p) for p in sorted(source.iterdir())}})
    validation = generate(np.random.default_rng(8200001),8192)
    models = []
    for seed in ([101] if args.pilot else [11,23,37]):
        recipes = ('supervised','accuracy','forecast') if args.pilot else ('supervised','accuracy','forecast','forecast_no_exploration')
        for recipe in recipes:
            models.append(train_one(root,recipe,seed,validation,args.rollouts,args.batch_size,
                                    args.evaluate_every,args.exploration_fraction))
    write_json(root/'sealed.json',{'final_test_opened':False,'models':models,
               'sealed_at':datetime.now(timezone.utc).isoformat()})
    from .inspection_evaluate import evaluate_run
    evaluate_run(root,pilot=args.pilot)
    print(root,flush=True)


if __name__=='__main__':
    main()
