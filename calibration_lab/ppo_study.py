"""Controlled algorithm, capacity, and data-coverage study with actual PPO-Clip."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

from .study_environment import DecisionEnvironment, generate, DOMAINS, EXPANDED_COMPONENTS, REPORT_GRID
from .train import ROOT, metrics, temperature_fit, reinforce_loss, write_json, digest


SOURCES = ('ppo_study.py', 'study_environment.py', 'environment.py', 'train.py')
GRID = torch.from_numpy(REPORT_GRID.copy())


class Network(nn.Module):
    def __init__(self, width, outputs=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3,width), nn.Tanh(), nn.Linear(width,width),
                                    nn.Tanh(), nn.Linear(width,outputs))

    def forward(self,x):
        return self.layers(x*2-1)


def distribution(logits, objective):
    return (torch.distributions.Categorical(logits=logits) if objective == 'forecast'
            else torch.distributions.Bernoulli(logits=logits.squeeze(-1)))


def clipped_policy_loss(new_log_probability, old_log_probability, advantages, clip=.2):
    log_ratio = new_log_probability-old_log_probability.detach()
    ratio = log_ratio.exp()
    advantages = advantages.detach()
    objective = torch.minimum(ratio*advantages, ratio.clamp(1-clip,1+clip)*advantages)
    return -objective.mean()


@torch.no_grad()
def predict(model, observations, objective, temperature=1.):
    outputs = model(torch.from_numpy(observations))
    if objective == 'forecast':
        policy = outputs.softmax(-1)
        return {'forecast':(policy@GRID).numpy(), 'modal_forecast':GRID[policy.argmax(-1)].numpy(),
                'report_second_moment':(policy@GRID.square()).numpy()}
    return {'forecast':(outputs.squeeze(-1)/temperature).sigmoid().numpy(), 'logits':outputs.squeeze(-1).numpy()}


def selection_loss(model, batch, objective):
    result = predict(model,batch.observations,objective)
    q,y = result['forecast'].astype(float),batch.outcomes.astype(float)
    if objective == 'labels':
        q = np.clip(q,1e-7,1-1e-7)
        return float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)))
    if objective == 'accuracy':
        return -float(np.mean(y*q+(1-y)*(1-q)))
    return float(np.mean(result['report_second_moment']-2*y*q+y))


def ppo_update(policy, critic, policy_optimizer, value_optimizer, x, actions, rewards,
               old_log_probability, old_values, objective, epochs=4, clip=.2, target_divergence=.03):
    """Terminal episodes: return=reward; no bootstrap or discounting is needed."""
    rewards = rewards.detach()
    old_log_probability, old_values = old_log_probability.detach(), old_values.detach()
    advantages = rewards-old_values
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)
    actor_updates = 0
    peak_clip_fraction, peak_divergence = 0.,0.
    policy_loss_value, value_loss_value = 0.,0.
    for _ in range(epochs):
        new_logp = distribution(policy(x),objective).log_prob(actions)
        with torch.no_grad():
            log_ratio = new_logp-old_log_probability
            divergence = ((log_ratio.exp()-1)-log_ratio).mean().item()
            clip_fraction = ((log_ratio.exp()-1).abs()>clip).float().mean().item()
            peak_divergence,peak_clip_fraction = max(peak_divergence,divergence),max(peak_clip_fraction,clip_fraction)
        if divergence > target_divergence:
            break
        loss = clipped_policy_loss(new_logp,old_log_probability,advantages,clip)
        policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),5,error_if_nonfinite=True)
        policy_optimizer.step()
        actor_updates += 1
        policy_loss_value = loss.item()
    for _ in range(epochs):
        value_loss = .5*(critic(x).squeeze(-1)-rewards).square().mean()
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),5,error_if_nonfinite=True)
        value_optimizer.step()
        value_loss_value = value_loss.item()
    with torch.no_grad():
        log_ratio = distribution(policy(x),objective).log_prob(actions)-old_log_probability
        divergence = ((log_ratio.exp()-1)-log_ratio).mean().item()
        clip_fraction = ((log_ratio.exp()-1).abs()>clip).float().mean().item()
    return {'loss':policy_loss_value,'value_loss':value_loss_value,'policy_updates':actor_updates,
            'value_updates':epochs,'peak_clip_fraction':max(peak_clip_fraction,clip_fraction),
            'peak_divergence':max(peak_divergence,divergence),
            'final_divergence':divergence,'advantage_standard_deviation':float((rewards-old_values).std(unbiased=False))}


def train_one(args,root,algorithm,objective,width,seed,regime,validation,initial_name=None):
    label = 'supervised_continue' if algorithm == 'supervised' and initial_name else algorithm
    name = f'{label}-{objective}-{regime}-w{width}-s{seed}'
    folder = root/name
    folder.mkdir()
    torch.manual_seed(seed)
    policy = Network(width,21 if objective == 'forecast' else 1)
    if initial_name:
        policy.load_state_dict(load_file(root/initial_name/'policy.safetensors'))
    save_file(policy.state_dict(),folder/'initial.safetensors')
    critic = None
    if algorithm == 'ppo':
        torch.manual_seed(seed+18000)
        critic = Network(width)
        save_file(critic.state_dict(),folder/'initial_critic.safetensors')
        value_optimizer = torch.optim.Adam(critic.parameters(),lr=.001)
    torch.manual_seed(seed+40000)
    learning_rate = .0003 if algorithm == 'ppo' else .001
    policy_optimizer = torch.optim.Adam(policy.parameters(),lr=learning_rate)
    stream_seed = (4200000 if initial_name else 3200000 if regime == 'expanded' else 3100000)+seed
    environment = DecisionEnvironment(stream_seed,regime,objective)
    history,best,selected = [],float('inf'),-1
    updates,critic_updates,seen = 0,0,0
    started = time.perf_counter()
    def select(step):
        nonlocal best,selected
        value = selection_loss(policy,validation,objective)
        history.append({'rollout':step,'selection_loss':value})
        if value < best:
            best,selected = value,step
            save_file(policy.state_dict(),folder/'policy.safetensors')
            if critic:
                save_file(critic.state_dict(),folder/'critic.safetensors')
        if step:
            print(f'{name}: {step}/{args.rollouts}; validation {value:.6f}; selected {selected}',flush=True)
    select(0)
    with (folder/'trace.jsonl').open('w') as trace, (folder/'examples.jsonl').open('w') as examples:
        for step in range(1,args.rollouts+1):
            observation = environment.reset(args.batch_size)
            x = torch.from_numpy(observation)
            actions,rewards = None,None
            if algorithm == 'supervised':
                labels = torch.from_numpy(environment.labels())
                loss = F.binary_cross_entropy_with_logits(policy(x).squeeze(-1),labels)
                policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),5,error_if_nonfinite=True)
                policy_optimizer.step()
                row = {'loss':loss.item(),'gradient_norm':norm.item(),'policy_updates':1,'value_updates':0}
            elif algorithm == 'reinforce':
                action_distribution = distribution(policy(x),objective)
                actions = action_distribution.sample()
                reward_values,terminated = environment.step(actions.numpy())
                assert terminated.all()
                rewards = torch.from_numpy(reward_values)
                loss = reinforce_loss(action_distribution.log_prob(actions),rewards)
                policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),5,error_if_nonfinite=True)
                policy_optimizer.step()
                row = {'loss':loss.item(),'gradient_norm':norm.item(),'policy_updates':1,'value_updates':0}
            else:
                with torch.no_grad():
                    action_distribution = distribution(policy(x),objective)
                    actions = action_distribution.sample()
                    old_logp = action_distribution.log_prob(actions)
                    old_values = critic(x).squeeze(-1)
                reward_values,terminated = environment.step(actions.numpy())
                assert terminated.all()
                rewards = torch.from_numpy(reward_values)
                row = ppo_update(policy,critic,policy_optimizer,value_optimizer,x,actions,rewards,old_logp,
                                 old_values,objective,epochs=args.ppo_epochs)
            updates += row['policy_updates']
            critic_updates += row['value_updates']
            seen += args.batch_size
            row.update(rollout=step,episodes=args.batch_size,episode_sha256=environment.episode_digest())
            if rewards is not None:
                row.update(mean_reward=float(rewards.mean()),
                           actions_sha256=hashlib.sha256(actions.numpy().tobytes()).hexdigest(),
                           rewards_sha256=hashlib.sha256(rewards.numpy().tobytes()).hexdigest())
            if not all(np.isfinite(v) for v in row.values() if isinstance(v,float)):
                raise FloatingPointError('Non-finite update statistic')
            trace.write(json.dumps(row)+'\n')
            if step in (1,args.rollouts):
                for item in environment.audit_examples(None if actions is None else actions.numpy(),None if rewards is None else rewards.numpy()):
                    examples.write(json.dumps({'rollout':step,**item})+'\n')
            if step%args.evaluate_every==0 or step==args.rollouts:
                select(step)
    policy.load_state_dict(load_file(folder/'policy.safetensors'))
    if critic:
        critic.load_state_dict(load_file(folder/'critic.safetensors'))
    write_json(folder/'history.json',history)
    write_json(folder/'manifest.json',{'name':name,'algorithm':algorithm,'objective':objective,'width':width,
               'seed':seed,'regime':regime,'environment_seed':stream_seed,'initial_run':initial_name,
               'policy_parameters':sum(p.numel() for p in policy.parameters()),
               'critic_parameters':sum(p.numel() for p in critic.parameters()) if critic else 0,
               'rollouts':args.rollouts,'batch_size':args.batch_size,'episodes':seen,
               'policy_updates':updates,'critic_updates':critic_updates,'policy_learning_rate':learning_rate,
               'selected_rollout':selected,'selection_loss':best,'elapsed_seconds':time.perf_counter()-started,
               'policy_sha256':digest(folder/'policy.safetensors'),'initial_sha256':digest(folder/'initial.safetensors'),
               'source_sha256':{name:digest(Path(__file__).parent/name) for name in SOURCES}})
    return name,policy,objective


def run_evaluation(root,models,seed,size,domains):
    results = {}
    for index,domain in enumerate(domains):
        batch = generate(np.random.default_rng(seed+index*100),size,domain)
        folder = root/'test'/domain
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'environment.npz',observations=batch.observations,outcomes=batch.outcomes,posterior=batch.posterior)
        rows = {}
        for name,(policy,objective,temperature) in models.items():
            result = predict(policy,batch.observations,objective,temperature)
            np.savez_compressed(folder/f'{name}.npz',**result)
            rows[name] = metrics(result,batch)
        rows['oracle_posterior'] = metrics({'forecast':batch.posterior},batch)
        rows['constant_half'] = metrics({'forecast':np.full(size,.5)},batch)
        results[domain] = rows
        print('Final evaluation complete:',domain,flush=True)
    write_json(root/'results.json',results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--rollouts',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--ppo-epochs',type=int,default=4)
    parser.add_argument('--evaluate-every',type=int,default=100)
    parser.add_argument('--widths',nargs='+',type=int,default=[32,128])
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,23,37,53,71])
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args()
    if (args.rollouts<1 or args.batch_size<4 or args.batch_size%4 or args.ppo_epochs<1 or args.evaluate_every<1
            or any(w<1 for w in args.widths) or len(set(args.seeds))!=len(args.seeds) or len(set(args.widths))!=len(args.widths)):
        parser.error('Positive counts, batch divisible by four, and distinct widths/seeds required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = args.output or ROOT/'output/ppo_study_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root.mkdir(parents=True,exist_ok=False)
    snapshot = root/'code_snapshot'
    snapshot.mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    for name in ('requirements-calibration.txt','tests/test_ppo_study.py'):
        (snapshot/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,snapshot/name)
    shutil.copy2(ROOT/'docs/ppo-data-protocol.md',snapshot/'protocol.md')
    protocol = vars(args).copy()
    protocol['output'] = str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root)
    protocol.update(validation_seed=7100001,calibration_seed=7100003,
                    final_seed=7200001,validation_size=8192,calibration_size=8192,test_size=32768,domains=DOMAINS,
                    expanded_components=EXPANDED_COMPONENTS,expanded_ablation_width=max(args.widths),
                    value_target='one-step terminal reward',clip=.2,target_divergence=.03,entropy_bonus=0,
                    interpretation='experience-matched recipes; PPO reuses rollouts and trains a separate critic, so compute is not matched',
                    checkpoint_selection='own objective on the same narrow validation sample for every regime; observed outcomes only',
                    report_grid=REPORT_GRID.tolist(),
                    final_test_policy='all checkpoints and temperatures fixed before generating fresh final test')
    write_json(root/'protocol.json',protocol)
    write_json(root/'runtime.json',{'python':platform.python_version(),'platform':platform.platform(),
               'threads':torch.get_num_threads(),'packages':{n:importlib.metadata.version(n) for n in ('torch','numpy','safetensors')}})
    validation = generate(np.random.default_rng(7100001),8192)
    models = {}
    def run(algorithm,objective,width,seed,regime,initial_name=None):
        name,policy,objective = train_one(args,root,algorithm,objective,width,seed,regime,validation,initial_name)
        models[name] = (policy,objective,1.)
    # First fit the direct supervised references; their selected weights anchor
    # the paired correctness-reward continuations at both model sizes.
    for regime,widths in [('narrow',args.widths),('expanded',[max(args.widths)])]:
        for width in widths:
            for seed in args.seeds:
                run('supervised','labels',width,seed,regime)
            for algorithm in ('reinforce','ppo'):
                for seed in args.seeds:
                    run(algorithm,'forecast',width,seed,regime)
    for width in args.widths:
        for seed in args.seeds:
            run('supervised','labels',width,seed,'narrow',f'supervised-labels-narrow-w{width}-s{seed}')
        for algorithm in ('reinforce','ppo'):
            for seed in args.seeds:
                run(algorithm,'accuracy',width,seed,'narrow',f'supervised-labels-narrow-w{width}-s{seed}')
    if args.pilot:
        development = {name:metrics(predict(policy,validation.observations,objective),validation)
                       for name,(policy,objective,_) in models.items()}
        write_json(root/'development_validation.json',development)
        print('PILOT COMPLETE; no final test opened:',root,flush=True)
        return
    calibration = generate(np.random.default_rng(7100003),8192)
    temperatures = {}
    for name,(policy,objective,_) in list(models.items()):
        if objective == 'accuracy':
            temperature = temperature_fit(predict(policy,calibration.observations,objective)['logits'],calibration.outcomes)
            temperatures[name] = temperature
            models[name+'-temperature'] = (policy,objective,temperature)
    write_json(root/'selection.json',{'selected_rollouts':{p.parent.name:json.loads(p.read_text())['selected_rollout'] for p in sorted(root.glob('*/manifest.json'))},
                                     'temperatures':temperatures,'final_test_opened':False})
    print('All selections sealed. Opening the reserved final test.',flush=True)
    run_evaluation(root,models,7200001,32768,DOMAINS)
    write_json(root/'artifacts_sha256.json',{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    print('COMPLETE:',root,flush=True)


if __name__ == '__main__':
    main()
