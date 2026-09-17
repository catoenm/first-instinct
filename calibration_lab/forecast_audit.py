"""Does continuing forecast practice repair a learned inspection policy?"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from .inspection_environment import generate, report_grid
from .inspection_train import (ROOT, Policy, ScalarNetwork, collect, digest, load_model,
                               ppo_update, predict, selection_loss, write_json)

RECIPES = ('terminal_only','audit_early','audit_continuous','supervised')
SOURCES = ('forecast_audit.py','forecast_audit_evaluate.py','probability_benchmark.py',
           'inspection_environment.py','inspection_train.py','inspection_evaluate.py','jev_benchmark.py')


class ForecastExercise:
    """Forced reports on initial and acquired states; labels stay in the scorer."""
    def __init__(self, world):
        self.world = world
        self.observations = np.concatenate([world.observations(),world.observations(True)])
        self._outcomes = np.concatenate([world.outcomes,world.outcomes])

    def score(self, actions):
        actions = np.asarray(actions)
        if actions.shape!=self._outcomes.shape or not np.isin(actions,np.arange(21)).all():
            raise ValueError('One report-grid index is required per state')
        return 1-(report_grid('forecast')[actions.astype(int)]-self._outcomes)**2

    def labels(self):
        return self._outcomes.copy()


def exercise_distribution(policy,x):
    return torch.distributions.Categorical(logits=policy.heads(x)[0])


def audit_blocks_at(step,rollouts,recipe):
    if recipe=='audit_early':
        return 4 if step<=rollouts//4 else 0
    return 1 if recipe in ('audit_continuous','supervised') else 0


def exercise_update(policy,critic,optimizer,value_optimizer,exercise,generator,coefficient):
    x = torch.from_numpy(exercise.observations)
    with torch.no_grad():
        old_distribution = exercise_distribution(policy,x)
        actions = torch.multinomial(old_distribution.probs,1,generator=generator).squeeze(-1)
        old_logp = old_distribution.log_prob(actions)
        old_values = critic(x)
    # The reward learner receives only the sampled report's scalar reward.
    rewards = torch.from_numpy(exercise.score(actions.numpy()))
    advantages = rewards-old_values
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)
    updates,peak_divergence = 0,0.
    for _ in range(4):
        distribution = exercise_distribution(policy,x)
        log_ratio = distribution.log_prob(actions)-old_logp
        ratio = log_ratio.exp()
        divergence = ((ratio-1)-log_ratio).mean().detach().item()
        peak_divergence = max(peak_divergence,divergence)
        if divergence>.03:
            break
        loss = -torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages).mean()
        loss -= coefficient*distribution.entropy().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),5,error_if_nonfinite=True)
        optimizer.step()
        updates += 1
    for _ in range(4):
        value_loss = .5*(critic(x)-rewards).square().mean()
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),5,error_if_nonfinite=True)
        value_optimizer.step()
    return {'policy_updates':updates,'value_updates':4,'loss':float(loss.detach()),
            'value_loss':float(value_loss.detach()),'mean_reward':float(rewards.mean()),
            'peak_divergence':peak_divergence,
            'actions_sha256':hashlib.sha256(actions.numpy().tobytes()).hexdigest(),
            'rewards_sha256':hashlib.sha256(rewards.numpy().tobytes()).hexdigest()}


def validation_metrics(model, world):
    q = np.concatenate([predict(model,world.observations(v))['mean'] for v in (False,True)])
    y = np.tile(world.outcomes.astype(float),2)
    return {'observed_audit_brier':float(np.mean((q-y)**2)),
            'selection_loss_diagnostic_only':selection_loss(model,world)}


def train_one(root,recipe,seed,validation,rollouts=4000,batch_size=512,audit_worlds=128,every=200):
    if recipe not in RECIPES or rollouts%4 or rollouts<4:
        raise ValueError('Known recipe and a positive rollout count divisible by four are required')
    folder = root/f'{recipe}-s{seed}'
    folder.mkdir()
    torch.manual_seed(seed)
    model = ScalarNetwork() if recipe=='supervised' else Policy('forecast')
    save_file(model.state_dict(),folder/'initial.safetensors')
    optimizer = torch.optim.Adam(model.parameters(),lr=.001 if recipe=='supervised' else .0005)
    critic,audit_critic = None,None
    if recipe!='supervised':
        torch.manual_seed(seed+18000)
        critic = ScalarNetwork()
        value_optimizer = torch.optim.Adam(critic.parameters(),lr=.001)
        save_file(critic.state_dict(),folder/'initial_critic.safetensors')
    if recipe.startswith('audit_'):
        torch.manual_seed(seed+28000)
        audit_critic = ScalarNetwork()
        audit_optimizer = torch.optim.Adam(audit_critic.parameters(),lr=.001)
        save_file(audit_critic.state_dict(),folder/'initial_audit_critic.safetensors')
    torch.manual_seed(seed+40000)
    action_generator = torch.Generator().manual_seed(seed+50000)
    world_rng = np.random.default_rng(9100000+seed)
    audit_rng = np.random.default_rng(9400000+seed)
    audit_index = 0
    history = []
    counters = dict(root_episodes=0,interaction_transitions=0,audit_states=0,
                    interaction_policy_updates=0,interaction_value_updates=0,
                    audit_policy_updates=0,audit_value_updates=0)
    started = time.perf_counter()

    def record(step):
        row = {'rollout':step,**validation_metrics(model,validation)}
        history.append(row)
        print(f"{folder.name}: {step}/{rollouts}; audit Brier {row['observed_audit_brier']:.6f}",flush=True)

    record(0)
    with (folder/'interaction.jsonl').open('w') as interaction, (folder/'audits.jsonl').open('w') as audits:
        for step in range(1,rollouts+1):
            if critic is not None:
                world = generate(world_rng,batch_size)
                forced_buy = .5 if step<=rollouts//4 else None
                batch = collect(model,critic,world,forced_buy)
                coefficient = .01*max(0.,1-step/(.75*rollouts))
                row = ppo_update(model,critic,optimizer,value_optimizer,batch,coefficient,
                                 exploration_buy_probability=forced_buy)
                row.update(rollout=step,world_sha256=world.digest(),transitions=len(batch['x']),
                           mean_return=float(batch['returns'][:batch_size].mean()))
                interaction.write(json.dumps(row,allow_nan=False)+'\n')
                counters['root_episodes'] += batch_size
                counters['interaction_transitions'] += len(batch['x'])
                counters['interaction_policy_updates'] += row['policy_updates']
                counters['interaction_value_updates'] += row['value_updates']
                if step in (1,rollouts):
                    np.savez_compressed(folder/f'rollout-{step}.npz',world=world.data,
                                        **{key:value.numpy() for key,value in batch.items()})
            for _ in range(audit_blocks_at(step,rollouts,recipe)):
                audit_index += 1
                world = generate(audit_rng,audit_worlds)
                exercise = ForecastExercise(world)
                if recipe=='supervised':
                    x = torch.from_numpy(exercise.observations)
                    y = torch.from_numpy(exercise.labels())
                    for _ in range(4):
                        loss = F.binary_cross_entropy_with_logits(model(x),y)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                        optimizer.step()
                    row = {'loss':float(loss.detach()),'policy_updates':4,'value_updates':0}
                else:
                    coefficient = .01*max(0.,1-audit_index/(.75*rollouts))
                    row = exercise_update(model,audit_critic,optimizer,audit_optimizer,exercise,
                                          action_generator,coefficient)
                row.update(rollout=step,audit_block=audit_index,world_sha256=world.digest(),
                           observation_sha256=hashlib.sha256(exercise.observations.tobytes()).hexdigest())
                audits.write(json.dumps(row,allow_nan=False)+'\n')
                counters['audit_states'] += len(exercise.observations)
                counters['audit_policy_updates'] += row['policy_updates']
                counters['audit_value_updates'] += row['value_updates']
            if step==rollouts//4:
                save_file(model.state_dict(),folder/'after_first_quarter.safetensors')
            if step%every==0 or step==rollouts:
                record(step)
    # The final training step is preselected; validation cannot choose a checkpoint.
    save_file(model.state_dict(),folder/'model.safetensors')
    if critic is not None:
        save_file(critic.state_dict(),folder/'critic.safetensors')
    if audit_critic is not None:
        save_file(audit_critic.state_dict(),folder/'audit_critic.safetensors')
    write_json(folder/'history.json',history)
    manifest = {'recipe':recipe,'seed':seed,'objective':'forecast','width':64,
                'rollouts':rollouts,'batch_size':batch_size,'audit_worlds_per_block':audit_worlds,
                'audit_blocks':audit_index,'selected_rollout':rollouts,'selection_rule':'fixed_final_step',
                'parameters':sum(p.numel() for p in model.parameters()),
                'seconds':time.perf_counter()-started,'counts':counters,
                'training_world_seed':9100000+seed,'audit_world_seed':9400000+seed,
                'validation_world_sha256':validation.digest(),
                'files':{p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    write_json(folder/'manifest.json',manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--rollouts',type=int,default=4000)
    args = parser.parse_args()
    if args.rollouts<4 or args.rollouts%4:
        parser.error('Rollouts must be positive and divisible by four')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    source = args.output/'source'
    source.mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent/name,source/name)
    for name in ('test_forecast_audit.py','test_jev_benchmark.py','requirements-calibration.txt'):
        shutil.copy2(ROOT/name,source/name)
    shutil.copy2(ROOT/'docs'/'forecast-audit-protocol.md',source/'protocol.md')
    write_json(args.output/'run.json',{'pilot':args.pilot,'rollouts':args.rollouts,'batch_size':512,
        'audit_worlds_per_block':128,'threads':1,'torch':torch.__version__,'numpy':np.__version__,
        'python':platform.python_version(),'platform':platform.platform(),
        'source_sha256':{p.name:digest(p) for p in sorted(source.iterdir())}})
    validation = generate(np.random.default_rng(9200001),8192)
    models = []
    for seed in ([101] if args.pilot else [11,23,37]):
        for recipe in RECIPES:
            models.append(train_one(args.output,recipe,seed,validation,args.rollouts))
    write_json(args.output/'sealed.json',{'final_test_opened_at_seal':False,'models':models,
               'sealed_at':datetime.now(timezone.utc).isoformat()})
    from .forecast_audit_evaluate import evaluate_run
    evaluate_run(args.output,pilot=args.pilot)
    print(args.output,flush=True)


if __name__=='__main__':
    main()
