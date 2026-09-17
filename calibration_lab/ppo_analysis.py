"""Audit the PPO study and summarize all algorithms, widths, seeds, and domains."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from safetensors.torch import load_file

from .analyze import stats
from .ppo_study import Network, predict, selection_loss, ppo_update, distribution, SOURCES
from .study_environment import DecisionEnvironment, generate
from .train import write_json, digest, temperature_fit, metrics


FIELDS = ('expected_brier','posterior_rmse','expected_log_loss','hard_accuracy','sampled_binary_accuracy',
          'mean_workflow_regret','mean_inspection_workflow_regret','expected_calibration_error_10_bins',
          'sampled_report_expected_brier','modal_report_expected_brier','modal_report_posterior_rmse')


def read(path):
    return json.loads(Path(path).read_text())


def group(name):
    base,seed = name.rsplit('-s',1)
    return base+('-temperature' if seed.endswith('-temperature') else '')


def compare_tree(actual, expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key in expected:
            compare_tree(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            compare_tree(a,b)
    elif isinstance(expected,(float,int)):
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
    else:
        assert actual==expected


def audit_metrics(result, saved, probability):
    q = saved['forecast'].astype(float)
    p = probability
    values = {'expected_brier':np.mean(p*(1-p)+(q-p)**2),
              'posterior_rmse':np.sqrt(np.mean((q-p)**2)),
              'hard_accuracy':np.mean(np.where(q>=.5,p,1-p)),
              'sampled_binary_accuracy':np.mean(p*q+(1-p)*(1-q))}
    if 'report_second_moment' in saved:
        values['sampled_report_expected_brier'] = np.mean(saved['report_second_moment']-2*p*q+p)
    for key,value in values.items():
        np.testing.assert_allclose(value,result[key],atol=1e-12,rtol=1e-12)
    assert sum(row['count'] for row in result['reliability_bins']) == len(q)
    for row in result['workflows']:
        t,c = row['threshold'],row['inspection_cost']
        estimated = np.column_stack([(1-t)*q,t*(1-q)])
        actual = np.column_stack([(1-t)*p,t*(1-p)])
        if c is not None:
            estimated = np.column_stack([estimated,np.full(len(q),c)])
            actual = np.column_stack([actual,np.full(len(q),c)])
        actions = estimated.argmin(1)
        costs = actual[np.arange(len(q)),actions]
        np.testing.assert_allclose(costs.mean(),row['expected_cost'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose((costs-actual.min(1)).mean(),row['regret'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose((actions==2).mean(),row['inspection_rate'],atol=1e-12,rtol=1e-12)


def verify(root):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protocol,selection = read(root/'protocol.json'),read(root/'selection.json')
    hashes = read(root/'artifacts_sha256.json')
    for name,value in hashes.items():
        assert digest(root/name) == value, name
    for name in SOURCES:
        assert digest(Path(__file__).parent/name) == digest(root/'code_snapshot'/name), 'Use the frozen study source'
    validation = generate(np.random.default_rng(protocol['validation_seed']),protocol['validation_size'])
    models,manifests,streams = {},{},{}
    actor_updates,critic_updates,ppo_replays,clipped_rollouts = 0,0,0,0
    for name,selected in selection['selected_rollouts'].items():
        folder = root/name
        manifest,history = read(folder/'manifest.json'),read(folder/'history.json')
        manifests[name] = manifest
        assert selected == min(history,key=lambda row:row['selection_loss'])['rollout']
        assert selected == manifest['selected_rollout']
        for source,value in manifest['source_sha256'].items():
            assert digest(root/'code_snapshot'/source) == value
        assert digest(folder/'policy.safetensors') == manifest['policy_sha256']
        assert digest(folder/'initial.safetensors') == manifest['initial_sha256']
        outputs = 21 if manifest['objective']=='forecast' else 1
        initial = load_file(folder/'initial.safetensors')
        if manifest['initial_run']:
            origin = load_file(root/manifest['initial_run']/'policy.safetensors')
        else:
            torch.manual_seed(manifest['seed'])
            origin = Network(manifest['width'],outputs).state_dict()
        for key in initial:
            torch.testing.assert_close(initial[key],origin[key],atol=0,rtol=0)
        policy = Network(manifest['width'],outputs)
        policy.load_state_dict(load_file(folder/'policy.safetensors'))
        np.testing.assert_allclose(selection_loss(policy,validation,manifest['objective']),manifest['selection_loss'],atol=1e-12,rtol=1e-12)
        models[name] = (policy,manifest['objective'],1.)
        rows = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        assert len(rows) == protocol['rollouts']
        assert [r['rollout'] for r in rows] == list(range(1,protocol['rollouts']+1))
        assert all(r['episodes']==protocol['batch_size'] for r in rows)
        assert all(np.isfinite(v) for row in rows for v in row.values() if isinstance(v,float))
        assert sum(r['episodes'] for r in rows)==manifest['episodes']
        assert sum(r['policy_updates'] for r in rows)==manifest['policy_updates']
        assert sum(r['value_updates'] for r in rows)==manifest['critic_updates']
        actor_updates += manifest['policy_updates']
        critic_updates += manifest['critic_updates']
        stream_key = manifest['environment_seed']
        episode_hashes = [r['episode_sha256'] for r in rows]
        if stream_key in streams:
            assert episode_hashes == streams[stream_key]
        else:
            environment = DecisionEnvironment(stream_key,manifest['regime'],manifest['objective'])
            for expected in episode_hashes:
                environment.reset(protocol['batch_size'])
                assert environment.episode_digest()==expected
            streams[stream_key] = episode_hashes
        for item in [json.loads(x) for x in (folder/'examples.jsonl').read_text().splitlines()]:
            if 'reward' not in item:
                continue
            expected = (float(item['action']==item['outcome']) if manifest['objective']=='accuracy'
                        else 1-(protocol['report_grid'][item['action']]-item['outcome'])**2)
            np.testing.assert_allclose(expected,item['reward'],atol=1e-7,rtol=1e-7)
        if manifest['algorithm'] == 'ppo':
            assert all(1<=r['policy_updates']<=protocol['ppo_epochs'] and r['value_updates']==protocol['ppo_epochs'] for r in rows)
            assert all(0<=r['peak_clip_fraction']<=1 for r in rows)
            clipped_rollouts += sum(r['peak_clip_fraction']>0 for r in rows)
            # Replay the first actual PPO update from saved initial weights,
            # including sampled actions, scalar rewards, all reuse epochs, and critic.
            replay_policy,replay_critic = Network(manifest['width'],outputs),Network(manifest['width'])
            replay_policy.load_state_dict(initial)
            replay_critic.load_state_dict(load_file(folder/'initial_critic.safetensors'))
            environment = DecisionEnvironment(stream_key,manifest['regime'],manifest['objective'])
            x = torch.from_numpy(environment.reset(protocol['batch_size']))
            torch.manual_seed(manifest['seed']+40000)
            with torch.no_grad():
                dist = distribution(replay_policy(x),manifest['objective'])
                actions = dist.sample()
                old_logp,old_values = dist.log_prob(actions),replay_critic(x).squeeze(-1)
            reward_values,terminated = environment.step(actions.numpy())
            assert terminated.all()
            assert hashlib.sha256(actions.numpy().tobytes()).hexdigest()==rows[0]['actions_sha256']
            assert hashlib.sha256(reward_values.tobytes()).hexdigest()==rows[0]['rewards_sha256']
            replay = ppo_update(replay_policy,replay_critic,
                                torch.optim.Adam(replay_policy.parameters(),lr=manifest['policy_learning_rate']),
                                torch.optim.Adam(replay_critic.parameters(),lr=.001),x,actions,torch.from_numpy(reward_values),
                                old_logp,old_values,manifest['objective'],epochs=protocol['ppo_epochs'])
            for key,value in replay.items():
                np.testing.assert_allclose(value,rows[0][key],atol=1e-7,rtol=1e-7)
            ppo_replays += 1
    calibration = generate(np.random.default_rng(protocol['calibration_seed']),protocol['calibration_size'])
    for name,temperature in selection['temperatures'].items():
        policy,objective,_ = models[name]
        z = predict(policy,calibration.observations,objective)['logits']
        np.testing.assert_allclose(temperature_fit(z,calibration.outcomes),temperature,atol=1e-9,rtol=1e-9)
        models[name+'-temperature'] = (policy,objective,temperature)
    assert selection['final_test_opened'] is False
    results = read(root/'results.json')
    largest_delta,count = 0.,0
    for index,domain in enumerate(protocol['domains']):
        batch = generate(np.random.default_rng(protocol['final_seed']+index*100),protocol['test_size'],domain)
        folder = root/'test'/domain
        environment = np.load(folder/'environment.npz')
        np.testing.assert_array_equal(environment['observations'],batch.observations)
        np.testing.assert_array_equal(environment['outcomes'],batch.outcomes)
        prior,reliability,signal = environment['observations'].astype(float).T
        odds = prior/(1-prior)*np.where(signal==1,reliability/(1-reliability),(1-reliability)/reliability)
        p = odds/(1+odds)
        np.testing.assert_allclose(p,environment['posterior'],atol=1e-14,rtol=1e-14)
        if domain=='held_out_combination':
            assert np.all(((prior<.15)|(prior>.85))&(reliability<.5))
        for name,(policy,objective,temperature) in models.items():
            saved = np.load(folder/f'{name}.npz')
            rebuilt = predict(policy,batch.observations,objective,temperature)
            for key,value in rebuilt.items():
                largest_delta = max(largest_delta,float(np.max(np.abs(value-saved[key]))))
                np.testing.assert_allclose(value,saved[key],atol=1e-7,rtol=1e-7)
            compare_tree(metrics(rebuilt,batch),results[domain][name])
            audit_metrics(results[domain][name],saved,p)
            count += 1
    return {'passed':True,'hashed_artifacts':len(hashes),'trained_policies':len(manifests),
            'auxiliary_value_networks':ppo_replays,'policy_updates':actor_updates,'value_updates':critic_updates,
            'distinct_episode_streams_rebuilt':len(streams),'unique_generated_training_episodes':len(streams)*protocol['rollouts']*protocol['batch_size'],
            'ppo_initial_updates_replayed':ppo_replays,'ppo_rollouts_with_clipped_samples':clipped_rollouts,
            'model_domain_evaluations_rebuilt':count,'maximum_reloaded_prediction_difference':largest_delta,
            'checks':['every artifact checksum','frozen source','cold and warm initial weights','all selected validation objectives',
                      'every training stream regenerated and matched across methods/sizes','scalar rollout rewards',
                      'all PPO first updates independently replayed','update and critic counts','temperature refitting',
                      'fresh final states regenerated','independent Bayes formula','all predictions reloaded','all recorded metrics recomputed',
                      'independent Brier identity and every downstream expected cost']}


def summarize(root):
    results,protocol = read(root/'results.json'),read(root/'protocol.json')
    summary = {'domains':{},'paired_algorithm_differences':{},'paired_capacity_differences':{},'paired_data_differences':{},
               'note':'Five-seed means, sample standard deviations and full ranges. Paired differences use the same seed and test states. Negative Brier/cost differences are improvements. These ranges are not confidence intervals.'}
    for domain,runs in results.items():
        groups = sorted({group(name) for name in runs if '-s' in name})
        summary['domains'][domain] = {}
        for key in groups:
            rows = [value for name,value in runs.items() if '-s' in name and group(name)==key]
            value = {field:stats([r[field] for r in rows]) for field in FIELDS if field in rows[0]}
            example_rows = [next(w for w in r['workflows'] if w['threshold']==.5 and w['inspection_cost']==.05) for r in rows]
            value['inspection_example'] = {k:stats([r[k] for r in example_rows]) for k in ['expected_cost','regret','inspection_rate']}
            summary['domains'][domain][key] = value
        summary['domains'][domain]['oracle_posterior'] = runs['oracle_posterior']
        summary['domains'][domain]['constant_half'] = runs['constant_half']
        def differences(first,second):
            return {field:stats([runs[f'{first}-s{seed}'][field]-runs[f'{second}-s{seed}'][field] for seed in protocol['seeds']])
                    for field in ['expected_brier','posterior_rmse','hard_accuracy','mean_workflow_regret']}
        summary['paired_algorithm_differences'][domain] = {str(w):differences(f'ppo-forecast-narrow-w{w}',f'reinforce-forecast-narrow-w{w}') for w in protocol['widths']}
        if len(protocol['widths'])==2:
            small,large = sorted(protocol['widths'])
            summary['paired_capacity_differences'][domain] = {algorithm:differences(f'{algorithm}-{objective}-narrow-w{large}',f'{algorithm}-{objective}-narrow-w{small}')
                for algorithm,objective in [('supervised','labels'),('reinforce','forecast'),('ppo','forecast')]}
        width = protocol['expanded_ablation_width']
        summary['paired_data_differences'][domain] = {algorithm:differences(f'{algorithm}-{objective}-expanded-w{width}',f'{algorithm}-{objective}-narrow-w{width}')
                for algorithm,objective in [('supervised','labels'),('reinforce','forecast'),('ppo','forecast')]}
    manifests = {p.parent.name:read(p) for p in root.glob('*/manifest.json')}
    summary['training'] = {}
    for key in sorted({group(name) for name in manifests}):
        rows = [m for name,m in manifests.items() if group(name)==key]
        summary['training'][key] = {field:stats([r[field] for r in rows]) for field in ['policy_parameters','critic_parameters','episodes','policy_updates','critic_updates','elapsed_seconds','selected_rollout']}
    summary['temperatures'] = read(root/'selection.json')['temperatures']
    return summary


def tables(root,summary,output):
    lines = ['# Every method, width, training-data regime, and seed','',summary['note'],'']
    results = read(root/'results.json')
    for domain,groups in summary['domains'].items():
        lines += [f'## {domain}','','| Recipe | Brier mean [min, max] | Posterior error | Hard-choice accuracy | Extra workflow cost |','| :--- | ---: | ---: | ---: | ---: |']
        for key,v in groups.items():
            if key in ('oracle_posterior','constant_half'):
                continue
            b=v['expected_brier']
            lines.append(f"| {key} | {b['mean']:.6f} [{b['minimum']:.6f}, {b['maximum']:.6f}] | {v['posterior_rmse']['mean']:.6f} | {100*v['hard_accuracy']['mean']:.3f}% | {v['mean_workflow_regret']['mean']:.6f} |")
        lines += ['','### Individual runs','','| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |','| :--- | ---: | ---: | ---: | ---: |']
        for name,v in results[domain].items():
            lines.append(f"| {name} | {v['expected_brier']:.6f} | {v['posterior_rmse']:.6f} | {100*v['hard_accuracy']:.3f}% | {v['mean_workflow_regret']:.6f} |")
        lines.append('')
    Path(output).write_text('\n'.join(lines).rstrip()+'\n')


def figures(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#fafaf7','axes.facecolor':'#fafaf7','savefig.facecolor':'#fafaf7','svg.fonttype':'none'})
    output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(output/f'{name}.png',dpi=180,bbox_inches='tight')
        svg=output/f'{name}.svg'
        fig.savefig(svg,bbox_inches='tight')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    normal=summary['domains']['in_distribution']
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
    recipes=[('supervised','labels','Supervised'),('reinforce','forecast','Simple policy gradient'),('ppo','forecast','Proximal policy optimization')]
    for ax,field,title in zip(axes,['expected_brier','posterior_rmse'],['Expected Brier score','Error against the true probability']):
        for offset,width,color in [(-.18,32,'#247c85'),(.18,128,'#b08120')]:
            vs=[normal[f'{a}-{o}-narrow-w{width}'][field] for a,o,_ in recipes]
            means=np.array([v['mean'] for v in vs])
            error=np.array([[v['mean']-v['minimum'] for v in vs],[v['maximum']-v['mean'] for v in vs]])
            ax.bar(np.arange(3)+offset,means,width=.34,color=color,label=f'{width} units per hidden layer',yerr=error,capsize=3)
        ax.set_xticks(range(3),['Supervised','Simple policy\ngradient','Proximal policy\noptimization'])
        ax.set_title(title+' ↓')
    axes[0].axhline(normal['oracle_posterior']['expected_brier'],color='#333',ls='--',label='Exact-probability reference')
    axes[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Does a larger model or a different learning method help?',fontsize=15)
    fig.supxlabel('Narrow training data; fresh test from the same distribution. Five-seed means and full ranges.\nReward-trained models choose among 21 probability reports; their mean report is evaluated.',fontsize=9)
    save(fig,'method-and-capacity')
    domains=list(summary['domains'])
    rows=[(f'{algorithm}-{objective}-{regime}-w128',f'{label} · {regime}')
          for algorithm,objective,label in recipes for regime in ['narrow','expanded']]
    data=np.array([[100*summary['domains'][domain][key]['posterior_rmse']['mean'] for domain in domains] for key,_ in rows])
    fig,ax=plt.subplots(figsize=(12,5.7),layout='constrained')
    im=ax.imshow(data,cmap='YlOrRd',vmin=0,vmax=data.max(),aspect='auto')
    ax.set_xticks(range(6),['Original\nconditions','Weak\nsensor','Strong\nsensor','Extreme\npriors','Reversed\nsensor','Held-out\ncombination'])
    ax.set_yticks(range(6),[label for _,label in rows])
    for i in range(6):
        for j in range(6):
            ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>data.max()*.6 else '#222',fontsize=13)
    fig.colorbar(im,ax=ax,label='Probability error in percentage points (root mean square) ↓',shrink=.8)
    ax.set_title('Change data coverage while keeping the model size and episode budget fixed',pad=18)
    fig.supxlabel('All policies use width 128. Expanded training divides the same number of examples among four conditions.\nExtreme priors and reversed sensors are never combined in either training regime. Five-seed means.',fontsize=9)
    save(fig,'data-coverage')
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.5),layout='constrained')
    colors=['#247c85','#b08120','#c34d3f']
    for row,width in enumerate([32,128]):
        keys=[f'supervised_continue-labels-narrow-w{width}',f'reinforce-accuracy-narrow-w{width}',f'ppo-accuracy-narrow-w{width}']
        for col,field in enumerate(['hard_accuracy','expected_brier']):
            ax=axes[row,col]
            scale=100 if col==0 else 1
            vs=[normal[key][field] for key in keys]
            means=np.array([v['mean'] for v in vs])*scale
            error=np.array([[v['mean']-v['minimum'] for v in vs],[v['maximum']-v['mean'] for v in vs]])*scale
            ax.bar(range(3),means,color=colors,yerr=error,capsize=3,width=.65)
            ax.set_xticks(range(3),['Supervised\ncontinued','Simple policy\ngradient','Proximal policy\noptimization'])
            ax.set_title(f'Width {width}: '+('hard-choice accuracy (%) ↑' if col==0 else 'expected Brier score ↓'))
            for i,v in enumerate(means):
                ax.text(i,v+max(means)*.04,f'{v:.2f}' if col==0 else f'{v:.4f}',ha='center',fontsize=10)
            ax.set_ylim(0,max(means)*1.25)
    fig.suptitle('Do larger policies and a different learning method change the distinction?',fontsize=15)
    fig.supxlabel('Same supervised starting weights and same additional experience per width and seed.\nFive-seed means and full ranges. Raw action probabilities are deliberately evaluated as event forecasts.',fontsize=9)
    save(fig,'paired-correctness')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures',type=Path)
    args=parser.parse_args()
    verified=verify(args.run)
    summary=summarize(args.run)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'verification.json',verified)
    write_json(args.output/'summary.json',summary)
    write_json(args.output/'models.json',{p.parent.name:read(p) for p in sorted(args.run.glob('*/manifest.json'))})
    for name in ['protocol.json','selection.json','runtime.json','artifacts_sha256.json']:
        shutil.copy2(args.run/name,args.output/name)
    shutil.copytree(args.run/'code_snapshot',args.output/'code_snapshot',dirs_exist_ok=True)
    tables(args.run,summary,args.output/'tables.md')
    if args.figures:
        figures(summary,args.figures)
    print(json.dumps(verified,indent=2))
    print('Every recipe, seed and domain summarized:',args.output)


if __name__=='__main__':
    main()
