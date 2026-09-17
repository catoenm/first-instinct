"""Try the two-step policies on four fixed, authored situations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .inspection_environment import InspectionEnvironment, World
from .inspection_train import load_model, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/learned-inspection-v1'))
    parser.add_argument('--seed',type=int,choices=(11,23,37),default=11)
    parser.add_argument('--sample-seed',type=int,default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = {
        'fresh_reliable_source':[.5,.6,1,.95,0,.02,1,1],
        'exact_copy':[.5,.6,1,.6,1,.02,1,1],
        'expensive_source':[.5,.6,1,.95,0,.25,1,1],
        'reversed_source':[.5,.6,1,.15,0,.02,1,0],
    }
    for objective in ('accuracy','forecast'):
        model = load_model(args.run/f'{objective}-s{args.seed}')
        torch.manual_seed(args.sample_seed)
        for name,data in cases.items():
            world = World(np.array([data],dtype=np.float32))
            environment = InspectionEnvironment(world,objective)
            prediction = predict(model,environment.observe())
            steps,total = [],0.
            while len(environment.active):
                x = torch.from_numpy(environment.observe())
                with torch.no_grad():
                    action = int(model.distribution(x).sample()[0])
                buying = action==environment.buy_action
                reward,done = environment.step(np.array([action]))
                steps.append({'action':'inspect' if buying else 'report',
                              'report':None if buying else float(environment.grid[action]),
                              'reward':float(reward[0])})
                total += float(reward[0])
            key = 'probability_of_yes_action_if_stopping' if objective=='accuracy' else 'mean_probability_report_if_stopping'
            print(json.dumps({'case':name,'objective':objective,'fixture_is_authored':True,
                              'visible_initial_state':world.observations()[0].tolist(),
                              'inspection_probability':float(prediction['buy'][0]),
                              key:float(prediction['mean'][0]),'sampled_steps':steps,'episode_return':total}))


if __name__=='__main__':
    main()
