"""Buy one additional observation, or stop and report a probability.

The hidden event and potential source reading are sampled before the agent acts.
Only purchased observations enter the model input. Exact posteriors are for
evaluation; the environment returns rewards based on the realized binary event.
"""
from dataclasses import dataclass
import hashlib

import numpy as np


DOMAINS = ('in_distribution','expensive_inspection','more_duplicates','reversed_new_source')


def report_grid(objective):
    if objective not in ('accuracy','forecast'):
        raise ValueError('Unknown objective')
    return np.array([0.,1.],dtype=np.float32) if objective=='accuracy' else np.linspace(0,1,21,dtype=np.float32)


@dataclass(frozen=True)
class World:
    # prior, initial reliability, initial reading, offered reliability, duplicate,
    # price, hidden event, potential offered reading
    data: np.ndarray

    @property
    def outcomes(self):
        return self.data[:,6]

    def observations(self, revealed=False, reading=None):
        x=np.zeros((len(self.data),8),dtype=np.float32)
        x[:,:6]=self.data[:,:6]
        x[:,6]=revealed
        x[:,7]=np.where(x[:,6]>0,self.data[:,7] if reading is None else reading,0)
        return x

    def posterior(self, revealed=False, reading=None):
        p,r,s,q,duplicate,_,_,offered=self.data.astype(float).T
        odds=p/(1-p)*np.where(s==1,r/(1-r),(1-r)/r)
        reading=offered if reading is None else reading
        odds *= np.where(np.asarray(revealed)&(duplicate==0),
                         np.where(np.asarray(reading)==1,q/(1-q),(1-q)/q),1.)
        return odds/(1+odds)

    def next_signal_probability(self):
        p=self.posterior()
        return np.where(self.data[:,4]==1,self.data[:,2],p*self.data[:,3]+(1-p)*(1-self.data[:,3]))

    def digest(self):
        return hashlib.sha256(self.data.tobytes()).hexdigest()


def generate(rng,size,domain='in_distribution'):
    if domain not in DOMAINS:
        raise ValueError(domain)
    p=rng.uniform(.1,.9,size).astype(np.float32)
    r=rng.uniform(.55,.85,size).astype(np.float32)
    q=rng.uniform(.1,.4,size) if domain=='reversed_new_source' else rng.uniform(.55,.98,size)
    q=q.astype(np.float32)
    duplicate=(rng.random(size)<(.8 if domain=='more_duplicates' else .25)).astype(np.float32)
    q=np.where(duplicate==1,r,q)
    cost=rng.uniform(.15,.30,size) if domain=='expensive_inspection' else rng.uniform(.002,.12,size)
    cost=cost.astype(np.float32)
    y=(rng.random(size)<p).astype(np.float32)
    s=np.where(rng.random(size)<r,y,1-y)
    fresh=np.where(rng.random(size)<q,y,1-y)
    offered=np.where(duplicate==1,s,fresh)
    return World(np.column_stack([p,r,s,q,duplicate,cost,y,offered]).astype(np.float32))


class InspectionEnvironment:
    def __init__(self,world,objective):
        self.world=world
        self.grid=report_grid(objective)
        self.buy_action=len(self.grid)
        self.revealed=np.zeros(len(world.data),dtype=bool)
        self.active=np.arange(len(world.data))

    def observe(self):
        return self.world.observations(self.revealed)[self.active]

    def step(self,actions):
        actions=np.asarray(actions)
        if not len(self.active) or actions.shape!=self.active.shape or not np.isin(actions,np.arange(self.buy_action+1)).all():
            raise ValueError('One valid action per active episode is required')
        buy=actions==self.buy_action
        if np.any(buy&self.revealed[self.active]):
            raise ValueError('The observation can only be purchased once')
        ids=self.active.copy()
        reward=np.empty(len(ids),dtype=np.float32)
        reward[buy]=-self.world.data[ids[buy],5]
        reward[~buy]=1-(self.grid[actions[~buy].astype(int)]-self.world.outcomes[ids[~buy]])**2
        self.revealed[ids[buy]]=True
        self.active=ids[buy]
        return reward,~buy


def optimal_report(probability,objective):
    grid=report_grid(objective).astype(float)
    return grid[np.abs(np.asarray(probability)[:,None]-grid).argmin(1)]


def squared_risk(report,probability):
    return probability*(1-probability)+(report-probability)**2


def oracle(world,objective):
    p=world.posterior()
    next_p=[world.posterior(True,s) for s in [0,1]]
    w=world.next_signal_probability()
    stop=squared_risk(optimal_report(p,objective),p)
    future=[squared_risk(optimal_report(v,objective),v) for v in next_p]
    buy=world.data[:,5]+(1-w)*future[0]+w*future[1]
    return {'buy':buy<stop,'cost':np.minimum(stop,buy),'stop_cost':stop,'buy_cost':buy}
