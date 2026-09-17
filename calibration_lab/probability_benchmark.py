"""Paired probability probes with a separate answer file and faithful text views."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .inspection_environment import World, generate, oracle
from .inspection_train import predict, write_json

VARIANTS = ('initial','independent_zero','independent_one','copy_initial','copy_revealed','expensive')
FAMILIES = ('ordinary','reversed')
QUESTION = ('Given the stated data-generating mechanism and only the revealed readings, '
            'is the hidden switch on? Return the probability of yes, not the probability '
            'that taking an action is worthwhile. The offered reading is unknown unless explicitly revealed.')


def records(seed=9500001,groups_per_family=256):
    rng = np.random.default_rng(seed)
    result = []
    for family in FAMILIES:
        base = generate(rng,groups_per_family)
        base.data[:,4] = 0
        base.data[:,3] = rng.uniform(.1,.4,groups_per_family) if family=='reversed' else rng.uniform(.55,.98,groups_per_family)
        base.data[:,5] = .01
        for index,row in enumerate(base.data):
            for variant in VARIANTS:
                data = row.copy()
                revealed,reading = False,None
                if variant.startswith('independent_'):
                    revealed,reading = True,int(variant=='independent_one')
                if variant.startswith('copy_'):
                    data[4],data[3] = 1,data[1]
                    revealed,reading = variant=='copy_revealed',data[2]
                if variant=='expensive':
                    data[5] = .25
                world = World(data[None,:])
                observation = world.observations(revealed,reading)[0]
                # Hidden labels and unobserved potential readings never enter this record.
                result.append({'id':f'{family}-{index:04d}-{variant}',
                    'group':f'{family}-{index:04d}','family':family,'variant':variant,
                    'observation':observation.tolist(),
                    'expected_probability':float(world.posterior(revealed,reading)[0]),
                    'optimal_inspection':bool(oracle(world,'forecast')['buy'][0]) if not revealed else False})
    return result


def numeric(value):
    # Nine significant decimal digits round-trip the float32 source parameters.
    return format(value,'.9g')


def describe(observation,version=0):
    p,r,s,q,copy,price,revealed,reading = observation
    values = dict(prior_on=numeric(p),initial_reliability=numeric(r),initial_reading=int(s),
                  offered_reliability=numeric(q),price=numeric(price))
    mechanism = ('The hidden switch has one fixed binary state: on=1, off=0. '
                 'Its state is drawn first. Each stated sensor reliability is the probability '
                 'that its reading equals the hidden state, both when the state is 0 and when it is 1. '
                 'The reading process and prices do not change the switch. ')
    provenance = ('The offered source is an exact copy of the initial sensor reading; '
                  'it has no independent measurement or additional information.' if copy else
                  'The offered source is a fresh sensor whose errors are independent of the initial sensor conditional on the hidden state.')
    visibility = (f'The offered source was inspected and its reading is {int(reading)}.' if revealed else
                  'The offered source has not been inspected. Its reading is unknown.')
    if version==0:
        return (mechanism+f'Before any readings, the probability of on is {values["prior_on"]}. '
                f'The initial sensor reliability is {values["initial_reliability"]}, and its observed reading is {int(s)}. '
                f'The offered source reliability is {values["offered_reliability"]}. '+provenance+' '+visibility+
                f' Inspecting it costs {values["price"]} reward units. This price does not provide evidence about the switch.')
    if version==1:
        return (mechanism+'Case record:\n'+json.dumps(values,sort_keys=True)+'\n'
                'prior_on is the chance the switch is on before seeing any readings. '
                'initial_reading is observed. '+visibility+' '+provenance+' '
                'price is only an inspection expense; it contains no information about the switch.')
    raise ValueError('Unknown text version')


def export(folder,items):
    folder.mkdir(parents=True,exist_ok=False)
    public,answers = [],[]
    for item in items:
        for version in (0,1):
            identifier = f"{item['id']}-text{version}"
            public.append({'id':identifier,'group':item['group'],'family':item['family'],
                           'variant':item['variant'],'text_version':version,
                           'observation':item['observation'],'state':describe(item['observation'],version),
                           'instructions':QUESTION})
            answers.append({'id':identifier,'expected_probability':item['expected_probability'],
                            'optimal_inspection':item['optimal_inspection']})
    for name,rows in [('requests.jsonl.gz',public),('answers.jsonl.gz',answers)]:
        data=''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows).encode()
        (folder/name).write_bytes(gzip.compress(data,mtime=0))
    write_json(folder/'manifest.json',{'requests':len(public),'unique_numeric_cases':len(items),
        'families':list(FAMILIES),'variants':list(VARIANTS),'text_versions':2,
        'files':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                 for name in ('requests.jsonl.gz','answers.jsonl.gz')}})


def read_jsonl(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def numeric_metrics(items,probabilities,buy=None):
    if len(items)!=len(probabilities):
        raise ValueError('One probability per unique numeric case is required')
    q = np.asarray(probabilities,dtype=float)
    if not np.isfinite(q).all() or np.any((q<0)|(q>1+1e-6)):
        raise ValueError('Invalid probabilities')
    groups = {}
    for index,item in enumerate(items):
        groups.setdefault(item['group'],{})[item['variant']]=index
    result = {}
    for family in FAMILIES:
        ids=[i for i,r in enumerate(items) if r['family']==family]
        target=np.array([items[i]['expected_probability'] for i in ids])
        result[family]={'probability_rmse':float(np.sqrt(np.mean((q[ids]-target)**2)))}
        paired=[v for k,v in groups.items() if k.startswith(family+'-')]
        for label,left,right in [('copy_change','copy_initial','copy_revealed'),
                                 ('price_change','initial','expensive'),
                                 ('unseen_source_change','initial','copy_initial')]:
            result[family][label]=float(np.mean([abs(q[g[right]]-q[g[left]]) for g in paired]))
        deltas=[]; wrong=[]
        for g in paired:
            for variant in ('independent_zero','independent_one'):
                i,j=g['initial'],g[variant]
                expected=items[j]['expected_probability']-items[i]['expected_probability']
                delta=q[j]-q[i]
                deltas.append(abs(delta-expected))
                wrong.append(delta*np.sign(expected)<-1e-5)
        result[family]['independent_update_absolute_error']=float(np.mean(deltas))
        result[family]['wrong_direction_rate']=float(np.mean(wrong))
        if buy is not None:
            result[family]['price_inspection_increase_rate']=float(np.mean([
                buy[g['expensive']]>buy[g['initial']]+1e-5 for g in paired]))
    return result


def evaluate_model(model,items):
    output=predict(model,np.array([r['observation'] for r in items],dtype=np.float32))
    return numeric_metrics(items,output['mean'],output.get('buy')),output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=9500001)
    parser.add_argument('--groups-per-family',type=int,default=256)
    args=parser.parse_args()
    export(args.output,records(args.seed,args.groups_per_family))


if __name__=='__main__':
    main()
