"""Apply a separately frozen action-forward correction to the unchanged pilot."""
import argparse
import json
from pathlib import Path

from scale_lab.common import ROOT, file_hash, write_json
from tool_lab.canonical_actions import CanonicalActionPolicy, CONTRACT
from tool_lab import live_pilot_train as original
from tool_lab.live_pilot_plan import ARMS, SEEDS


def verify_correction(path):
    frozen=json.loads(path.read_text())
    if frozen['contract']!=CONTRACT or frozen['probability_tolerance']!=.001 or frozen['optimizer_attempts_in_failed_run']!=0:
        raise ValueError('Changed runtime correction contract')
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Changed correction source: '+name)
    return frozen


def train(args):
    correction=verify_correction(args.correction)
    if file_hash(args.data/'freeze.json')!=correction['data_freeze_sha256']:
        raise ValueError('Correction must use the unchanged data and recipe')
    # The original trainer is immutable. This explicit entrypoint substitutes
    # only its policy class and records the derived runtime in every receipt.
    original.LivePolicy=CanonicalActionPolicy
    try:
        original.train(args)
    finally:
        path=args.output/'run.json'
        if path.exists():
            receipt=json.loads(path.read_text())
            receipt.update(action_forward_contract=CONTRACT,runtime_correction_sha256=file_hash(args.correction))
            write_json(path,receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapter','output','worker-python','worker-source','retail-python','retail-plan','retail-source','correction'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--max-hours',type=float,default=.8);p.add_argument('--device',default='cuda')
    p.add_argument('--backend',choices=('catalog','docker'),default='catalog')
    train(p.parse_args())
