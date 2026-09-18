"""Bounded native/SQLite differential execution with exact conditional targets."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from .contract import (ACTIONS, CODES, OUTCOMES, PREFIXES, PROFILES, PUFFER_REVISION,
                       canonical, continuation, digest, expected_observation)
from .native import NativeEpisode, compile_core, library
from .sql_oracle import AttemptBudget, SqlEpisode

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def compare(native, actual):
    a,b = native.state(),actual.state()
    if a != b: raise AssertionError({'native':a,'sqlite':b})
    if native.public() != actual.public(): raise AssertionError('Public state differs')
    features = expected_observation(b,actual.profile)
    if len(features) != len(native.observe()) or any(abs(x-y)>1e-6 for x,y in zip(native.observe(),features)):
        raise AssertionError('Native feature encoding differs from public state')


def branch(lib, profile_id, world, prefix, first, budget):
    profile = PROFILES[profile_id]
    identity = {'profile':profile_id,'world':world,'prefix':prefix,'first':first}
    with NativeEpisode(lib,world,profile) as native, SqlEpisode(world,profile,budget,identity) as actual:
        compare(native,actual)
        public_history = [{'observation':actual.public()}]
        for action in prefix:
            x,y = native.step(action),actual.step(action)
            assert abs(x-y)<1e-6
            compare(native,actual)
            public_history.append({'action':action,'observation':actual.public()})
        context = {'profile':profile_id,'history':public_history}
        before_cost = actual.cost
        action = first
        while not actual.done:
            x,y = native.step(action),actual.step(action)
            assert abs(x-y)<1e-6
            compare(native,actual)
            if not actual.done: action = continuation(actual.public())
        verified = actual.verify()
        assert verified == native.state()['outcome']
        return dict(identity,context=context,context_sha256=digest(context),
                    outcome=OUTCOMES[verified],future_cost_quarters=actual.cost-before_cost,
                    prior_weight=profile['prior'][world],states_match=True,
                    steps=actual.history,sql_statement_count=len(actual.statements))


def conditional_targets(records):
    contexts = {}
    groups = defaultdict(list)
    for row in records:
        contexts[row['context_sha256']] = row['context']
        groups[(row['context_sha256'],row['first'])].append(row)
    targets = []
    for (key,action), rows in sorted(groups.items()):
        supported = [r for r in rows if r['prior_weight']]
        total = sum(r['prior_weight'] for r in supported)
        if not total: continue
        joint = defaultdict(int)
        for row in supported: joint[(row['outcome'],row['future_cost_quarters'])] += row['prior_weight']
        distribution = [{'outcome':outcome,'cost_quarters':cost,'weight':weight,'denominator':total}
                        for (outcome,cost),weight in sorted(joint.items())]
        success = Fraction(sum(w for (outcome,c),w in joint.items() if outcome == 'success'),total)
        expected_cost = sum(Fraction(cost*weight,total) for (outcome,cost),weight in joint.items())
        expected_value = sum(Fraction(((400 if outcome == 'success' else -400 if outcome in ('partial','forbidden') else 0)-cost)*weight,total*400)
                             for (outcome,cost),weight in joint.items())
        targets.append({'context_sha256':key,'public_context':contexts[key],'action':action,
                        'continuation':'public-repair-v1','joint':distribution,
                        'success_probability':[success.numerator,success.denominator],
                        'expected_cost_quarters':[expected_cost.numerator,expected_cost.denominator],
                        'expected_return':[expected_value.numerator,expected_value.denominator],
                        'supporting_worlds':len(supported)})
    return targets


def guards(lib,budget):
    rows=[]
    for world in range(6):
        with SqlEpisode(world,PROFILES[3],budget,{'guard':'noop','world':world}) as actual:
            actual.step('finish');assert actual.outcome==2
            rows.append({'guard':'noop','world':world,'outcome':'incomplete'})
    with SqlEpisode(0,PROFILES[0],budget,{'guard':'protected_corruption'}) as actual:
        actual.step('atomic')
        actual.db.execute("UPDATE inventory SET qty=8 WHERE sku='C'")
        actual.step('finish');assert actual.outcome==4
        rows.append({'guard':'protected_corruption','outcome':'forbidden'})
    with SqlEpisode(2,PROFILES[0],budget,{'guard':'scoped_undo'}) as actual:
        actual.step('sequential');assert actual.state()['alloc_a']==1
        actual.step('replenish_b');actual.step('undo')
        assert actual.state()['a']==2 and actual.state()['b']==1 and actual.added_b==1
        actual.step('atomic');actual.step('finish');assert actual.outcome==1
        rows.append({'guard':'scoped_undo_preserves_replenishment','outcome':'success'})
    # Public initial observations must not identify the sampled hidden world.
    observations=[]
    for world in range(6):
        with NativeEpisode(lib,world,PROFILES[3]) as native: observations.append(native.observe())
    assert all(o==observations[0] for o in observations)
    return rows


def run(output, references, debug_ledger=None):
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    records=[];status='running';error=None;baselines=[]
    budget=AttemptBudget(output/'attempts.jsonl',1424)
    debug_count=0
    if debug_ledger:
        debug_path=Path(debug_ledger)
        if debug_path.exists():
            debug_count=len(debug_path.read_text().splitlines())
            shutil.copyfile(debug_path,output/'prequalification-debug-attempts.jsonl')
    budget.maximum-=debug_count
    sources=list((ROOT/'puffer_lab').glob('*.py'))+list((ROOT/'puffer_lab').glob('*.h'))+list((ROOT/'puffer_lab').glob('*.c'))+[ROOT/'puffer_lab/reservation.ini',ROOT/'docs/puffer-reservation-v1-protocol.md']
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    reference_hashes={p.name:sha(p) for p in Path(references).iterdir() if p.is_file()}
    freeze={'schema':'puffer-reservation-qualification-v1','source_sha256':source_hashes,
            'puffer_revision':PUFFER_REVISION,'reference_sha256':reference_hashes,
            'profiles':PROFILES,'prefixes':PREFIXES,'actions':ACTIONS,
            'max_database_attempts_including_debug':1424,'prequalification_debug_attempts':debug_count,
            'planned_main_branches':1296,'sqlite_version':sqlite3.sqlite_version,
            'training_performed':False,'model_predictions':0}
    write(output/'freeze.json',freeze)
    try:
        binary=compile_core(output/'libreservation.so')
        lib=library(binary)
        adapter=output/'check-adapter'
        command=['cc','-std=c11','-O2','-Wno-unused-function','-Wno-unused-parameter',
                 '-I'+str(Path(references).resolve()),'-I'+str(ROOT/'puffer_lab'),
                 str(ROOT/'puffer_lab/check_adapter.c'),'-lm','-o',str(adapter)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        adapter_receipt=json.loads(subprocess.check_output([str(adapter.resolve())],text=True))
        write(output/'puffer-api-check.json',dict(adapter_receipt,binary_sha256=sha(adapter),
                                               header_sha256=reference_hashes['pufferenv.h']))
        with (output/'branches.jsonl').open('w') as handle:
            for profile in range(4):
                for world in range(6):
                    for prefix in PREFIXES:
                        for first in ACTIONS:
                            if time.monotonic()-started>1800 or handle.tell()>100_000_000:
                                raise RuntimeError('Qualification time/output cap reached')
                            before=branch(lib,profile,world,prefix,first,budget)
                            handle.write(canonical(dict(before,replay=0))+'\n');handle.flush()
                            repeated=branch(lib,profile,world,prefix,first,budget)
                            handle.write(canonical(dict(repeated,replay=1))+'\n');handle.flush()
                            if before != repeated: raise AssertionError('Replay differs')
                            records.append(before)
        guard_rows=guards(lib,budget);write(output/'guards.json',guard_rows)
        targets=conditional_targets(records)
        with (output/'conditional-targets.jsonl').open('w') as handle:
            for target in targets:handle.write(canonical(target)+'\n')
        for profile in range(4):
            initial=[t for t in targets if t['public_context']['profile']==profile and len(t['public_context']['history'])==1]
            values={t['action']:float(Fraction(*t['expected_return'])) for t in initial}
            top=max(values.values())
            baselines.append({'profile':PROFILES[profile]['name'],'expected_returns':values,
                              'best_first_actions':[k for k,v in values.items() if math.isclose(v,top,abs_tol=1e-9)]})
        if len({tuple(b['best_first_actions']) for b in baselines})<2:
            raise AssertionError('First-action ranking does not change across profiles')
        probabilities={tuple(t['success_probability']) for t in targets}
        if not any(0<n<d for n,d in probabilities): raise AssertionError('No uncertain success targets')
        status='qualified'
    except BaseException as exc:
        error={'type':type(exc).__name__,'detail':str(exc)}
        status='failed'
        raise
    finally:
        write(output/'summary.json',{'status':status,'error':error,'training_performed':False,
            'mechanisms':1,'public_profiles':4,'matched_replayed_branches':len(records),
            'database_attempts':budget.count,'debug_attempts':debug_count,
            'database_attempts_total':budget.count+debug_count,
            'sql_statements_primary_branches':sum(r['sql_statement_count'] for r in records),
            'primary_simulated_transitions':sum(len(r['steps']) for r in records),
            'baselines':baselines,'elapsed_seconds':time.monotonic()-started})
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--debug-ledger',type=Path)
    args=parser.parse_args()
    print(run(args.output,args.references,args.debug_ledger))
