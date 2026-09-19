"""Reconstruct public inputs, rewards and executed forecast labels offline."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from scale_lab.common import digest, file_hash, read_rows, write_json
from tool_lab.contextual_shell_audit import public_expected
from tool_lab.evidence_shell import COMMAND_COST, SESSION_WORKER, actions, decision_input, forecast_input
from tool_lab.protocol import choose
from tool_lab.shell_supervision import verify


def audit(data, probe):
    freeze = json.loads((data/'freeze.json').read_text())
    receipt = json.loads((probe/'receipt.json').read_text())
    control_report = json.loads((data/'report.json').read_text())
    model_report = json.loads((probe/'report.json').read_text())
    if (receipt['status'] != 'complete' or file_hash(data/'private-fixtures.jsonl') != freeze['fixtures_sha256']
            or file_hash(Path(__file__).with_name('evidence_shell.py')) != freeze['source_sha256']
            or hashlib.sha256(SESSION_WORKER.encode()).hexdigest() != freeze['worker_sha256']
            or file_hash(Path(__file__).with_name('evidence_shell_probe.py')) != receipt['source_sha256']):
        raise ValueError('Source, input or completion identity changed')
    for path, checksum in [(data/'episodes.jsonl', control_report['episodes_sha256']),
                           (data/'report.json', receipt['qualification_sha256']),
                           *[(probe/name, receipt[name.replace('.jsonl','').replace('.json','')+'_sha256'])
                             for name in ('episodes.jsonl','forecasts.jsonl','report.json')]]:
        if file_hash(path) != checksum:raise ValueError('Changed recorded evidence: '+str(path))
    fixtures = {c['id']:c for c in read_rows(data/'private-fixtures.jsonl')}
    controls = read_rows(data/'episodes.jsonl'); models = read_rows(probe/'episodes.jsonl')
    branches = {}; decisions = 0
    for trace in controls+models:
        case = fixtures[trace['fixture_id']]
        if case['expected'] != public_expected(case):raise ValueError('Semantic target changed')
        history = []; count = 0
        for i,event in enumerate(trace['events']):
            menu = actions(case,i); item = decision_input(case['goal'],history,i,menu)
            if menu != event['menu'] or item != event['input'] or digest(item) != event['input_sha256']:
                raise ValueError('Public observation/menu identity differs')
            selected = choose(event['behavior_probabilities'], menu, 'argmax', None)
            if selected != event['choice'] or not math.isclose(math.log(event['behavior_probabilities'][selected]),event['chosen_log_probability'],abs_tol=1e-12):
                raise ValueError('Selected action or likelihood differs')
            action = menu[selected]
            if action['command'] is not None:
                history.append(dict(command=action['command'],**event['observation'])); count += 1
            elif event['observation'] is not None:raise ValueError('Finish cannot have command output')
            if event['done'] != (i==len(trace['events'])-1):raise ValueError('Nonterminal or extra trajectory step')
            expected_reward = (float(trace['success']) if event['done'] else 0.) - (COMMAND_COST if action['command'] else 0.)
            if abs(event['reward']-expected_reward)>1e-12:raise ValueError('Reward differs')
            if abs(event['undiscounted_return']-sum(e['reward'] for e in trace['events'][i:]))>1e-12:raise ValueError('Return differs')
            if event['done'] and action['kind']=='repair' and trace['policy'] in ('blind_left','blind_right','evidence_reference','wrong_after_evidence'):
                key=(case['id'],digest(forecast_input(item,action)))
                if key in branches and branches[key]!=trace['success']:raise ValueError('Conflicting executed branch')
                branches[key]=trace['success']
            decisions += 1
        if (verify(case,trace['private_verifier_evidence'])!=trace['success'] or count!=trace['commands']
                or abs(trace['reward']-sum(e['reward'] for e in trace['events']))>1e-12):
            raise ValueError('Independent verification or episode accounting differs')
    forecasts = read_rows(probe/'forecasts.jsonl'); grouped=defaultdict(list)
    for row in forecasts:
        key=(row['fixture_id'],digest(row['input']))
        if row['input_sha256']!=key[1] or branches.get(key) is not row['actual_success']:
            raise ValueError('Forecast lacks matching executed evidence')
        grouped[row['phase']].append(row)
    for phase,rows in grouped.items():
        brier=sum((r['probability_yes']-r['actual_success'])**2 for r in rows)/len(rows)
        accuracy=sum((r['probability_yes']>=.5)==r['actual_success'] for r in rows)/len(rows)
        if abs(brier-model_report['forecasts'][phase]['brier'])>1e-12 or accuracy!=model_report['forecasts'][phase]['accuracy']:
            raise ValueError('Forecast metrics differ')
    if len(models)!=model_report['episodes'] or sum(r['success'] for r in models)!=model_report['successes']:
        raise ValueError('Model episode metrics differ')
    return dict(status='passed',control_episodes=len(controls),model_episodes=len(models),
                public_decisions_reconstructed=decisions,forecasts_bound_to_executed_branches=len(forecasts),
                independent_terminal_verifications=len(controls)+len(models),
                source_sha256=file_hash(Path(__file__)),
                note='Offline audit. No new commands, model calls, training, or retries.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();write_json(args.output,audit(args.data,args.probe))
