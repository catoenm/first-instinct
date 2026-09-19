"""Bounded resident-model probe on opened inspect-then-act engineering fixtures."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import urllib.request

from scale_lab.common import digest, file_hash, read_rows, write_json
from tool_lab.evidence_shell import HORIZON, forecast_input, run_episode


class ResidentModel:
    def __init__(self):
        self.base = 'http://127.0.0.1:8766'
        status = json.load(urllib.request.urlopen(self.base + '/api/status', timeout=10))
        self.csrf = status.pop('csrf_token')
        if (not status['ready'] or status['model'] != 'Qwen/Qwen3.5-9B'
                or status['checkpoint']['step'] != 2742 or status['device'] != 'mps'
                or status['model_revision'] != 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'):
            raise ValueError('Probe requires the original supervised resident 9B model')
        self.metadata = {k: status[k] for k in ('model', 'model_revision', 'checkpoint', 'device')}
        self.calls = self.questions = 0

    def predict(self, items):
        states = {item['state'] for item in items.values()}
        if len(states) != 1:
            raise ValueError('A request shares exactly one observed state')
        self.calls += 1; self.questions += len(items)
        if self.calls > 60 or self.questions > 84:
            raise ValueError('Declared resident-model probe budget exceeded')
        payload = dict(state=next(iter(states)), questions={key:dict(type='choice', instructions=item['question'],
                       criteria={o['id']:o['description'] for o in item['options']}) for key,item in items.items()})
        request = urllib.request.Request(self.base + '/api/answer', data=json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json','X-CSRF-Token':self.csrf}, method='POST')
        response = json.load(urllib.request.urlopen(request, timeout=120))
        if any(response[k] != v for k,v in self.metadata.items()):
            raise ValueError('Model metadata changed during the probe')
        return response['answers']


def probe(data, output):
    output.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((data / 'freeze.json').read_text())
    qualification = json.loads((data / 'report.json').read_text())
    if (qualification['status'] != 'qualified' or qualification['episodes'] != 72
            or file_hash(data / 'episodes.jsonl') != qualification['episodes_sha256']
            or file_hash(data / 'private-fixtures.jsonl') != freeze['fixtures_sha256']
            or file_hash(Path(__file__).with_name('evidence_shell.py')) != freeze['source_sha256']):
        raise ValueError('Qualified environment evidence changed')
    cases = read_rows(data / 'private-fixtures.jsonl')
    if len(cases) != 12:
        raise ValueError('Exactly twelve engineering contexts were declared')
    controls = {(r['fixture_id'],r['policy']):r for r in read_rows(data / 'episodes.jsonl')}
    model = ResidentModel()
    receipt = dict(status='running', started_at=time.time(), model=model.metadata,
                   source_sha256=file_hash(Path(__file__)), qualification_sha256=file_hash(data / 'report.json'),
                   maximum_model_episodes=12, maximum_action_predictions=12*HORIZON,
                   maximum_forecast_predictions=48, maximum_http_requests=60,
                   scope='Opened engineering fixtures, original resident MPS model. No training or cloud rental; no in-memory parameter attestation.')
    write_json(output / 'receipt.json', receipt)
    episodes, forecasts = [], []
    try:
        with (output / 'episodes.jsonl').open('w') as episode_stream, (output / 'forecasts.jsonl').open('w') as forecast_stream:
            for case in cases:
                def policy(item, menu):
                    return model.predict({'action':item})['action']['probabilities']
                result = run_episode(case, policy, policy_name='original-9b-supervised-greedy')
                episode_stream.write(json.dumps(result,allow_nan=False)+'\n');episode_stream.flush();episodes.append(result)
                reference = controls[case['id'],'evidence_reference']
                wrong = controls[case['id'],'wrong_after_evidence']
                for phase,index in [('before_inspection',0),('after_inspection',1)]:
                    event = reference['events'][index]
                    candidates = [(k,a) for k,a in event['menu'].items() if a['kind']=='repair']
                    items = {k:forecast_input(event['input'],a) for k,a in candidates}
                    predictions = model.predict(items)
                    for key,action in candidates:
                        if phase == 'before_inspection':
                            label_receipt = controls[case['id'],'blind_left' if action['side']==0 else 'blind_right']
                        else:
                            label_receipt = reference if reference['events'][-1]['choice']==key else wrong
                        executed = label_receipt['events'][-1]
                        if executed['menu'][executed['choice']]['command'] != action['command']:
                            raise ValueError('Forecast label not bound to the executed candidate')
                        if forecast_input(executed['input'], executed['menu'][executed['choice']]) != items[key]:
                            raise ValueError('Forecast evidence differs from the executed branch observation')
                        probabilities = predictions[key]['probabilities']
                        if (set(probabilities) != {'yes','no'}
                                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                                or abs(sum(probabilities.values())-1) > 1e-5):
                            raise ValueError('Invalid forecast distribution')
                        row = dict(fixture_id=case['id'], bundle_id=case['bundle_id'], family=case['family'], phase=phase,
                                   input=items[key], input_sha256=digest(items[key]), actual_success=label_receipt['success'],
                                   probability_yes=predictions[key]['probabilities']['yes'], answer=predictions[key])
                        forecast_stream.write(json.dumps(row,allow_nan=False)+'\n');forecast_stream.flush();forecasts.append(row)
                print(json.dumps(dict(completed_episodes=len(episodes),http_requests=model.calls,typed_questions=model.questions)),flush=True)
        by_phase = {}
        for phase in ('before_inspection','after_inspection'):
            rows = [r for r in forecasts if r['phase']==phase]
            by_phase[phase] = dict(n=len(rows), brier=sum((r['probability_yes']-r['actual_success'])**2 for r in rows)/len(rows),
                                  accuracy=sum((r['probability_yes']>=.5)==r['actual_success'] for r in rows)/len(rows))
        grouped = defaultdict(list)
        for row in forecasts:
            if row['phase']=='before_inspection':grouped[row['input_sha256']].append(row)
        if len(grouped)!=12 or any(sorted(r['actual_success'] for r in rows)!=[False,True] for rows in grouped.values()):
            raise ValueError('Paired initial uncertainty labels lost')
        report = dict(status='complete', episodes=len(episodes), successes=sum(r['success'] for r in episodes),
                      mean_reward=sum(r['reward'] for r in episodes)/len(episodes),
                      by_family={f:dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
                          first_actions=[r['events'][0]['menu'][r['events'][0]['choice']]['kind'] for r in rows])
                          for f in sorted({r['family'] for r in episodes}) for rows in [[r for r in episodes if r['family']==f]]},
                      forecasts=by_phase,
                      initial_mean_absolute_distance_from_known_half_probability=sum(abs(r['probability_yes']-.5) for rows in grouped.values() for r in rows)/24,
                      prior_only_optimal_brier=.25, fully_observed_deterministic_oracle_brier=0.,
                      http_requests=model.calls, typed_questions=model.questions,
                      note='Twelve opened engineering contexts, not a held-out generalization benchmark. Forecasts use actual executed candidate branches. No model updates.')
        write_json(output / 'report.json', report)
        receipt.update(status='complete', completed_at=time.time(), report_sha256=file_hash(output / 'report.json'),
                       episodes_sha256=file_hash(output / 'episodes.jsonl'), forecasts_sha256=file_hash(output / 'forecasts.jsonl'))
    except BaseException as error:
        receipt.update(status='failed', completed_at=time.time(), error=dict(type=type(error).__name__,detail=str(error)))
        raise
    finally:
        receipt.update(http_requests=model.calls, typed_questions=model.questions)
        write_json(output / 'receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();probe(args.data,args.output)
