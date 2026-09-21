"""Exercise the real localhost Mac API, memory bound and direct-output parity."""
import argparse
import copy
from http.client import HTTPConnection
import json
from pathlib import Path
import resource
import threading
import time

from release_lab.typed_interface import requests
from general_lab.serve import ThreadingHTTPServer, handler_for
from release_lab.mlx_api import MacDemo, MacPredictor
from scale_lab.common import digest, encode, file_hash, write_json


class CountedPredictor(MacPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.inputs = set()

    def predict(self, item):
        self.calls += 1
        self.inputs.add(digest(self.validate(item)))
        return super().predict(item)


def run(package, qualification, output):
    import mlx.core as mx
    if output.exists():
        raise ValueError('Preserve earlier serving checks')
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path('release_lab/mlx_api.py'), Path('release_lab/mlx_package.py'),
             Path('release_lab/mlx_scorer.py'), Path('general_lab/serve.py'),
             Path('general_lab/interface.py'), Path('scale_lab/common.py'),
             Path('release_lab/typed_interface.py'),
             Path('docs/mlx-api-v1-protocol.md'), Path('examples/general-decisions.json'),
             package/'package.json', qualification]
    plan = dict(version='mlx-api-v1', paths={str(p.resolve()):file_hash(p) for p in files},
                max_seconds=1200, probability_tolerance=1e-5, peak_active_bytes=10*1024**3)
    write_json(output/'plan-private.json', plan)
    start = time.monotonic(); server = None; thread = None
    records = []; checks = {}; differences = []
    try:
        predictor = CountedPredictor(package, qualification)
        demo = MacDemo(predictor)
        server = ThreadingHTTPServer(('127.0.0.1',0), handler_for(demo))
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.1}, daemon=True)
        thread.start()
        mx.clear_cache(); mx.reset_peak_memory()

        def http(method, path, payload=None):
            if time.monotonic()-start > 1200:
                raise TimeoutError('Serving check deadline')
            conn = HTTPConnection('127.0.0.1',port,timeout=60)
            headers = {}
            body = None if payload is None else json.dumps(payload)
            if body is not None:
                headers = {'Content-Type':'application/json','X-CSRF-Token':demo.csrf_token}
            began = time.perf_counter()
            try:
                conn.request(method,path,body=body,headers=headers)
                response = conn.getresponse(); data = response.read()
                return response.status, json.loads(data) if path.startswith('/api/') else data, time.perf_counter()-began
            finally:
                conn.close()

        status, data, _ = http('GET','/api/status')
        checks['metadata'] = (status == 200 and data['checkpoint']['step'] == 2742
                              and data['limits']['max_tokens'] == 4096
                              and data['limits']['max_total_tokens'] == 8192)
        status, data, _ = http('GET','/')
        checks['page'] = status == 200 and b'First Instinct' in data

        def valid(name, payload):
            status, result, elapsed = http('POST','/api/answer',payload)
            if status != 200:
                raise ValueError('Valid request refused: '+name)
            with demo.gate:
                direct = demo.answer(payload)
            for identity in result['answers']:
                a,b = result['answers'][identity],direct['answers'][identity]
                differences.extend(abs(p-b['probabilities'][k]) for k,p in a['probabilities'].items())
                for key in ('choice','selected_level','legend','type'):
                    if a.get(key) != b.get(key):
                        raise ValueError('Answer contract differs: '+name)
            records.append(dict(name=name,payload=payload,result=result,seconds=elapsed))
            write_json(output/'requests-private.json',records)
            return result['answers']

        def refused(name, payload):
            with demo.gate:
                before = predictor.calls
            status, result, elapsed = http('POST','/api/answer',payload)
            with demo.gate:
                unchanged = before == predictor.calls
            checks[name] = status == 400 and unchanged
            records.append(dict(name=name,status=status,zero_model_calls=unchanged,seconds=elapsed))
            write_json(output/'requests-private.json',records)

        original = json.loads(Path('examples/general-decisions.json').read_text())
        a = valid('choice_binary_score',original)
        checks['original_example'] = (a['next_action']['choice'] == 'expedite'
                                      and a['eligible']['probability_yes'] > .5
                                      and a['delay_level']['selected_level'] == 2)
        changed = copy.deepcopy(original); changed['state']['parcel']['days_late'] = 1
        b = valid('changed_observation',changed)
        checks['changed_example'] = (b['next_action']['choice'] == 'monitor'
                                     and b['eligible']['probability_yes'] < .5
                                     and b['delay_level']['selected_level'] == 1)
        single = copy.deepcopy(original)
        single['questions'] = {'next_action':single['questions']['next_action']}
        c = valid('independent_question',single)
        checks['independent_question'] = all(abs(p-c['next_action']['probabilities'][k]) <= 1e-5
                                            for k,p in a['next_action']['probabilities'].items())
        repeated = valid('repeat',original)
        checks['repeatable'] = all(abs(p-repeated[name]['probabilities'][key]) <= 1e-5
                                   for name,answer in a.items() for key,p in answer['probabilities'].items())
        many = {'state':'The desired item number is 35.','questions':{'item':{
            'type':'choice','instructions':'Select the desired item number.',
            'criteria':{f'item_{i}':f'Item number {i}' for i in range(36)}}}}
        values = valid('all_36_native_labels',many)
        checks['36_options'] = len(values['item']['probabilities']) == 36

        long_payload = {'state':'x ','questions':{'nonempty':{'type':'binary','instructions':'Is the state nonempty?'}}}
        # Repeated one-token text creates an exact-limit memory probe; it has no
        # benchmark truth label and is excluded from every quality aggregate.
        count = 3800
        for _ in range(8):
            long_payload['state'] = 'x '*count
            item = requests(long_payload)[0][2]
            length = len(encode(predictor.tokenizer,item,8192))
            if length == 4096:
                break
            count += 4096-length
        else:
            raise ValueError('Could not construct complete 4096-token probe')
        valid('complete_4096_tokens',long_payload)
        checks['max_context'] = length == 4096
        over = copy.deepcopy(long_payload); over['state'] += 'x '
        refused('overlength_rejected_before_inference',over)
        too_many = copy.deepcopy(single)
        too_many['questions'] = {str(i):single['questions']['next_action'] for i in range(5)}
        refused('question_count_rejected_before_inference',too_many)
        too_much = copy.deepcopy(long_payload)
        too_much['questions'] = {str(i):long_payload['questions']['nonempty'] for i in range(3)}
        refused('aggregate_budget_rejected_before_inference',too_much)

        checks['probability_parity'] = max(differences) <= 1e-5
        peak = mx.get_peak_memory()
        checks['active_memory'] = peak <= 10*1024**3
        report = dict(status='localhost_serving_passed' if all(checks.values()) else 'localhost_serving_failed',
                      checks=checks, max_probability_delta=max(differences), model_calls=predictor.calls,
                      unique_tokenized_questions=len(predictor.inputs), http_answer_requests=len(records),
                      peak_active_inference_bytes=peak,active_model_bytes=mx.get_active_memory(),
                      cache_bytes=mx.get_cache_memory(),process_peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      total_seconds=time.monotonic()-start,package_sha256=file_hash(package/'package.json'),
                      qualification_sha256=file_hash(qualification),plan_sha256=file_hash(output/'plan-private.json'),
                      new_training_presentations=0,optimizer_updates=0,
                      limitation='One local M5 Max with 128 GB. Not a public deployment or a Mac mini measurement.')
        write_json(output/'report.json',report)
        return report
    except BaseException as error:
        write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),checks=checks,
                                               elapsed_seconds=time.monotonic()-start))
        raise
    finally:
        if server is not None:
            server.shutdown();server.server_close()
        if thread is not None:
            thread.join(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package','qualification','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)),indent=2))
