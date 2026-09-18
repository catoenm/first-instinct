"""Explore verified software evidence with a resident language-model forecaster.

The human can select inspections or follow a separately trained small inspector.
The language forecaster was not reinforcement trained. No uploaded code executes.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

from .build import read_rows
from .environment import Environment, INSPECTIONS, MASKS, REPORTS
from .export import convert

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).with_name('web')
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}


def value_text(value):
    """Display typed execution values without losing large integer precision in JavaScript."""
    kind, data = value
    if kind in ('NoneType', 'bool', 'int', 'float', 'str'):
        return repr(data)
    if kind == 'dict':
        return '{' + ', '.join(value_text(k) + ': ' + value_text(v) for k, v in data) + '}'
    if kind == 'set':
        return '{' + ', '.join(value_text(v) for v in data) + '}' if data else 'set()'
    if kind == 'tuple':
        return '(' + ', '.join(value_text(v) for v in data) + (',' if len(data) == 1 else '') + ')'
    if kind == 'list':
        return '[' + ', '.join(value_text(v) for v in data) + ']'
    raise ValueError('Unsupported execution value')


def adjusted_probability(probability, temperature):
    if not 0 <= probability <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid probability or temperature')
    p = min(1 - 1e-7, max(1e-7, probability))
    logit = (math.log(p) - math.log1p(-p)) / temperature
    return 1 / (1 + math.exp(-max(-700, min(700, logit))))


class Demo:
    def __init__(self, predictor, rows, temperature=1., maximum_cases=24, selector_path=None):
        self.predictor, self.temperature = predictor, float(temperature)
        adjusted_probability(.5, self.temperature)
        self.cases, self.sessions = {}, {}
        tasks = set()
        # Outcome-blind ordering, one variant per held-out function. All seven
        # views must fit, so an episode cannot fail halfway through on length.
        ordered = sorted((r for r in rows if r['split'] == 'test'),
                         key=lambda r: hashlib.sha256(('browser-v1:' + r['id']).encode()).hexdigest())
        for row in ordered:
            if row['task_id'] in tasks:
                continue
            try:
                for mask in MASKS:
                    predictor.validate(convert(row, mask)['input'])
            except ValueError:
                continue
            self.cases[row['id']] = row
            tasks.add(row['task_id'])
            if len(self.cases) >= maximum_cases:
                break
        if not self.cases:
            raise ValueError('No complete held-out episodes fit the model context')
        self.selector = None
        if selector_path is not None:
            from safetensors.torch import load_file
            from .features import Pool
            from .hybrid import Selector
            self.pool = Pool(list(self.cases.values()))
            self.case_indices = {key: i for i, key in enumerate(self.cases)}
            self.selector = Selector(self.pool.dimensions)
            self.selector.load_state_dict(load_file(selector_path))
            self.selector.eval()

    def catalog(self):
        return {'model': self.predictor.spec['id'], 'temperature': self.temperature,
                'cases': [{'id': key, 'function': row['function'], 'summary': ' '.join(row['description'].split())[:90]}
                          for key, row in self.cases.items()],
                'note': 'You choose inspections. The language model forecasts the fixed-suite outcome.'}

    def recommendation(self, state):
        if self.selector is None:
            return None
        import numpy as np
        import torch
        from .environment import legal_actions
        from .train import tensor
        env = state['environment']
        masks = np.array([env.mask])
        features = self.pool.features(np.array([self.case_indices[state['row']['id']]]), masks,
                                      np.array([env.costs], dtype=np.float32))
        with torch.no_grad():
            probabilities = self.selector.distribution(tensor(features), tensor(legal_actions(masks))).probs[0].tolist()
        actions = ('finish', *INSPECTIONS)
        return {'choice': actions[max(range(4), key=probabilities.__getitem__)],
                'probabilities': dict(zip(actions, probabilities)),
                'note': 'Separate small inspector, frozen-forecaster training condition, seed 17. Most likely action; published evaluation integrates all action paths.'}

    def forecast(self, row, mask):
        result = self.predictor.predict(convert(row, mask)['input'])
        raw = result['probabilities']['passes']
        p = adjusted_probability(raw, self.temperature)
        report = min(range(len(REPORTS)), key=lambda i: abs(float(REPORTS[i]) - p))
        return {'raw_probability': raw, 'probability': p, 'report_index': report,
                'report_probability': round(float(REPORTS[report]), 2),
                'milliseconds': result['milliseconds'], 'input_tokens': result['input_tokens'],
                'mask': mask}

    def start(self, candidate_id):
        if candidate_id not in self.cases:
            raise ValueError('Choose a listed candidate')
        row = self.cases[candidate_id]
        forecast = self.forecast(row, 0)
        session = secrets.token_urlsafe(24)
        if len(self.sessions) >= 64:
            del self.sessions[next(iter(self.sessions))]
        self.sessions[session] = {'row': row, 'environment': Environment(row),
                                  'history': [forecast], 'finished': False}
        return self.public(session)

    def public(self, session):
        state = self.sessions[session]
        observation = state['environment'].observe()
        display = [{'source': item['source'], 'checks': [
            {'call': check['call'], 'passed': check['passed'], 'expected': value_text(check['expected']),
             'actual': value_text(check['actual']['value']) if 'value' in check['actual']
                       else 'Raised ' + check['actual'].get('exception', 'an execution error')}
            for check in item['checks']]} for item in observation['evidence']]
        return {'session': session, 'observation': observation, 'evidence_display': display,
                'recommendation': self.recommendation(state),
                'history': state['history'], 'total_cost': state['environment'].total_cost,
                'finished': False}

    def step(self, session, action):
        if session not in self.sessions or self.sessions[session]['finished']:
            raise ValueError('Start a new episode')
        state = self.sessions[session]
        env = state['environment']
        if action == 'policy':
            recommendation = self.recommendation(state)
            if recommendation is None:
                raise ValueError('No trained inspector is loaded')
            action = recommendation['choice']
        if action == 'finish':
            _, reward, _, info = env.step(state['history'][-1]['report_index'])
            state['finished'] = True
            return {'session': session, 'finished': True, 'history': state['history'],
                    'total_cost': env.total_cost, 'outcome': info['realized_outcome'],
                    'reported_probability': info['reported_probability'],
                    'reward': reward - env.total_cost}
        if action not in INSPECTIONS:
            raise ValueError('Unknown inspection')
        proposed = copy.deepcopy(env)
        proposed.step(21 + INSPECTIONS.index(action))
        forecast = self.forecast(state['row'], proposed.mask)
        state['environment'] = proposed
        state['history'].append(forecast)
        return self.public(session)


def handler_for(demo):
    gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return (self.headers.get('Host') in allowed and
                    (origin is None or origin in {'http://' + host for host in allowed}))

        def do_GET(self):
            if not self.local_request():
                self.reply(403, {'error': 'Use the local demo address'})
                return
            path = urlsplit(self.path).path
            if path in ASSETS:
                name, kind = ASSETS[path]
                self.reply(200, (WEB / name).read_bytes(), kind)
            elif path == '/api/catalog':
                self.reply(200, demo.catalog())
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.local_request():
                self.reply(403, {'error': 'Use the local demo address'})
                return
            if self.path not in ('/api/start', '/api/step'):
                self.reply(404, {'error': 'Not found'})
                return
            if self.headers.get_content_type() != 'application/json':
                self.reply(415, {'error': 'Send application/json'})
                return
            try:
                self.connection.settimeout(10)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Request is too large or empty')
                item = json.loads(self.rfile.read(size))
                if not isinstance(item, dict):
                    raise ValueError('Expected an object')
                required = ('id',) if self.path == '/api/start' else ('session', 'action')
                if any(not isinstance(item.get(key), str) for key in required):
                    raise ValueError('Missing request fields')
            except (ValueError, UnicodeError, TimeoutError) as error:
                self.reply(400, {'error': str(error)})
                return
            if not gate.acquire(blocking=False):
                self.reply(429, {'error': 'The model is busy. Try again shortly.'})
                return
            try:
                result = demo.start(item['id']) if self.path == '/api/start' else demo.step(item['session'], item['action'])
                self.reply(200, result)
            except ValueError as error:
                self.reply(400, {'error': str(error)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.reply(500, {'error': 'Prediction failed. Check the local server log.'})
            finally:
                gate.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'output/pretrained/first-instinct-software-outcome-v1/model')
    parser.add_argument('--data', type=Path, default=ROOT / 'output/pretrained/first-instinct-software-inspection-v1/curated')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--manual-only', action='store_true', help='Run without the small trained inspection policy')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use a port between 1024 and 65535')
    from scale_lab.infer import Predictor
    predictor = Predictor(run=args.run, device=args.device)
    calibration = args.run / 'calibration.json'
    temperature = json.loads(calibration.read_text())['temperature'] if calibration.exists() else 1.
    selector_path = None if args.manual_only else args.data.parent / 'hybrid/frozen-s17/selector.safetensors'
    demo = Demo(predictor, read_rows(args.data / 'cases.jsonl.gz'), temperature, selector_path=selector_path)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(demo))
    print(f'Inspection demo ready at http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
