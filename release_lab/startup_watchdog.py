"""Stop one explicitly named training rental if its startup guard is not disarmed.

Run on an independent hosted runner before allocating the GPU. The launcher
cancels this job only after it verifies the pod's own authenticated deadline
guard. This service never allocates, resumes, or deletes a pod.
"""
import argparse
import json
import os
import re
import time
import urllib.request


def provider(method, path):
    token = os.environ['RUNPOD_WATCHDOG_TOKEN']
    request = urllib.request.Request('https://rest.runpod.io/v1'+path, method=method,
        headers={'Authorization': 'Bearer '+token, 'User-Agent': 'first-instinct-startup-guard'})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read()
        return json.loads(body) if body else {}


def validate(name, seconds):
    if not re.fullmatch(r'first-instinct-decision-supervision-v1-[a-z0-9]{8}', name):
        raise ValueError('Only a unique decision-comparison rental may be watched')
    if type(seconds) is not int or not 10 <= seconds <= 1200:
        raise ValueError('Startup watch must be between 10 and 1200 seconds')


def watch(name, seconds, *, call=provider, now=time.monotonic, sleep=time.sleep):
    validate(name, seconds)
    deadline = now()+seconds
    owned = None
    while True:
        try:
            matches = [p for p in call('GET', '/pods') if p.get('name') == name]
            if len(matches) > 1: raise ValueError('Ambiguous rental name')
            if matches:
                pod = matches[0]
                if owned is not None and owned != pod['id']: raise ValueError('Rental identity changed')
                owned = pod['id']
                if not re.fullmatch(r'[a-z0-9]+', owned): raise ValueError('Invalid provider identity')
                if pod['desiredStatus'] in ('EXITED', 'STOPPED', 'TERMINATED'):
                    return dict(status='already_stopped', pod_id=owned)
            elif owned is not None:
                return dict(status='already_absent', pod_id=owned)
        except (OSError, json.JSONDecodeError):
            # A transient provider failure must not silently disarm the deadline.
            if now() >= deadline and owned is None: raise
        if now() >= deadline: break
        sleep(min(15, max(0, deadline-now())))
    if owned is None: return dict(status='expired_without_rental')
    for attempt in range(8):
        try:
            call('POST', '/pods/'+owned+'/stop')
            return dict(status='stop_requested', pod_id=owned)
        except OSError:
            if attempt == 7: raise
            sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authenticate', action='store_true')
    parser.add_argument('--name'); parser.add_argument('--seconds', type=int)
    args = parser.parse_args()
    if args.authenticate:
        if not isinstance(provider('GET', '/pods'), list): raise ValueError('Unrecognized provider response')
        print('Authenticated startup watchdog; no provider changes made.', flush=True)
    else:
        print(json.dumps(watch(args.name, args.seconds)), flush=True)
