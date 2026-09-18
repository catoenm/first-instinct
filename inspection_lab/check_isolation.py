"""Run harmless containment probes before executing the source corpus."""
import argparse
from pathlib import Path
import tempfile

from .build import write_json
from .sandbox import IMAGE, Worker


def check():
    with tempfile.TemporaryDirectory(prefix='first-instinct-isolation-') as temp:
        canary = Path(temp) / 'nonsecret-canary.txt'; canary.write_text('synthetic containment test')
        with Worker() as worker:
            uid = worker.run('import os\ndef f(): return os.getuid()', [{'call': 'f()'}])
            network = worker.run('import socket\ndef f(): return socket.create_connection(("1.1.1.1",443),timeout=.2)', [{'call': 'f()'}])
            root_write = worker.run('def f(): return open("/first-instinct-test", "w").write("test")', [{'call': 'f()'}])
            host_read = worker.run('def f(): return open(' + repr(str(canary)) + ').read()', [{'call': 'f()'}])
            bounded = worker.run('def f():\n while True: pass', [{'call': 'f()'}])
            first = worker.run('def f(): return hash("seed-check")', [{'call': 'f()'}], 17)
            second = worker.run('def f(): return hash("seed-check")', [{'call': 'f()'}], 91)
    assertions = {'nonroot': uid == {'results': [{'value': ['int', 65534]}]},
                  'network_unavailable': 'exception' in network['results'][0],
                  'root_write_denied': 'exception' in root_write['results'][0],
                  'host_canary_unavailable': 'exception' in host_read['results'][0],
                  'infinite_loop_terminated': 'worker_error' in bounded,
                  'distinct_python_hash_seeds': first != second}
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return {'image': IMAGE, 'checks': assertions, 'scope': 'Harmless probes, not a proof of security against hostile code.'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    result = check(); write_json(a.output, result); print(result)


if __name__ == '__main__':
    main()
