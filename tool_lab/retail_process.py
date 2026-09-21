"""Isolated local retail worker; only public observations cross the actor API."""
import argparse
import builtins
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import types
from unittest.mock import patch

from tool_lab.telecom_local import canonical, digest, read, sha


def worker(plan_path, source_path, fees, journal):
    from tool_lab.telecom_hidden_causes import verify_sources
    from tool_lab.retail_live import RetailEpisode
    from tool_lab.retail_evidence_policy import EXPECTED_ERRORS, READS
    from tool_lab.retail_live_qualify import FEES
    plan = read(plan_path)
    verify_sources(plan)
    source_path = source_path.resolve()
    audit = read(Path(plan['source']) / 'audit.json')
    if (plan['candidate_role'] != 'training_candidate' or tuple(fees) not in FEES or
            source_path.parent != Path(plan['source']).resolve() or
            audit['receipt_hashes'].get(source_path.name) != sha(source_path) or 'tau2' in sys.modules):
        raise ValueError('Unqualified reset or cost schedule')
    source = read(source_path)
    journal.touch(exist_ok=False)
    def log(event):
        with journal.open('a') as stream:
            stream.write(canonical(event)+'\n')
    def send(message):
        print(canonical(message), flush=True)
    os.environ.update(PYTHON_DOTENV_DISABLED='1', LITELLM_LOCAL_MODEL_COST_MAP='True',
                      TAU2_DATA_DIR=str(Path(plan['upstream'])/'data'))
    namespace = types.ModuleType('tau2')
    namespace.__path__ = [str(Path(plan['upstream'])/'src/tau2')]
    sys.modules['tau2'] = namespace
    original_open, denied, data_reads = builtins.open, [], []
    data_root = (Path(plan['upstream'])/'data').resolve()
    def block(*args, **kwargs):
        denied.append('network')
        raise PermissionError('No outbound calls')
    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if path.is_relative_to(data_root):
                if str(path) != plan['allowed_data_file']:
                    denied.append('upstream_data')
                    raise PermissionError('Official task/database reads denied')
                data_reads.append(str(path))
        return original_open(file, *args, **kwargs)
    with ExitStack() as stack:
        for target, method in ((socket.socket,'connect'),(socket.socket,'connect_ex'),(socket,'create_connection')):
            stack.enter_context(patch.object(target,method,block))
        stack.enter_context(patch.object(builtins,'open',guarded_open))
        stack.enter_context(patch.object(io,'open',guarded_open))
        with redirect_stdout(sys.stderr):
            from tau2.domains.retail.data_model import RetailDB
            from tau2.domains.retail.environment import get_environment
            from loguru import logger
            logger.remove()
            env = get_environment(db=RetailDB.model_validate(copy.deepcopy(source['initial'])))
        def state():
            return env.tools.db.model_dump(mode='json')
        events=[]
        def invoke(name, arguments):
            if len(events)>=6 or not env.tools.has_tool(name):
                raise ValueError('Tool call ceiling or scope')
            before=state()
            log(dict(status='started',index=len(events),tool=name,arguments=arguments,before_sha256=digest(before)))
            error=None
            try:
                with redirect_stdout(sys.stderr):
                    value=env.make_tool_call(name,requestor='assistant',**arguments)
                    env.sync_tools()
                    response=env.to_json_str(value)
            except Exception as exc:
                if type(exc) is not ValueError or str(exc) not in EXPECTED_ERRORS.get(name,set()):
                    log(dict(status='infrastructure_error',error_type=type(exc).__name__))
                    raise
                error={'error_type':'ValueError','message':str(exc)}
                response=error
            after=state()
            is_read=env.tools.tool_type(name).value=='read'
            if is_read!=(name in READS) or ((is_read or error is not None) and before!=after):
                raise ValueError('Unexpected tool type or mutation')
            event=dict(tool=name,arguments=arguments,response=response,read_only=is_read,expected_error=error is not None,
                       before_sha256=digest(before),after_sha256=digest(after))
            events.append(event)
            log(dict(status='completed',index=len(events)-1,event=event))
            return response
        episode=RetailEpisode(source['task'],source['visible'],source['initial'],invoke,state,*fees)
        send({'observation':episode.observation()})
        done=False
        for request_number in range(7):
            line=sys.stdin.readline()
            if not line:
                raise RuntimeError('Controller disconnected before receipt recovery')
            request=json.loads(line)
            if set(request)=={'action'} and not done:
                delivery=episode.step(request['action'])
                done=delivery['done']
                log(dict(status='delivery',delivery=delivery))
                send(delivery)
            elif request=={'receipt':True} and done:
                imported={}
                for name,module in sys.modules.items():
                    if name.startswith('tau2.') and getattr(module,'__file__',None):
                        path=str(Path(module.__file__).resolve())
                        if path not in plan['paths'] or sha(path)!=plan['paths'][path]:
                            raise ValueError('Unfrozen simulator import')
                        imported[name]=sha(path)
                if denied or data_reads!=[plan['allowed_data_file']]:
                    raise ValueError('Data or network scope escaped')
                receipt=episode.private_record()
                receipt.update(events=events,source_receipt=str(source_path),source_sha256=sha(source_path),
                               imported_sources=imported,guards=dict(outbound_attempts=0,official_task_reads=0))
                log(dict(status='receipt',sha256=digest(receipt)))
                send({'receipt':receipt})
                return
            else:
                raise ValueError('Invalid actor protocol or closed episode')
        raise ValueError('Worker request ceiling')


class RetailProcess:
    """Drop-in episode boundary for the actor collector; receipt stays private."""
    def __init__(self, python, plan, source, fees, journal, log, timeout=30):
        self.timeout=timeout
        self._receipt=None
        self._log=Path(log).open('x')
        environment={k:os.environ[k] for k in ('PATH','HOME','TMPDIR') if k in os.environ}
        command=[str(python),'-m','tool_lab.retail_process','--plan',str(plan),'--source',str(source),
                 '--fees',json.dumps(fees),'--journal',str(journal)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._log,
                                      text=True,env=environment)
        try:
            self._observation=self._read()['observation']
        except BaseException:
            self.close()
            raise

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError('Retail worker response deadline')
        line=self.process.stdout.readline()
        if not line:
            raise RuntimeError('Retail worker failed; preserve its journal and error log')
        return json.loads(line)

    def _request(self, value):
        self.process.stdin.write(canonical(value)+'\n')
        self.process.stdin.flush()
        return self._read()

    def observation(self):
        return copy.deepcopy(self._observation)

    def step(self, action):
        result=self._request({'action':action})
        if set(result)!={'observation','reward','done'}:
            raise ValueError('Malformed public delivery')
        self._observation=result['observation']
        return result

    def private_record(self):
        if self._receipt is None:
            self._receipt=self._request({'receipt':True})['receipt']
            if self.process.wait(timeout=5)!=0:
                raise RuntimeError('Retail worker failed after receipt')
        return copy.deepcopy(self._receipt)

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self._log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--fees',type=json.loads,required=True)
    parser.add_argument('--journal',type=Path,required=True)
    args=parser.parse_args()
    worker(args.plan,args.source,args.fees,args.journal)
