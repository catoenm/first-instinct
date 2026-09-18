"""Harbor external agent: public proposals -> local selector -> real shell.

The proposer mailbox permits a model in this Codex account to participate without
an external paid model API. Only public observations are written to requests.
Commands execute through Harbor, never in the host shell. This collects evidence;
it does not update any weights or claim to be an on-policy training implementation.
"""

import asyncio
import json
import math
from pathlib import Path
import random
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from harbor.agents.base import BaseAgent
from .protocol import canonical, choose, digest, proposal_request, selector_payload, validate_proposal


def atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class LocalSelector:
    def __init__(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.path not in ('','/') or parsed.username or parsed.query or parsed.fragment:
            raise ValueError('Selector must be the local resident model service')
        self.url = url.rstrip('/')
        self.metadata = None

    def predict(self,payload):
        with urlopen(self.url+'/api/status',timeout=5) as response:
            status=json.load(response)
        metadata={k:status[k] for k in ('model','model_revision','checkpoint','device')}
        if not status.get('ready') or (self.metadata is not None and metadata!=self.metadata):
            raise ValueError('Local model unavailable or changed within the episode')
        self.metadata=metadata
        request=Request(self.url+'/api/answer',data=canonical(payload).encode(),
                        headers={'Content-Type':'application/json','X-CSRF-Token':status['csrf_token']},method='POST')
        with urlopen(request,timeout=60) as response:
            result=json.load(response)
        if any(result[k]!=metadata[k] for k in metadata):
            raise ValueError('Model response provenance changed')
        return result['answers']['action'],metadata


class ProposalChoiceAgent(BaseAgent):
    def __init__(self,*args,mailbox=None,selector_url='http://127.0.0.1:8766',
                 max_steps=6,proposal_timeout=180,seed=101,mode='argmax',**kwargs):
        super().__init__(*args,**kwargs)
        if mailbox is None:raise ValueError('An isolated host-side proposer mailbox is required')
        self.mailbox=Path(mailbox).resolve()
        self.max_steps=int(max_steps);self.proposal_timeout=float(proposal_timeout)
        self.seed=int(seed);self.mode=mode
        if not 1<=self.max_steps<=8 or not 1<=self.proposal_timeout<=240 or mode not in ('sample','argmax'):
            raise ValueError('Invalid bounded episode settings')
        self.selector=LocalSelector(selector_url)

    @staticmethod
    def name():return 'first-instinct-proposal-choice'

    def version(self):return '1.0.1'

    async def setup(self,environment):
        pass

    async def proposal(self,request):
        folder=self.mailbox/str(self.session_id)
        folder.mkdir(parents=True,exist_ok=True)
        request_path=folder/f"step-{request['step']:02d}.request.json"
        response_path=folder/f"step-{request['step']:02d}.response.json"
        if request_path.exists() or response_path.exists():raise ValueError('Mailbox request already exists')
        atomic_json(request_path,{'request_sha256':digest(request),'request':request})
        deadline=time.monotonic()+self.proposal_timeout
        while not response_path.exists():
            if time.monotonic()>=deadline:raise TimeoutError('No proposer response; no fabricated candidates')
            await asyncio.sleep(.25)
        if response_path.stat().st_size>12000:raise ValueError('Oversized proposal')
        response=json.loads(response_path.read_text())
        candidates=validate_proposal(response,request)
        return response,candidates

    async def run(self,instruction,environment,context):
        self.logs_dir.mkdir(parents=True,exist_ok=True)
        history=[];commands=0;rng=random.Random(self.seed)
        receipt={'schema':'harbor-command-trajectory-v1','status':'running','mode':self.mode,
                 'seed':self.seed,'max_steps':self.max_steps,'command_cost':.01,
                 'reward_contract':'Independent terminal verifier success (0 or 1) minus 0.01 per selected executed command.',
                 'proposer_contract':'Only instruction, public command history and remaining steps. No task path, solution, verifier, hidden state or rewards.',
                 'checkpoint_updated':False,'events':[]}
        path=self.logs_dir/'choices.json'
        atomic_json(path,receipt)
        try:
            for step in range(self.max_steps):
                request=proposal_request(instruction,history,step,self.max_steps)
                response,candidates=await self.proposal(request)
                payload,mapping=selector_payload(request,candidates,self.seed+step)
                predicted,metadata=await asyncio.to_thread(self.selector.predict,payload)
                probabilities=predicted['probabilities']
                selected=choose(probabilities,list(mapping),self.mode,rng)
                event={'step':step,'request':request,'proposal':response,'selector_input':payload,
                       'selector_input_sha256':digest(payload),'mapping':mapping,'probabilities':probabilities,
                       'selected':selected,'selected_log_probability':math.log(probabilities[selected]),
                       'selector_metadata':metadata,'selector_milliseconds':predicted['milliseconds']}
                # Persist the actual sampled/greedy decision before executing it.
                receipt['events'].append(event);atomic_json(path,receipt)
                command=mapping[selected]['command']
                if command is None:
                    receipt['status']='finished_by_selector';break
                commands+=1
                result=await environment.exec(command=command,cwd='/app',user='runner',timeout_sec=15)
                stdout=result.stdout or '';stderr=result.stderr or ''
                observation={'command':command,'exit_code':result.return_code,
                             'stdout':stdout[:1200],'stderr':stderr[:600],
                             'stdout_truncated':len(stdout)>1200,'stderr_truncated':len(stderr)>600}
                event['observation']=observation
                event['immediate_reward']=-.01
                history.append(observation);atomic_json(path,receipt)
            else:receipt['status']='step_budget_exhausted'
        except BaseException as exc:
            receipt.update(status='infrastructure_error',error_type=type(exc).__name__)
            # The Harbor trial is an error, not an ordinary failed task reward.
            raise
        finally:
            receipt['executed_commands']=commands
            receipt['terminal_reward']='pending independent Harbor verifier'
            atomic_json(path,receipt)
            context.metadata={'trajectory_schema':receipt['schema'],'status':receipt['status'],
                              'executed_commands':commands,'selector':self.selector.metadata,
                              'note':'Inference-only prototype. No training or generalization claim.'}
