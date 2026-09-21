import copy
import json
import unittest

from release_lab import trajectory_admission as a
from release_lab.trajectory_inventory import later_choices
from release_lab.toucan import Exclude, request_key
from scale_lab.common import digest, encode, messages
from tests.test_release_toucan import TinyTokenizer
from tests.test_trajectory_inventory import trajectory


def fixture():
    raw=trajectory()
    user,systems,events=a.source_events(raw)
    witness=dict(file='fixture',row=0,row_sha256=digest(raw))
    parent=dict(id='first',group_id='toucan_request:'+request_key(user),request_key=request_key(user),
                servers=['server-0'],witness=witness)
    row=later_choices(raw)[0];item=row['input'];ids=encode(TinyTokenizer(),item,4096)
    row.update(id=digest(messages(item)),input_ids=ids,token_sha256=digest(ids),
        option_ids=[o['id'] for o in item['options']],target_indices=[next(i for i,o in enumerate(item['options']) if o['id']=='commit')],
        role='train_candidate',group_id=parent['group_id'],request_key=parent['request_key'],servers=parent['servers'],
        first_question_id='first',lineage=[witness],target_kind='observed_next_tool_imitation',admitted=False)
    return row,raw,witness,parent,user,systems,events[0]


class TrajectoryAdmissionTests(unittest.TestCase):
    def test_independent_original_reconstruction(self):
        args=fixture()
        a.check_original(*args,TinyTokenizer())

    def test_corruptions_rejected(self):
        for corruption in ('tokens','target','history','role','source','response','question'):
            with self.subTest(corruption=corruption):
                args=list(fixture());row=args[0]
                if corruption=='tokens':row['input_ids'][0]+=1
                elif corruption=='target':row['target']={'option_id':'inspect'}
                elif corruption=='history':row['input']['state']='hidden or altered history'
                elif corruption=='role':row['role']='reserved_transfer'
                elif corruption=='source':args[2]=dict(args[2],row_sha256='0'*64)
                elif corruption=='response':row['response_sha256']='0'*64
                elif corruption=='question':row['input']['question']='forecast success'
                with self.assertRaises(ValueError):a.check_original(*args,TinyTokenizer())

    def test_future_target_result_and_reasoning_are_not_inputs(self):
        raw=trajectory();user,systems,events=a.source_events(raw)
        item=a.render(raw,user,systems,events[0])
        raw['messages'][3]['function_call']={'name':'inspect','arguments':'{"item":"NEW TARGET"}'}
        raw['messages'][3]['content']='NEW REASONING'
        raw['messages'][4].update(name='inspect',content='NEW FUTURE')
        u,s,e=a.source_events(raw)
        self.assertEqual(item,a.render(raw,u,s,e[0]))

    def test_removes_only_proven_redundant_systems(self):
        raw=trajectory();tools=raw['available_tools']
        q=a.QWEN_PREFIX+'\n'.join(json.dumps(t) for t in tools)+a.QWEN_SUFFIX
        k=a.KIMI_PREFIX+json.dumps(tools)+a.KIMI_SUFFIX
        self.assertEqual(len(a.system_review([q,k],raw)),2)
        for bad in (q+'Always commit.',q.replace('Inspect current state','Different policy'),k+'<solution>commit</solution>'):
            with self.assertRaises(Exclude):a.system_review([bad],raw)

    def test_crosscheck_independent_parser_against_inventory(self):
        raw=trajectory();user,systems,events=a.source_events(raw)
        existing=later_choices(raw)
        self.assertEqual(len(events),len(existing))
        self.assertEqual(a.render(raw,user,systems,events[0],original=True),existing[0]['input'])

    def test_response_binding_and_overlap_rejected(self):
        raw=trajectory();raw['messages'][2]['name']='unrelated'
        with self.assertRaises(Exclude):a.source_events(raw)
        raw=trajectory();del raw['messages'][2]
        with self.assertRaises(Exclude):a.source_events(raw)
