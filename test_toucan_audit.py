import copy
import unittest

from scale_lab.toucan_audit import inspect


def row():
    return {'uuid':'a','question':'Find an item','subset_name':'fixture','messages':[
        {'role':'user','content':'Find an item'},
        {'role':'assistant','function_call':{'name':'find','arguments':'{"query":"item"}'}},
        {'role':'function','name':'find','content':'{"isError":true}'}],
        'available_tools':[{'function':{'name':'find','parameters':{'required':['query']}}},
                           {'function':{'name':'read','parameters':{}}}],
        'metadata':{'prompt_id':'one','mcp_servers':[{'server_id':1}]},
        'response_quality_assessment':{'completeness':{'score':1},'overall_score':2},'question_quality_assessment':{}}


class ToucanAuditTests(unittest.TestCase):
    def test_response_is_not_success(self):
        x=inspect(row())
        self.assertTrue(x['structurally_usable_first_call'])
        self.assertEqual(x['error_shaped_responses'],1)
        self.assertEqual(x['judge_completeness'],1)

    def test_results_must_match_calls(self):
        x=row();x['messages'][-1]['name']='wrong'
        result=inspect(x)
        self.assertEqual(result['unmatched_responses'],1)
        self.assertEqual(result['calls_without_response'],1)
        self.assertFalse(result['structurally_usable_first_call'])

    def test_required_arguments_and_declared_tools(self):
        x=row();x['messages'][1]['function_call']['arguments']='{}'
        self.assertEqual(inspect(x)['missing_required_fields'],1)
        self.assertFalse(inspect(x)['structurally_usable_first_call'])
        x['messages'][1]['function_call']['name']='unknown'
        self.assertEqual(inspect(x)['calls_without_exact_declared_name'],1)

    def test_parallel_calls_link_by_id(self):
        x=row();x['messages']=[{'role':'assistant','tool_calls':[
            {'id':'1','function':{'name':'find','arguments':'{"query":"x"}'}},
            {'id':'2','function':{'name':'read','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'2','content':'{}'},
            {'role':'tool','tool_call_id':'1','content':'{}'}]
        result=inspect(x)
        self.assertEqual(result['parallel_call_turns'],1)
        self.assertEqual(result['calls_without_response'],0)
        self.assertEqual(result['unmatched_responses'],0)


if __name__=='__main__':unittest.main()
