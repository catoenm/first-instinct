from collections import Counter
import copy
import json
import os
from pathlib import Path
import unittest

from tool_lab import application_curriculum as app


class ApplicationContractTest(unittest.TestCase):
    def test_fixture_ownership_and_menu_independence(self):
        cases=app.fixtures()
        self.assertEqual(len(cases),84)
        self.assertEqual(len({c['group_id'] for c in cases}),1)
        self.assertEqual({c['split'] for c in cases},{'train'})
        self.assertEqual(len({app.digest(app.menu(c)) for c in cases}),1)
        self.assertEqual(len(cases)*len(app.menu(cases[0]))*len(app.PLANS)*2,app.MAX_BRANCHES)


@unittest.skipUnless(os.environ.get('FIRST_INSTINCT_TOOL_SANDBOX_TESTS')=='1','Pinned ToolSandbox integration is opt-in')
class ApplicationExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=app.replay_guard();cls.proof=cls.guard.__enter__()
        cls.backend=app.backend_at(Path('.local/toolsandbox-upstream'))
        cls.post_import_attempts=len(cls.proof['blocked_attempts'])
        cls.budget=Counter(branches=0,setup_calls=0,prefix_calls=0,branch_calls=0)
        cls.traces=[]

    @classmethod
    def tearDownClass(cls):
        cls.guard.__exit__(None,None,None)
        if len(cls.proof['blocked_attempts'])!=cls.post_import_attempts:
            raise AssertionError('Tool execution attempted a blocked capability')
        if os.environ.get('FIRST_INSTINCT_APPLICATION_TEST_RECEIPT'):
            Path(os.environ['FIRST_INSTINCT_APPLICATION_TEST_RECEIPT']).write_text(json.dumps(
                dict(budget=cls.budget,guard=cls.proof,executions=cls.traces),sort_keys=True,indent=2)+'\n')

    def branch(self, action, plan='stop_now', **criteria):
        match={**dict(ledger='none',connection='ready',goal='a',regime='hidden'),**criteria}
        case=next(c for c in app.fixtures() if all(c[k]==v for k,v in match.items()))
        trace=app.execute_branch(self.backend,case,action,plan,self.budget)
        self.traces.append(trace);app.reconstruct(case,trace)
        return case,trace

    def test_prerequisite_chain_recovers_failed_send(self):
        _,trace=self.branch('send_a','evidence_then_complete',connection='low_battery')
        self.assertEqual(trace['outcome'],'completed')
        self.assertEqual(trace['events'][0]['observation']['error']['type'],'ConnectionError')
        actions=[e['action'] for e in trace['events']]
        self.assertLess(actions.index('battery_off'),actions.index('cellular_on'))
        self.assertEqual(actions[-1],'send_a')

    def test_stop_and_duplicate_have_different_real_outcomes(self):
        _,stopped=self.branch('finish',ledger='a',regime='fresh')
        _,duplicate=self.branch('send_a',ledger='a',regime='fresh')
        self.assertEqual(stopped['outcome'],'completed')
        self.assertEqual(stopped['cost'],0)
        self.assertEqual(duplicate['outcome'],'incorrect')
        self.assertEqual(len(duplicate['after']['messaging']),2)

    def test_hidden_outcomes_differ_without_input_leakage(self):
        _,first=self.branch('send_a')
        _,second=self.branch('send_a',ledger='a')
        self.assertEqual(first['forecast_input'],second['forecast_input'])
        self.assertEqual({first['outcome'],second['outcome']},{'completed','incorrect'})

    def test_historical_setup_does_not_change_current_world(self):
        _,hidden=self.branch('finish',ledger='b')
        _,stale=self.branch('finish',ledger='b',regime='stale')
        self.assertEqual(hidden['before'],stale['before'])
        self.assertEqual(stale['prefix'][0]['observation']['result'],[])
        self.assertTrue(stale['prefix'][0]['historical'])

    def test_independent_input_audit_rejects_changed_goal(self):
        from tool_lab.application_audit import audit_initial
        case,trace=self.branch('finish')
        audit_initial(case,trace)
        changed=copy.deepcopy(trace);state=json.loads(changed['input']['state'])
        state['goal']['recipient_phone']=case['phones']['b']
        changed['input']['state']=app.canonical(state)
        with self.assertRaisesRegex(ValueError,'Public input disagrees'):
            audit_initial(case,changed)

    def test_protected_setting_damage_is_terminal(self):
        _,trace=self.branch('battery_on','evidence_then_complete')
        self.assertEqual(trace['outcome'],'incorrect')
        self.assertEqual(len(trace['events']),1)
        self.assertFalse(trace['after']['setting'][0]['wifi'])

    def test_independent_verifier_negative_controls(self):
        case,trace=self.branch('send_a')
        for mutation in ('wrong_recipient','contact','wifi','duplicate'):
            with self.subTest(mutation=mutation):
                after=copy.deepcopy(trace['after'])
                if mutation=='wrong_recipient':after['messaging'][0]['recipient_phone_number']=case['phones']['b']
                elif mutation=='contact':after['contact'][0]['phone_number']='+12025550199'
                elif mutation=='wifi':after['setting'][0]['wifi']=False
                else:after['messaging'].append(dict(after['messaging'][0],message_id='duplicate'))
                self.assertTrue(app.verify(case,trace['before'],after)['incorrect'])


if __name__=='__main__':
    unittest.main()
