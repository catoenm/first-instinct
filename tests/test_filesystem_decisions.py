import copy
import unittest

from tool_lab.filesystem_decisions import (World,fixtures,branch,audit_branch,verify,
    continuation,public_input,costs,PROGRAMS)


class FilesystemExecutionTests(unittest.TestCase):
    def setUp(self):self.world=World();self.cases=fixtures()
    def tearDown(self):self.world.close()
    def case(self,goal='one_path',partition=('a.txt','b.txt'),regime='cheap_write'):
        return next(c for c in self.cases if c['goal']==goal and c['partition']==list(partition) and c['regime']==regime)

    def test_identical_contents_hide_real_opposite_outcomes(self):
        a=self.case(partition=());b=self.case()
        traces=[branch(c,self.world,'write_a','stop_now') for c in (a,b)]
        self.assertEqual(traces[0]['forecast_input'],traces[1]['forecast_input'])
        self.assertEqual([t['outcome'] for t in traces],['completed','incorrect'])
        self.assertEqual(traces[1]['after']['b.txt']['contents'],b'new'.hex())
        for c,t in zip((a,b),traces):audit_branch(c,t)

    def test_same_commands_reverse_with_goal(self):
        for goal,correct,wrong in [('one_path','replace_a','write_a'),('physical_object','write_a','replace_a')]:
            c=self.case(goal=goal,regime='fresh')
            a=branch(c,self.world,correct,'stop_now');b=branch(c,self.world,wrong,'stop_now')
            self.assertEqual(a['outcome'],'completed');self.assertEqual(b['outcome'],'incorrect')
            self.assertEqual(continuation(a['input']),correct)

    def test_byte_correct_but_broken_alias_is_wrong(self):
        c=self.case(goal='physical_object');t=branch(c,self.world,'replace_ab','stop_now')
        self.assertEqual(t['after']['a.txt']['contents'],b'new'.hex())
        self.assertEqual(t['after']['b.txt']['contents'],b'new'.hex())
        self.assertEqual(t['outcome'],'incorrect')
        self.assertFalse(verify(c,t['before'],t['after']))

    def test_collateral_bytes_permissions_and_extra_paths_fail(self):
        c=self.case();t=branch(c,self.world,'replace_a','stop_now')
        self.assertTrue(verify(c,t['before'],t['after']))
        for kind in ('bytes','mode','extra'):
            after=copy.deepcopy(t['after'])
            if kind=='bytes':after['reference.txt']['contents']=b'corrupt'.hex()
            elif kind=='mode':after['a.txt']['mode']=0o644
            else:after['junk']=after['a.txt']
            self.assertFalse(verify(c,t['before'],after))

    def test_replay_and_invented_observation_or_target(self):
        c=self.case();t=branch(c,self.world,'inspect','evidence_then_complete')
        self.assertEqual(t,branch(c,self.world,'inspect','evidence_then_complete'))
        audit_branch(c,t)
        changed=copy.deepcopy(t);changed['outcome']='incorrect'
        with self.assertRaises(ValueError):audit_branch(c,changed)
        changed=copy.deepcopy(t);changed['events'][0]['observation']['stdout']='{}\n'
        with self.assertRaises(ValueError):audit_branch(c,changed)
        with self.assertRaises(ValueError):self.world.execute('user supplied shell command')

    def test_costs_change_reference_and_stopping(self):
        c=self.case(partition=(),regime='fresh')
        a=branch(c,self.world,'inspect','stop_now')
        self.assertEqual(continuation(a['input']),'write_a')
        changed=copy.deepcopy(c);changed.update(write_cost=.12,replace_cost=.02)
        b=branch(changed,self.world,'inspect','stop_now')
        self.assertEqual(continuation(b['input']),'replace_a')
        expensive=self.case(regime='expensive_mutation')
        self.assertEqual(continuation(public_input(expensive,[],0)),'finish')


if __name__=='__main__':unittest.main()
