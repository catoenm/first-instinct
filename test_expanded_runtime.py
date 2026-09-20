from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tool_lab import filesystem_decisions as fs, calendar_decisions as cal
from tool_lab.expanded_runtime import Budget, execute, make_episode, reservation_cases, audit_trace


class ExpandedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.budget=Budget(Path(self.temp.name)/'attempts.jsonl',max_episodes=6,max_actions=64,max_seconds=60)

    def fs(self, goal='one_path', regime='fresh', partition=None):
        return next(c for c in fs.fixtures() if c['goal']==goal and c['regime']==regime and
                    c['partition']==(['a.txt','b.txt'] if partition is None else partition))

    def calendar(self, world=2, regime='hidden'):
        return next(c for c in cal.fixtures() if c['structure']=='reschedule' and c['world']==world and c['regime']==regime)

    def reservation(self, world=2, prefix=None, horizon=6):
        return next(c for c in reservation_cases() if c['world']==world and c['prefix']==(['sequential'] if prefix is None else prefix)
                    and c['profile']['horizon']==horizon and c['profile']['cohort']=='familiar'
                    and c['profile']['price']=='cheap_queries')

    def test_filesystem_goal_changes_correct_mutation(self):
        for goal,action,outcome in [('one_path','write_a','incorrect'),('one_path','replace_a','completed'),
                                    ('physical_object','write_a','completed'),('physical_object','replace_a','incorrect')]:
            t=execute(self.fs(goal),self.budget,actions=[action])
            self.assertEqual(t['outcome'],outcome)
            self.assertEqual(t['cost'],fs.costs(self.fs(goal))[action])

    def test_calendar_failed_atomic_recovers_but_sequential_loses_original(self):
        case=self.calendar()
        t=execute(case,self.budget,actions=['atomic_first','inspect','finish'])
        self.assertEqual(t['outcome'],'unfinished');self.assertEqual(t['before'],t['after'])
        self.assertAlmostEqual(t['reward'],-.1)
        self.assertFalse(t['events'][0]['terminal'])
        other=execute(case,self.budget,actions=['sequential_first'])
        self.assertEqual(other['outcome'],'incorrect');self.assertAlmostEqual(other['reward'],-1.015)
        self.assertNotIn(200,[r[0] for r in other['after']['events']])

    def test_reservation_partial_write_repair_and_sunk_cost(self):
        case=self.reservation()
        t=execute(case,self.budget,actions=['undo','replenish_b','atomic','finish'])
        self.assertEqual(t['outcome'],'completed')
        self.assertAlmostEqual(t['sunk_prefix_cost'],.01)
        self.assertAlmostEqual(t['cost'],.09);self.assertAlmostEqual(t['reward'],.91)
        self.assertFalse(any(e['terminal'] for e in t['events'][:-1]))
        state=json.loads(t['input']['state'])
        self.assertNotIn('costs_quarters',state['observation'])
        self.assertEqual(state['costs']['atomic'],.04)

    def test_already_completed_prefix_gets_terminal_utility_once(self):
        t=execute(self.reservation(world=0),self.budget,actions=['finish'])
        self.assertEqual(t['reward'],1.);self.assertEqual(t['events'][0]['reward'],1.)
        self.assertGreater(t['sunk_prefix_cost'],0)
        self.assertIn('no fixed continuation',t['input']['question'])

    def test_redundant_queries_exhaust_horizon(self):
        f=execute(self.fs(regime='cheap_write'),self.budget,actions=['inspect']*4)
        self.assertAlmostEqual(f['reward'],-.06);self.assertEqual(f['outcome'],'unfinished')
        r=execute(self.reservation(prefix=[],horizon=3),self.budget,actions=['inspect_stock']*3)
        self.assertAlmostEqual(r['reward'],-.0075);self.assertEqual(r['outcome'],'unfinished')

    def test_private_worlds_do_not_change_unobserved_inputs(self):
        pairs=[(self.fs(regime='cheap_write'),self.fs(regime='cheap_write',partition=[])),
               (self.calendar(world=0),self.calendar(world=2)),
               (self.reservation(world=0,prefix=[]),self.reservation(world=2,prefix=[]))]
        for first,second in pairs:
            a=make_episode(first,self.budget);b=make_episode(second,self.budget)
            try:
                self.assertEqual(a.input(),b.input())
                for key in ('case_id','expected_span','initial_a','outcome'):
                    self.assertNotIn('"'+key+'"',a.input()['state'])
            finally:a.close();b.close()

    def test_tampered_observations_targets_and_transitions_rejected(self):
        cases=[self.fs(),self.calendar(regime='failed'),self.reservation()]
        for case in cases:
            t=execute(case,self.budget)
            corrupt=deepcopy(t);corrupt['events'][0]['reward']+=1
            with self.assertRaises(ValueError):audit_trace(case,corrupt)
            corrupt=deepcopy(t);corrupt['events'][0]['input']['state']+=' fabricated'
            with self.assertRaises(ValueError):audit_trace(case,corrupt)
            corrupt=deepcopy(t);corrupt['outcome']='made_up'
            with self.assertRaises((ValueError,KeyError)):audit_trace(case,corrupt)
            if case['family']=='reservation':
                corrupt=deepcopy(t);corrupt['events'][0]['observation']['account']=0
                with self.assertRaises(ValueError):audit_trace(case,corrupt)

    def test_illegal_actions_terminal_guards_and_attempt_cap(self):
        budget=Budget(Path(self.temp.name)/'bounded.jsonl',max_episodes=1,max_actions=4,max_seconds=60)
        ep=make_episode(self.fs(),budget)
        try:
            before=budget.actions
            with self.assertRaises(ValueError):ep.step('invented-command')
            self.assertEqual(budget.actions,before)
            ep.step('finish')
            with self.assertRaises(ValueError):ep.step('finish')
            with self.assertRaises(RuntimeError):make_episode(self.fs(),budget)
        finally:ep.close()


if __name__=='__main__':unittest.main()
