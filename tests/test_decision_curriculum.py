import copy
import json
import os
import unittest

from tool_lab import decision_curriculum as env
from tool_lab.decision_collect import make_questions, reconstruct, PLANS


class PublicContracts(unittest.TestCase):
    def test_serialization_and_private_metadata_boundary(self):
        for c in [c for c in env.cases() if c['regime']=='hidden']:
            def public(case):
                ep=env.Episode.__new__(env.Episode);ep.case=case;ep.depth=0;ep.history=[]
                return ep.input()
            saved=json.loads(json.dumps(c,sort_keys=True))
            self.assertEqual(public(c),public(saved))
            poisoned=copy.deepcopy(saved);poisoned['base']['expected']={'private':'secret'}
            poisoned['base']['visible']={'private':'secret'}
            self.assertEqual(public(c),public(poisoned))


@unittest.skipUnless(os.environ.get('FIRST_INSTINCT_DOCKER_TESTS')=='1','Explicit local Docker integration tests')
class DecisionCurriculum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=env.cases(1);cls.engine=env.CatalogExecutor()

    @classmethod
    def tearDownClass(cls):cls.engine.close()

    def case(self,family='config',regime='hidden'):
        return next(c for c in self.cases if c['family']==family and c['regime']==regime)

    def test_worlds_have_identical_public_histories_and_opposite_outcomes(self):
        c=self.case();goal=c['base']['goal']
        pair=[x for x in self.cases if x['family']=='config' and x['regime']=='hidden' and x['base']['goal']==goal]
        traces=[env.execute_branch(x,self.engine,'repair_0','stop_now') for x in pair]
        self.assertEqual(len(traces),2)
        self.assertEqual(traces[0]['input'],traces[1]['input'])
        self.assertEqual({t['outcome'] for t in traces},{'completed','incorrect'})

    def test_current_evidence_resolves_goal_and_menu_permutation(self):
        for c in [x for x in self.cases if x['regime']=='fresh']:
            ep=env.Episode(c,self.engine);item=ep.input();permuted=copy.deepcopy(item)
            permuted['options'].reverse()
            self.assertEqual(env.continuation(item),env.continuation(permuted))
            while not ep.done:ep.step(env.continuation(ep.input()))
            self.assertEqual(ep.outcome,'completed')

    def test_recovery_after_actual_command_failures(self):
        for family in env.OWNERS:
            for regime in ('offline','prerequisite','failed_write'):
                c=self.case(family,regime)
                t=env.execute_branch(c,self.engine,'summary','evidence_then_commit')
                self.assertEqual(t['outcome'],'completed',(family,regime))
                self.assertTrue(any(e['observation']['returncode'] for e in t['prefix']))
                reconstruct(c,t)

    def test_publication_requires_validation_of_current_bytes(self):
        c=self.case('publish');ep=env.Episode(c,self.engine)
        ep.step('repair_0');ep.step('prepare');ep.step('repair_1');ep.step('commit')
        self.assertEqual(ep.events[-1]['observation']['returncode'],75)
        self.assertFalse(ep.done)
        ep.step('prepare');ep.step('commit')
        self.assertEqual(ep.events[-1]['observation']['returncode'],0)
        self.assertTrue(ep.done)

    def test_parent_verifier_rejects_collateral_and_wrong_output(self):
        for family in env.OWNERS:
            c=self.case(family);t=env.execute_branch(c,self.engine,'summary','evidence_then_commit')
            self.assertTrue(env.verify_state(c,t['before'],t['after']))
            changed=copy.deepcopy(t['after']);changed['reference.txt']['sha256']='0'*64
            self.assertFalse(env.verify_state(c,t['before'],changed))
            changed=copy.deepcopy(t['after']);changed['unexpected.txt']={'text':'x','sha256':'0'*64}
            self.assertFalse(env.verify_state(c,t['before'],changed))
            changed=copy.deepcopy(t['after']);changed['ops.py']['sha256']='0'*64
            self.assertFalse(env.verify_state(c,t['before'],changed))

    def test_json_roundtrip_preserves_public_inputs_and_base_worlds(self):
        for family in env.OWNERS:
            c=self.case(family,'failed_write')
            original=env.Episode(c,self.engine).input()
            saved=json.loads(json.dumps(c,sort_keys=True))
            self.assertEqual(original,env.Episode(saved,self.engine).input())
            if family=='publish':self.assertEqual(set(c['base']['files']),{'source0.json','source1.json','live.json','reference.txt'})

    def test_historical_contradictions_do_not_reveal_current_truth(self):
        c=self.case(regime='contradictory');goal=c['base']['goal']
        pair=[x for x in self.cases if x['family']=='config' and x['regime']=='contradictory' and x['base']['goal']==goal]
        inputs=[env.Episode(x,self.engine).input() for x in pair]
        self.assertEqual(inputs[0],inputs[1])
        observations=json.loads(inputs[0]['state'])['observations']
        self.assertNotEqual(observations[0]['stdout'],observations[1]['stdout'])

    def test_immediate_and_continued_forecasts_mean_different_things(self):
        c=self.case('publish')
        now=env.execute_branch(c,self.engine,'summary','stop_now')
        later=env.execute_branch(c,self.engine,'summary','evidence_then_commit')
        self.assertEqual(now['outcome'],'unfinished');self.assertEqual(later['outcome'],'completed')
        self.assertEqual(now['input'],later['input']);self.assertNotEqual(now['forecast_input'],later['forecast_input'])
        tampered=copy.deepcopy(later);tampered['outcome']='unfinished'
        with self.assertRaises(ValueError):reconstruct(c,tampered)

    def test_branch_means_not_private_winners_supply_decision_labels(self):
        selected=[c for c in self.cases if c['family']=='config' and c['regime'] in ('hidden','expensive','redundant')]
        traces=[]
        for c in selected:
            options=env.Episode(c,self.engine).input()['options']
            for o in options:
                for plan in PLANS:traces.append(env.execute_branch(c,self.engine,o['id'],plan))
        rows,contexts=make_questions(selected,traces)
        self.assertTrue(rows)
        for context in contexts:
            if context['regime']=='hidden':self.assertEqual(context['best_actions'],['summary'])
            if context['regime']=='expensive':self.assertEqual(context['best_actions'],['finish'])
            if context['regime']=='redundant':self.assertLess(context['observation_value'],0)
        self.assertTrue(all('outcome' not in json.loads(r['input']['state']) for r in rows))


if __name__=='__main__':unittest.main()
