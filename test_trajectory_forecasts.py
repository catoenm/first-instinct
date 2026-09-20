import copy
import os
import unittest

from tool_lab import decision_curriculum as env
from tool_lab.trajectory_forecasts import branch, fixtures, reconstruct


class OwnershipTests(unittest.TestCase):
    def test_old_publication_is_never_a_new_test(self):
        cases=fixtures()
        self.assertEqual(len(cases),16)
        self.assertEqual({c['split'] for c in cases},{'exposed_diagnostic'})
        self.assertEqual(len({c['group_id'] for c in cases}),1)
        self.assertEqual(len({c['base']['id'] for c in cases}),4)


@unittest.skipUnless(os.environ.get('FIRST_INSTINCT_DOCKER_TESTS')=='1','Explicit Docker integration')
class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=env.CatalogExecutor();cls.cases=fixtures()

    @classmethod
    def tearDownClass(cls):cls.engine.close()

    def case(self,regime='hidden'):
        return next(c for c in self.cases if c['regime']==regime)

    def test_precommit_has_real_conditional_uncertainty(self):
        case=self.case()
        pair=[c for c in self.cases if c['regime']=='hidden' and c['base']['goal']==case['base']['goal']]
        traces=[branch(c,self.engine,('repair_0','prepare'),'commit','stop_now') for c in pair]
        self.assertEqual(traces[0]['forecast_input'],traces[1]['forecast_input'])
        self.assertEqual({t['outcome'] for t in traces},{'completed','incorrect'})
        for c,t in zip(pair,traces):reconstruct(c,t)

    def test_future_cost_excludes_sunk_history(self):
        c=self.case();t=branch(c,self.engine,('repair_0','prepare'),'finish','stop_now')
        self.assertEqual(t['cost'],0.)
        self.assertAlmostEqual(t['past_cost'],.05)
        self.assertEqual(t['reward'],0.)
        reconstruct(c,t)
        tampered=copy.deepcopy(t);tampered['cost']=t['past_cost']
        with self.assertRaises(ValueError):reconstruct(c,tampered)

    def test_stale_validation_fails_without_irreversible_damage(self):
        c=self.case();t=branch(c,self.engine,('repair_0','prepare','repair_1'),'commit','stop_now')
        self.assertEqual(t['outcome'],'unfinished')
        self.assertEqual(t['trajectory']['events'][3]['observation']['returncode'],75)
        reconstruct(c,t)

    def test_tampered_history_and_model_supplied_outcome_rejected(self):
        c=self.case();t=branch(c,self.engine,('commit',),'summary','evidence_then_commit')
        reconstruct(c,t)
        self.assertEqual(t['outcome'],'completed')
        wrong=copy.deepcopy(t);wrong['outcome']='incorrect'
        with self.assertRaises(ValueError):reconstruct(c,wrong)
        wrong=copy.deepcopy(t);wrong['history']=['summary']
        with self.assertRaises(ValueError):reconstruct(c,wrong)


if __name__=='__main__':unittest.main()
