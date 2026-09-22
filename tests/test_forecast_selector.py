import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tool_lab import forecast_selector as f


def item(costs):
    return dict(state=json.dumps({'costs':costs}),question='Choose',options=[dict(id=a,description=a) for a in costs])


def p(c=0.,i=0.):
    return dict(completed=c,incorrect=i,unfinished=1-c-i)


class ForecastSelectorTests(unittest.TestCase):
    def test_selector_uses_public_cost_and_named_outcomes_only(self):
        public=item({'expensive':.8,'cheap':.01,'finish':0.})
        choice,_=f.choose(public,{'expensive':p(1.),'cheap':p(.7,.2),'finish':p()})
        self.assertEqual(choice,'cheap')
        self.assertEqual(f.choose(item({'z':0.,'a':0.}),{'z':p(),'a':p()})[0],'a')

    def test_missing_and_duplicate_menus_fail(self):
        with self.assertRaises(ValueError):f.choose(item({'a':0.,'b':0.}),{'a':p()})
        public=item({'a':0.});public['options']*=2
        with self.assertRaises(ValueError):f.choose(public,{'a':p()})

    def test_bad_probabilities_rejected(self):
        for values in ({'completed':1.},p(math.nan),p(1.1),dict(completed=.1,incorrect=.1,unfinished=.1)):
            with self.assertRaises(ValueError):f.choose(item({'a':0.}),{'a':values})

    def branches(self):
        public=item({'left':0.,'right':0.})
        result=[]
        for plan in f.PLANS:
            for world in ('hidden_left','hidden_right'):
                for action in ('left','right'):
                    win=world=='hidden_'+action
                    result.append(dict(id=f'{plan}-{world}-{action}',plan=plan,case_id=world,action=action,
                        input=public,reward=1. if win else -1.,cost=0.,events=[{'cost':0.}],
                        outcome='completed' if win else 'incorrect'))
        return result

    def test_average_before_max_does_not_give_hidden_world_knowledge(self):
        cells=f.cells_for(self.branches())
        for plan in f.PLANS:
            self.assertEqual(max(v['reward'] for v in cells[plan].values()),0.)
            self.assertEqual(cells[plan]['left']['distribution'],p(.5,.5))
        # An invalid omniscient max-before-average would instead earn one.

    def test_incomplete_world_action_matrix_and_input_mismatch_rejected(self):
        branches=self.branches()
        with self.assertRaises(ValueError):f.cells_for(branches[:-1])
        with self.assertRaises(ValueError):f.cells_for(branches+[branches[0]])
        branches[0]['input']=item({'left':.5,'right':0.})
        with self.assertRaises(ValueError):f.cells_for(branches)

    def test_perfect_outcomes_are_not_full_future_costs(self):
        public=item({'cheap_first':.01,'expensive_first':.2})
        action,_=f.choose(public,{'cheap_first':p(1.),'expensive_first':p(1.)})
        self.assertEqual(action,'cheap_first')
        full_returns={'cheap_first':-.01,'expensive_first':.8}
        self.assertGreater(full_returns['expensive_first'],full_returns[action])
        # Full realized costs never enter the selection function.

    def test_freeze_rejects_mutated_artifacts(self):
        with TemporaryDirectory() as d,patch.object(f,'ROOT',Path(d)):
            path=Path(d)/'input';path.write_text('original')
            plan=dict(paths={'input':f.file_hash(path)},version=f.VERSION,contracts=f.PLANS)
            f.verify_freeze(plan)
            path.write_text('edited')
            with self.assertRaisesRegex(ValueError,'changed'):f.verify_freeze(plan)

    def test_case_weighted_reporting_is_separate(self):
        a=dict(compatible_cases=1,action='left',**{k:1. for k in ('return','completion','incorrect','cost','regret','optimal','oracle_return')})
        b={**a,'compatible_cases':3,**{'return':0.}}
        self.assertEqual(f.summary([a,b])['return'],.5)
        self.assertEqual(f.summary([a,b],True)['return'],.25)


if __name__=='__main__':unittest.main()
