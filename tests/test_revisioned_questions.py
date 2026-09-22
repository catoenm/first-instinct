from copy import deepcopy
import unittest
from tool_lab.revisioned_sqlite import GOALS,CONTRACT,DESCRIPTIONS,costs
from tool_lab.revisioned_questions import render,unrender,outcome_at,reverse_options
from tool_lab.revisioned_questions_audit import classify


def public():
    return dict(goal=GOALS['increment_latest'],contract=CONTRACT,cache=[10,'original',0],remaining=4,
                history=[dict(action='read',returncode=0,row=[10,'original',0],historical=True)],costs=costs('cheap'),
                options=[dict(id=k,description=v) for k,v in DESCRIPTIONS.items()])


class RevisionedQuestionsTests(unittest.TestCase):
    def test_lossless_public_state_and_private_or_fabricated_evidence_rejected(self):
        value=public();self.assertEqual(unrender(render(value)),value)
        for field in ('intervened','outcome','hidden_world','target'):
            bad=deepcopy(value);bad[field]='private'
            with self.assertRaises(ValueError):render(bad)
        bad=deepcopy(value);bad['cache'][0]=15
        with self.assertRaises(ValueError):render(bad)

    def test_immediate_state_verification_needs_no_fabricated_stop(self):
        start=dict(schema=['unchanged'],integrity=['ok'],rows=[[1,15,'colleague',1],[99,80,'protected',4]])
        after=deepcopy(start);after['rows'][0]=[1,16,'colleague',2]
        for verifier in (classify,outcome_at):
            self.assertEqual(verifier('increment_latest',start,after),'completed')
            self.assertEqual(verifier('approved_revision',start,after),'incorrect')
            self.assertEqual(verifier('approved_revision',start,start),'completed')
            bad=deepcopy(after);bad['rows'][1][1]=0
            self.assertEqual(verifier('increment_latest',start,bad),'incorrect')

    def test_menu_permutation_preserves_outcome_mass_and_acceptable_action_sets(self):
        row=dict(id='one',input=dict(state='state',question='question',options=[dict(id=n,description=n) for n in ('a','b','c')]),
                 option_ids=['a','b','c'],soft_target=[.5,.5,0.])
        reversed_row=reverse_options(row)
        self.assertEqual(dict(zip(row['option_ids'],row['soft_target'])),dict(zip(reversed_row['option_ids'],reversed_row['soft_target'])))
        del row['soft_target'];row.update(option_values=[3.,1.,3.],target_indices=[0,2],target={'option_ids':['a','c']})
        again=reverse_options(row)
        self.assertEqual({again['option_ids'][i] for i in again['target_indices']},{'a','c'})
        self.assertEqual(again['target'],row['target'])


if __name__=='__main__':unittest.main()
