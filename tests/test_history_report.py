import copy
import math
import unittest

from release_lab.history_report import audit_updates, audit_groups


class HistoryReceiptTests(unittest.TestCase):
    def test_wrong_group_hidden_by_correct_macro_is_rejected(self):
        from release_lab.pilot_metrics import summarize
        rows=[dict(id=str(i),option_ids=['a','b'],target_indices=[0],soft_target=None,
                   target_contract='acceptable_set',metric_groups=[str(i)],slices=['product']) for i in range(2)]
        probabilities=[[.9,.1],[.2,.8]]
        recorded=summarize(rows,probabilities)
        predictions=[dict(id=r['id'],probabilities=p) for r,p in zip(rows,probabilities)]
        audit_groups(rows,predictions,recorded)
        recorded['by_group']['0']['accuracy']=0.;recorded['by_group']['1']['accuracy']=1.
        with self.assertRaises(ValueError):audit_groups(rows,predictions,recorded)

    def fixture(self, after=(.52,.48)):
        index={'x':dict(id='x',learning_pool='general',option_ids=['a','b'])}
        config=dict(per_step={'general':1},probes_per_pool=1,max_mean_update_kl=.02,max_individual_update_kl=.10)
        value=sum(.5*math.log(.5/q) for q in after)
        diagnostic=dict(mean_full_kl=value,max_full_kl=value)
        accepted=value<=.02
        probe=dict(step=1,ids=['x'],before=[[.5,.5]],after=[list(after)],**diagnostic)
        events=[dict(step=1,phase='optimizer_attempt',physical_steps=1),
                dict(step=1,phase='accepted' if accepted else 'rejected',physical_steps=1,
                     accepted=accepted,checkpoint_eligible=accepted,parameters_and_optimizer_restored=not accepted,
                     reason=None if accepted else 'mean_policy_divergence',diagnostic=diagnostic)]
        return [index,[['x']],int(accepted),config,events,[probe]]

    def test_acceptance_and_rejection_have_distinct_counts(self):
        self.assertEqual(audit_updates(*self.fixture())['accepted_updates'],1)
        result=audit_updates(*self.fixture((.9,.1)))
        self.assertEqual(result['rejected_attempts'],1);self.assertEqual(result['accepted_updates'],0)

    def test_tampered_observations_fail(self):
        def vector(args): args[-1][0]['after']=[[.9,.1]]
        def receipt(args): args[-2][1]['parameters_and_optimizer_restored']=True
        def step(args): args[-2][0]['physical_steps']=2
        def selection(args): args[-1][0]['ids']=['evaluation-row']
        def omission(args): args[-2].pop(0)
        for corrupt in (vector,receipt,step,selection,omission):
            args=copy.deepcopy(self.fixture());corrupt(args)
            with self.subTest(corruption=corrupt.__name__),self.assertRaises(ValueError):
                audit_updates(*args)
