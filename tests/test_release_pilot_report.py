import unittest
from release_lab.pilot_report import recompute


class ReportTests(unittest.TestCase):
    def test_outcome_atoms_and_multiple_server_weighting(self):
        rows=[dict(id='a',option_ids=['y','n'],target_contract='categorical_distribution',soft_target=[.25,.75],metric_groups=['a']),
              dict(id='b',option_ids=['y','n'],target_contract='categorical_distribution',soft_target=[1.,0.],metric_groups=['b'])]
        values=recompute(rows,[dict(id='a',probabilities=[.25,.75]),dict(id='b',probabilities=[.5,.5])])
        self.assertAlmostEqual(values['brier'],(.375+.5)/2)

    def test_missing_duplicate_unknown_predictions_fail(self):
        rows=[dict(id='a',option_ids=['y','n'],target_contract='acceptable_set',target_indices=[0],metric_groups=['a'])]
        pred=dict(id='a',probabilities=[.5,.5])
        for predictions in ([],[pred,pred],[dict(pred,id='b')]):
            with self.assertRaises(ValueError):recompute(rows,predictions)


if __name__=='__main__':unittest.main()
