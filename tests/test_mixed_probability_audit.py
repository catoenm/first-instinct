import math
import unittest

from tool_lab.mixed_probability_audit import summarize


class ProbabilityAuditTests(unittest.TestCase):
    def sample(self):
        rows=[dict(id=str(i),group_id='world',public_input_sha256='same',input_ids=[10],
                   option_ids=['completed','incorrect','unfinished'],target_indices=[i]) for i in (0,1)]
        predictions=[dict(id=str(i),group_id='world',option_ids=r['option_ids'],target_index=i,
                          probabilities=[.5,.5,0.],brier=.5,log_loss=-math.log(.5),correct=i==0)
                     for i,r in enumerate(rows)]
        return rows,predictions

    def test_conflicting_labels_are_irreducible_uncertainty(self):
        rows,predictions=self.sample();result=summarize(predictions,rows,rows)
        self.assertEqual(result['subsets']['ambiguous']['questions'],2)
        self.assertEqual(result['enumerated_world_frequency_brier'],.5)
        self.assertEqual(result['mean_prediction_distance_from_world_frequencies'],0.)

    def test_different_encodings_cannot_be_called_identical_observations(self):
        rows,predictions=self.sample();rows[1]['input_ids']=[11]
        with self.assertRaises(ValueError):summarize(predictions,rows,rows)


if __name__=='__main__':unittest.main()
