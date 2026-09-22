import unittest
from puffer_lab.consequence_overlap import partition


class ConsequenceOverlapTests(unittest.TestCase):
    def test_different_histories_with_same_rendered_tokens_are_detected(self):
        train=[dict(id='train',group_id='g1',input_ids=[1,2,3],soft_target=[.25,.75])]
        validation=[dict(id='duplicate',group_id='g2',input_ids=[1,2,3],soft_target=[.25,.75]),
                    dict(id='novel',group_id='g3',input_ids=[1,4,3],soft_target=[1.,0.])]
        overlap,unique=partition(train,validation)
        self.assertEqual([r['id'] for r in overlap],['duplicate'])
        self.assertEqual([r['id'] for r in unique],['novel'])

    def test_identical_prompt_conflicting_target_is_rejected(self):
        row=dict(input_ids=[1,2],soft_target=[.25,.75])
        conflict=dict(input_ids=[1,2],soft_target=[.75,.25])
        with self.assertRaises(ValueError):partition([row],[conflict])
        with self.assertRaises(ValueError):partition([row,conflict],[row])


if __name__=='__main__':unittest.main()
