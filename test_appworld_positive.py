import copy
import unittest
from tool_lab.appworld_positive import proposals


class PositiveAlternativeTests(unittest.TestCase):
    def test_retry_keeps_the_original_call_and_uses_no_future_observation(self):
        ref=dict(points=[1],trace=[
            dict(app='example',api='show',method='get',arguments={'item_id':2},response={'item_id':2}),
            dict(app='example',api='update',method='post',arguments={'access_token':'good','item_id':2},response={'ok':True})])
        original=copy.deepcopy(ref)
        p=proposals(ref,1)
        self.assertEqual(ref,original)
        self.assertEqual(p['repeat_observation'][1],ref['trace'][0])
        self.assertEqual(p['recover_authentication'][1]['arguments']['access_token'],'invalid-local-control')
        self.assertEqual(p['recover_authentication'][2],ref['trace'][1])


if __name__=='__main__':
    unittest.main()
