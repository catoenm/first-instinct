import unittest
from tool_lab.appworld_answer_controls import proposal


class AnswerAlternatives(unittest.TestCase):
    def test_alternative_has_the_same_public_source_field(self):
        ref={'trace':[
            dict(app='music',api='search',arguments={},response=[{'title':'first','count':2},{'title':'second','count':3}]),
            dict(app='supervisor',api='complete_task',arguments={'answer':'first'},response={})]}
        alternative=proposal(ref,1)
        self.assertEqual(alternative[1]['arguments']['answer'],'second')
        self.assertEqual(ref['trace'][1]['arguments']['answer'],'first')
        ref['trace'][0]['response']=[{'title':'first','unrelated':'second'}]
        self.assertIsNone(proposal(ref,1))


if __name__=='__main__':
    unittest.main()
