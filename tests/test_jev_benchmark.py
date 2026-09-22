import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration_lab.jev_benchmark import main,parse_probability,payload,payload_digest,summarize
from calibration_lab.probability_benchmark import export,read_jsonl,records


class JevRunnerTests(unittest.TestCase):
    def test_only_visible_request_fields_are_transmitted(self):
        case={'state':'visible facts','instructions':'Is the switch on?',
              'expected_probability':.99,'hidden_event':1,'observation':[9]}
        actual=payload(case)
        self.assertEqual(set(actual),{'state','questions','model'})
        self.assertNotIn('expected_probability',json.dumps(actual))
        self.assertNotIn('hidden_event',json.dumps(actual))

    def test_noul_validation(self):
        response={'model':'test-fixture','answers':{'event':{'type':'noul','noul':.3}}}
        self.assertEqual(parse_probability(response),.3)
        for invalid in [True,float('nan'),-1,2,'0.3']:
            response['answers']['event']['noul']=invalid
            with self.assertRaises(ValueError): parse_probability(response)

    def test_preview_never_sends_and_does_not_read_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'cases';export(folder,records(17,1))
            (folder/'answers.jsonl.gz').unlink()
            with patch('sys.argv',['runner','--benchmark',str(folder),'--output',str(Path(tmp)/'responses.jsonl')]), \
                 patch('calibration_lab.jev_benchmark.send',side_effect=AssertionError('Network call')), \
                 contextlib.redirect_stdout(io.StringIO()) as capture:
                main()
            result=json.loads(capture.getvalue())
            self.assertEqual(result['network_calls'],0)
            self.assertEqual(result['pending_requests_in_limit'],24)

    def test_mock_results_are_scored_with_explicit_partial_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'cases';export(folder,records(18,1))
            public=read_jsonl(folder/'requests.jsonl.gz')
            answers={r['id']:r for r in read_jsonl(folder/'answers.jsonl.gz')}
            response_file=Path(tmp)/'mock-only.jsonl'
            rows=[]
            for case in public[:12]:
                q=answers[case['id']]['expected_probability']
                rows.append({'id':case['id'],'status':'ok','requested_model':'mock-only',
                    'payload_sha256':payload_digest(payload(case,'mock-only')),'probability':q,
                    'response':{'model':'mock-only','answers':{'event':{'type':'noul','noul':q}}}})
            response_file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=summarize(folder,response_file)
            self.assertEqual(result['successful_requests'],12)
            self.assertEqual(result['benchmark_requests'],24)
            self.assertEqual(result['returned_models'],['mock-only'])
            self.assertEqual(result['probability_rmse'],0.)
            self.assertEqual(result['paired_metrics']['copy_change']['mean'],0.)


if __name__=='__main__':
    unittest.main()
