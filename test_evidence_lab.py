import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evidence_lab.candidates import HELD_OUT,proposals
from evidence_lab.data import (LabelOracle,VIEWS,canonical,check,file_sha,private_inputs,
                              read_rows,render,sha,visible_inputs,write_json)
from evidence_lab.evaluate import metrics
from evidence_lab.features import load
from evidence_lab.study import acquire,bootstrap_receipts,evidence_features,fit,initial_selection,predict
from evidence_lab.tasks import TASKS
from evidence_lab.worker import execute,validate


def example():
    task=TASKS[0]
    row={'id':'example','task':task.name,'description':task.description,'code':task.implementation,
         'visible_checks':check(task,task.implementation,visible_inputs(task))['checks']}
    return task,row


class EvidenceLabTests(unittest.TestCase):
    def test_contract_fixtures_and_reference_implementations(self):
        for task in TASKS:
            for args,expected in task.fixtures:
                self.assertEqual(canonical(task.reference(*args)),canonical(expected),task.name)
            # Contract fixtures and both implementation paths are checked before mutation.
            inputs=[a for a,_ in task.fixtures]
            for source in (task.implementation,task.alternative):
                observed=execute(source,inputs)
                self.assertEqual([r['error'] for r in observed],[None]*len(inputs),task.name)
                self.assertEqual(canonical([r['value'] for r in observed]),
                                 canonical([v for _,v in task.fixtures]),task.name)

    def test_splits_and_private_input_separation(self):
        self.assertEqual([sum(t.split==s for t in TASKS) for s in ('train','validation','test','new_family')],
                         [10,5,5,4])
        for t in TASKS:
            visible={canonical(a) for a in visible_inputs(t)}
            private={canonical(a) for a in private_inputs(t)}
            self.assertEqual(len(visible),5);self.assertEqual(len(private),32)
            self.assertFalse(visible&private,t.name)

    def test_mutation_reproducibility_and_mechanism_holdout(self):
        for t in TASKS:
            ordinary=proposals(t,19);shifted=proposals(t,19,8,True)
            self.assertEqual(ordinary,proposals(t,19))
            self.assertFalse({r['mechanism'] for r in ordinary}&HELD_OUT)
            self.assertTrue({r['mechanism'] for r in shifted}<=HELD_OUT)
            self.assertEqual(len(ordinary),len({r['code_sha256'] for r in ordinary}))
        code=('from evidence_lab.tasks import TASKS; from evidence_lab.candidates import proposals; '
              'from evidence_lab.data import canonical,sha; print(sha(canonical([proposals(t,19) for t in TASKS])))')
        outputs=[subprocess.check_output([sys.executable,'-c',code],env={**os.environ,'PYTHONHASHSEED':str(s)})
                 for s in (1,42)]
        self.assertEqual(*outputs)

    def test_worker_rejects_unsupported_code_and_records_exceptions(self):
        for code in ('import os\ndef solve(xs): return xs',
                     'def solve(xs): return open("x")',
                     'def solve(xs): return solve(xs)',
                     'def solve(xs): return xs.__class__',
                     'def solve(xs):\n while True: pass'):
            with self.assertRaises(ValueError):validate(code)
        t=TASKS[0];result=check(t,'def solve(xs):\n return 1 // 0\n',[[[]]])
        self.assertTrue(result['stable']);self.assertFalse(result['passed'])
        self.assertIsNone(result['worker_error']);self.assertEqual(result['checks'][0]['error'],'ZeroDivisionError')
        self.assertEqual(result['test_executions'],2)

    def test_renderer_excludes_private_metadata_and_unused_checks(self):
        _,row=example()
        baseline=[render(row,v) for v in VIEWS]
        row.update(id='hidden-id',split='hidden-split',mechanism='hidden-edit',private_verdict=False)
        row['visible_checks'][3:]=[{'secret':'never render'}]*2
        self.assertEqual(baseline,[render(row,v) for v in VIEWS])
        self.assertEqual(baseline[0].replace('0.01 cost units','0.15 cost units'),baseline[3])
        self.assertIn('copy of Check 1, not another execution',baseline[1])
        self.assertNotIn('Check 3:',baseline[0]);self.assertIn('Check 3:',baseline[2])
        ev=evidence_features([row]);np.testing.assert_array_equal(ev[:,0],ev[:,1])
        np.testing.assert_array_equal(ev[:,0],ev[:,3])

    def test_oracle_only_queries_requested_ids_and_enforces_budget(self):
        _,row=example();other={**row,'id':'second'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);oracle=LabelOracle([row,other],root/'cache',1,root/'ledger')
            with patch('evidence_lab.data.check',wraps=check) as checker:
                self.assertFalse(list((root/'cache').iterdir()))
                self.assertEqual(oracle.query(['example'],'test'),[1]);self.assertEqual(checker.call_count,1)
                with self.assertRaisesRegex(ValueError,'Repeated'):oracle.query(['example'],'test')
                with self.assertRaisesRegex(ValueError,'budget'):oracle.query(['second'],'test')
                self.assertEqual(checker.call_count,1)
            record=read_rows(root/'ledger')[0]
            self.assertEqual(record['logical_test_executions'],64);self.assertFalse(record['cache_hit'])
            cached=LabelOracle([row],root/'cache',1,root/'other-ledger')
            with patch('evidence_lab.data.check',side_effect=AssertionError('Should reuse receipt')):
                self.assertEqual(cached.query(['example'],'test'),[1])
            record=read_rows(root/'other-ledger')[0]
            self.assertTrue(record['cache_hit']);self.assertEqual(record['logical_test_executions'],64)
            self.assertEqual(record['new_execution_seconds'],0)

    def test_hash_randomness_is_repeatable_and_invalid_labels_are_quarantined(self):
        t=next(t for t in TASKS if t.name=='merge_max')
        source='def solve(a, b):\n return list(set(a) | set(b))\n'
        inputs=[[{'a':1,'b':2,'c':3,'d':4,'e':5},{}]]
        a=check(t,source,inputs);b=check(t,source,inputs)
        self.assertFalse(a['stable']);self.assertIsNotNone(a['disagreement_witness'])
        for key in ('checks','stable','result_sha256','repeat_result_sha256','disagreement_witness'):
            self.assertEqual(a[key],b[key])
        row={'id':'unstable','task':t.name,'code':source}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);oracle=LabelOracle([row],root/'cache',1,root/'ledger',allow_quarantine=True)
            self.assertEqual(oracle.query(['unstable'],'test'),[None])
            receipt=read_rows(root/'ledger')[0]
            self.assertEqual(receipt['status'],'quarantined');self.assertEqual(receipt['logical_test_executions'],64)

    def test_acquisition_is_without_replacement_and_has_no_verifier_access(self):
        rows=[{'id':str(i),'task':str(i//20),'mechanism':str(i%3)} for i in range(100)]
        chosen=initial_selection(rows,11)
        self.assertEqual(len(chosen),10)
        bootstrap=bootstrap_receipts(rows,chosen)
        self.assertEqual([r['conditional_selection_probability'] for r in bootstrap], [1/20,1/19]*5)
        q=np.full((100,4),.5)
        with patch('evidence_lab.data.check',side_effect=AssertionError('Unqueried outcome access')):
            for recipe in ('random','coverage','adaptive'):
                picks,receipts=acquire(rows,chosen,q,recipe,np.random.default_rng(3))
                self.assertEqual(len(set(picks)),16);self.assertFalse(set(picks)&set(chosen))
                self.assertTrue(all(0<r['conditional_selection_probability']<=1 for r in receipts))
        with self.assertRaises(ValueError):acquire(rows,list(range(99)),q,'random',np.random.default_rng(3))

    def test_fits_are_deterministic_and_scores_use_outcomes(self):
        x=np.array([[-1.],[-1.],[1.],[1.]]);y=np.array([0,0,1,1])
        a,b=fit(x,y),fit(x,y);np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
        self.assertLess(metrics(predict(x,a),y)['brier'],.05)
        scores=metrics(np.array([.9,.2]),np.array([1,0]))
        self.assertAlmostEqual(scores['brier'],.025);self.assertEqual(scores['accuracy'],1)
        self.assertEqual(scores['automatic_pass_count'],1);self.assertEqual(scores['automatic_pass_errors'],0)

    def test_cached_features_require_matching_text_and_checksum(self):
        _,row=example()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'features.npz';np.savez(path,features=np.zeros((1,4,2)),ids=np.array([row['id']]))
            write_json(path.with_suffix('.json'),{'features_sha256':file_sha(path),
                'text_sha256':sha('\n'.join(render(row,v) for v in VIEWS))})
            self.assertEqual(load(path,[row]).shape,(1,4,2))
            with self.assertRaisesRegex(ValueError,'rendered'):
                load(path,[{**row,'description':'different contract'}])


if __name__=='__main__':unittest.main()
