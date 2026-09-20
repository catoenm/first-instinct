from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors.numpy import save_file

from tool_lab.expanded_checkpoint_audit import tensors,tensor_hash,delta,match_change,audit


class ExpandedCheckpointTests(unittest.TestCase):
    def test_saved_names_reconstruct_training_hash(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'adapter.safetensors';x=np.array([[1.,2.]],dtype=np.float32)
            save_file({'layer.lora_A.weight':x},str(p));v=tensors(p)
            h=hashlib.sha256();h.update(b'layer.lora_A.default.weight');h.update(x.tobytes())
            self.assertEqual(tensor_hash(v),h.hexdigest())
            self.assertEqual(list(v),['layer.lora_A.default.weight'])

    def test_exact_change_counts_and_independent_norm(self):
        before={'a':np.array([1.,2.,3.],dtype=np.float32)};after={'a':np.array([1.,5.,7.],dtype=np.float32)}
        actual=delta(after,before)
        self.assertEqual(actual['changed_elements'],2);self.assertEqual(actual['changed_tensors'],1)
        self.assertEqual(actual['l2_delta'],5.);self.assertEqual(actual['max_absolute_delta'],4.)
        claimed={k:v for k,v in actual.items() if k!='changed_names'};claimed['examples']=['a'];match_change(actual,claimed)
        for key,value in [('changed_elements',1),('l2_delta',4.),('examples',['unchanged'])]:
            bad=deepcopy(claimed);bad[key]=value
            with self.assertRaises(ValueError):match_change(actual,bad)

    def test_tensor_names_shapes_and_nonfinite_values_cannot_hide(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'bad.safetensors'
            for name,values in [('ordinary.weight',[1.]),('layer.lora_A.weight',[float('nan')])]:
                save_file({name:np.array(values,dtype=np.float32)},str(p))
                with self.assertRaises(ValueError):tensors(p)
        with self.assertRaises(ValueError):delta({'a':np.zeros(2)}, {'a':np.zeros(3)})
        with self.assertRaises(ValueError):delta({'a':np.zeros(2)}, {'b':np.zeros(2)})

    def test_open_study_cannot_trigger_checkpoint_or_prediction_reads(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'cloud-collection.json').write_text('{"pipeline_status":"training","pod_deleted":false}')
            with self.assertRaisesRegex(ValueError,'sealed'):audit(root,root/'nonexistent-adapter')


if __name__=='__main__':unittest.main()
