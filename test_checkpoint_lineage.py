import json
from pathlib import Path
import struct
import tempfile
import unittest

from tool_lab.checkpoint_lineage import compare, inventory


def write(path, values, metadata=None):
    header = {'x': {'dtype':'F32','shape':[len(values)],'data_offsets':[0,4*len(values)]}}
    if metadata is not None: header['__metadata__'] = metadata
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q',len(data))+data+struct.pack('<'+'f'*len(values),*values))


class CheckpointLineageTests(unittest.TestCase):
    def test_metadata_changes_do_not_count_as_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            write(a,[1.,2.]);write(b,[1.,2.],{'format':'pt'})
            self.assertNotEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(compare(inventory(a),inventory(b))['changed_tensors'],0)

    def test_real_payload_change_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            write(a,[1.,2.]);write(b,[1.,3.])
            result = compare(inventory(a),inventory(b))
            self.assertEqual(result['changed_tensors'],1)
            self.assertEqual(result['trainable_parameter_capacity'],2)

    def test_truncation_and_unaccounted_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'a';write(path,[1.,2.]);original=path.read_bytes()
            for changed in (original[:-1], original+b'extra'):
                path.write_bytes(changed)
                with self.assertRaises(ValueError):inventory(path)


if __name__ == '__main__': unittest.main()
