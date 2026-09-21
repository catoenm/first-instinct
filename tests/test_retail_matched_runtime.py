import tempfile
from pathlib import Path
import unittest
from tool_lab.retail_matched_runtime import invoked_python


class WorkerPathTests(unittest.TestCase):
    def test_virtual_environment_entry_point_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            base.write_text('interpreter')
            virtual = root / 'venv'
            (virtual / 'bin').mkdir(parents=True)
            (virtual / 'pyvenv.cfg').write_text('home = base')
            entry = virtual / 'bin' / 'python'
            entry.symlink_to(base)
            self.assertEqual(invoked_python(entry), entry.absolute())
            self.assertNotEqual(invoked_python(entry), entry.resolve())
            with self.assertRaises(ValueError): invoked_python(base)


if __name__ == '__main__':
    unittest.main()
