import tempfile
from pathlib import Path
import unittest

from scale_lab.common import file_hash, write_json, write_rows
from tool_lab.evidence_consumption import consumption


class Consumption(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';run=root/'run';data.mkdir();run.mkdir()
        write_rows(data/'train-forecasts.jsonl',[dict(id='f1',group_id='root',regime='hidden'),dict(id='f2',group_id='root',regime='hidden')])
        write_rows(data/'train-cases.jsonl',[dict(id='c1',group_id='root',family='config',base={'id':'task'})])
        write_rows(data/'replay.jsonl',[dict(id=str(i)) for i in range(4)])
        freeze=dict(files={p.name:file_hash(p) for p in data.glob('*.jsonl')},adapter={'adapter_model.safetensors':'original'})
        write_json(data/'freeze.json',freeze)
        write_json(run/'run.json',dict(status='complete',arm='outcome',seed=7,recipe={'replay_rows':2,'episodes_per_update':1},
            freeze_sha256=file_hash(data/'freeze.json'),starting_adapter=freeze['adapter'],
            optimizer_steps=2,model={'id':'test-only'},selected_update=1))
        write_rows(run/'optimizer-steps.jsonl',[dict(step=1,update=1,epoch=1),dict(step=2,update=1,epoch=2)])
        return data,run

    def test_prepared_unique_and_repeated_presentations_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            data,run=self.fixture(Path(folder));r=consumption(data,run)
            self.assertEqual(r['prepared']['general_replay_pool'],4)
            self.assertEqual(r['consumed']['general_replay']['unique_ids'],2)
            self.assertEqual(r['consumed']['general_replay']['optimizer_presentations'],4)
            self.assertEqual(r['consumed']['forecasts']['unique_ids'],2)
            self.assertEqual(r['consumed']['forecasts']['optimizer_presentations'],4)
            self.assertEqual(r['consumed']['policy_transitions']['optimizer_presentations'],0)

    def test_corrupt_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data,run=self.fixture(Path(folder))
            write_rows(run/'optimizer-steps.jsonl',[dict(step=2,update=1,epoch=1)])
            with self.assertRaises(ValueError):consumption(data,run)

    def test_lineage_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data,run=self.fixture(Path(folder))
            write_rows(data/'replay.jsonl',[dict(id='changed')])
            with self.assertRaises(ValueError):consumption(data,run)


if __name__=='__main__':unittest.main()
