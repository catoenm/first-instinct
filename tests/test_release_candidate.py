import copy
import json
from pathlib import Path
import tempfile
import unittest

from release_lab.candidate import export
from release_lab.pilot_plan import CONFIG,PARENT
from scale_lab.common import MODELS,file_hash,write_json


def fixture(root,probability_worse=False):
    (root/'data').mkdir(parents=True);(root/'run/checkpoint-0040/adapter').mkdir(parents=True)
    adapter=root/'run/checkpoint-0040/adapter'
    (adapter/'adapter_model.safetensors').write_bytes(b'fixture, not model weights')
    write_json(adapter/'adapter_config.json',{'fixture':True})
    (root/'data/protected.jsonl').write_text('private fixture must never enter package\n')
    freeze=dict(version='release-pilot-v1',config=CONFIG,parent_adapter_sha256=PARENT,model=MODELS['qwen35-9b'],
                sources={'scale_lab/common.py':'fixture'},label_token_ids=[1,2],pad_id=0)
    write_json(root/'data/freeze.json',freeze)
    base=dict(general=dict(n=8,macro=dict(accuracy=.8,log_loss=.5),slices={'rules':dict(accuracy=.8)}),
              tools=dict(n=8,macro=dict(accuracy=.6,log_loss=.8)),
              outcomes=dict(n=8,macro=dict(brier=.3,log_loss=.4),by_group={'a':dict(brier=.3,log_loss=.4)}))
    new=copy.deepcopy(base);new['tools']['macro']['accuracy']=.65
    if probability_worse:new['outcomes']['by_group']['a']['brier']=.31
    write_json(root/'run/0-metrics.json',base);write_json(root/'run/40-metrics.json',new)
    write_json(root/'run/selected.json',dict(step=40,source='checkpoint-0040/adapter',qualifies=True,
                                          adapter_sha256=file_hash(adapter/'adapter_model.safetensors')))
    write_json(root/'run/result.json',dict(selected_step=40,completed_steps=80,gates=dict(qualifies=True),
        consumed_presentations=5120,unique_consumed_questions=5100,freeze_sha256=file_hash(root/'data/freeze.json')))
    for name in ('runtime-qualification.json','restart-qualification.json'):write_json(root/'run'/name,dict(status='passed'))
    write_json(root/'artifact-hashes.json',{str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()})
    write_json(root/'cloud-collection.json',dict(pod_deleted=True))


class CandidateTests(unittest.TestCase):
    def test_failed_probability_gate_cannot_be_overridden_by_summary(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);fixture(root/'recovered',probability_worse=True)
            with self.assertRaisesRegex(ValueError,'advancement gates'):export(root/'recovered',root/'export')
            self.assertFalse((root/'export').exists())

    def test_export_excludes_private_examples_and_distinguishes_selection_from_latest(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);fixture(root/'recovered');export(root/'recovered',root/'export')
            files={str(p.relative_to(root/'export')) for p in (root/'export').rglob('*') if p.is_file()}
            self.assertEqual(files,{'best/adapter_config.json','best/adapter_model.safetensors','README.md','run.json','development-summary.json','package-hashes.json'})
            r=json.loads((root/'export/run.json').read_text())
            self.assertEqual(r['selected_optimizer_step'],40);self.assertEqual(r['completed_optimizer_steps'],80)
            self.assertEqual(r['selected_training_presentations'],2560)
            self.assertFalse(r['release_evaluation_complete']);self.assertEqual(r['status'],'candidate')

    def test_changed_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);fixture(root/'recovered')
            (root/'recovered/run/checkpoint-0040/adapter/adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):export(root/'recovered',root/'export')

    def test_balanced_version_has_explicit_supported_identity(self):
        from release_lab.balanced_plan import CONFIG as balanced
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);recovered=root/'recovered';fixture(recovered)
            freeze=json.loads((recovered/'data/freeze.json').read_text());freeze.update(version='release-balanced-v1',config=balanced)
            write_json(recovered/'data/freeze.json',freeze)
            result=json.loads((recovered/'run/result.json').read_text());result['freeze_sha256']=file_hash(recovered/'data/freeze.json')
            write_json(recovered/'run/result.json',result)
            hashes=json.loads((recovered/'artifact-hashes.json').read_text())
            for file in ('data/freeze.json','run/result.json'):hashes[file]=file_hash(recovered/file)
            write_json(recovered/'artifact-hashes.json',hashes)
            summary=export(recovered,root/'export')
            self.assertIn('release-balanced-v1',summary['version'])


if __name__=='__main__':unittest.main()
