import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import unittest

from tool_lab.expanded_final_audit import verify_files,consumption,audit,cached_development


def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
def rows(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in value))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class ExpandedFinalAuditTests(unittest.TestCase):
    def artifact(self,base):
        root=base/'recovered';root.mkdir();write(root/'data/example.json',{'x':1})
        write(root/'artifact-hashes.json',{'data/example.json':sha(root/'data/example.json')})
        archive=base/'artifacts.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            for name in ('data/example.json','artifact-hashes.json'):tf.add(root/name,arcname=name,recursive=False)
        write(root/'cloud-collection.json',dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,verified_files=1))
        return root,archive

    def test_archive_manifest_and_local_bytes_must_agree(self):
        with TemporaryDirectory() as d:
            root,archive=self.artifact(Path(d));self.assertEqual(verify_files(root,archive)['files'],1)
            write(root/'data/example.json',{'x':2})
            with self.assertRaises(ValueError):verify_files(root,archive)
            write(root/'artifact-hashes.json',{'data/example.json':sha(root/'data/example.json')})
            with self.assertRaisesRegex(ValueError,'archived manifest'):verify_files(root,archive)

    def test_extra_local_artifacts_are_not_silently_accepted(self):
        with TemporaryDirectory() as d:
            root,archive=self.artifact(Path(d));(root/'unexpected').write_text('extra')
            with self.assertRaisesRegex(ValueError,'Unexpected'):verify_files(root,archive)

    def test_question_identity_and_model_input_identity_are_distinct(self):
        with TemporaryDirectory() as d:
            root=Path(d);data=root/'data';run=root/'run';directory=run/'reward-1507'
            case=dict(id='world',family='reservation',world=0,group_id='root')
            a=dict(id='a',input_ids=[1,2],option_ids=['left','right']);b={**a,'id':'b'}
            replay=dict(id='r',input_ids=[3,4],option_ids=['yes','no'])
            rows(data/'train-cases.jsonl',[case]);rows(data/'train-forecasts.jsonl',[]);rows(data/'replay.jsonl',[replay])
            rows(directory/'rollouts.jsonl',[dict(case_id='world',actor_events=[{'encoded_input':a},{'encoded_input':b},{'encoded_input':a}])])
            rows(directory/'learning-ledger.jsonl',[
                dict(phase='completed_backward',component='policy',ids=['a','b','a']),
                dict(phase='completed_backward',component='replay',ids=['r','r']),
                dict(phase='completed_diagnostic_backward',component='actor',ids=['a'])])
            arms={'reward-1507':{'development':{'actual_learning':{'accepted_transactions':1,'rejected_transactions':0}}}}
            result=consumption(run,data,arms)
            self.assertEqual(result['objectives']['policy'],dict(completed_presentations=3,distinct_question_ids=2,
                distinct_model_token_inputs=1,repeated_question_presentations=1))
            self.assertEqual(result['underlying_training_tasks']['world_goal_tasks'],1)
            self.assertEqual(result['diagnostic_presentations'],{'actor':1})
            self.assertEqual(result['objectives']['replay']['completed_presentations'],2)

    def test_open_study_fails_before_archive_or_predictions_are_read(self):
        with TemporaryDirectory() as d:
            root=Path(d);write(root/'cloud-collection.json',{'pipeline_status':'training','pod_deleted':False})
            with self.assertRaisesRegex(ValueError,'sealed'):audit(root,root/'absent-archive',root/'absent-parent',root/'absent-cache',None)
            self.assertIsNone(cached_development(root,'hybrid-1609',root/'absent-arm'))


if __name__=='__main__':unittest.main()
