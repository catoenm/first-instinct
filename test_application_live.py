import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scale_lab.common import ROOT,write_json
from tool_lab.application_curriculum import fixtures,continuation
from tool_lab.application_live import Worker,Episode,audit_trajectory


@unittest.skipUnless(os.environ.get('FIRST_INSTINCT_APPLICATION_WORKER_TESTS')=='1','Pinned ToolSandbox worker is opt-in')
class ApplicationWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=None;destination=os.environ.get('FIRST_INSTINCT_APPLICATION_WORKER_RESULTS')
        if destination:
            cls.folder=Path(destination);cls.folder.mkdir(parents=True,exist_ok=False)
        else:
            cls.temporary=TemporaryDirectory();cls.folder=Path(cls.temporary.name)
        cls.workers=[]

    @classmethod
    def tearDownClass(cls):
        write_json(cls.folder/'summary.json',dict(workers=[dict(journal=w.journal.name,closure=w.closed) for w in cls.workers],
            top_level_tool_calls=sum(w.count for w in cls.workers)))
        if cls.temporary:cls.temporary.cleanup()

    def worker(self,name,**limits):
        worker=Worker(ROOT/'.local/toolsandbox-venv/bin/python',ROOT/'.local/toolsandbox-upstream',
                      self.folder/(name+'.jsonl'),max_seconds=30,**limits)
        self.workers.append(worker);self.addCleanup(worker.close)
        return worker

    def case(self,**changes):
        spec=dict(ledger='none',connection='ready',goal='a',regime='hidden');spec.update(changes)
        return next(c for c in fixtures() if all(c[k]==v for k,v in spec.items()))

    def test_simultaneous_workers_do_not_share_database_state(self):
        one,two=self.worker('isolation-one'),self.worker('isolation-two')
        case=self.case();a,b=Episode(case,one),Episode(case,two)
        self.assertEqual(a.input(),b.input())
        a.step('send_a');b.step('finish')
        self.assertEqual(a.receipt()['outcome'],'completed')
        self.assertEqual(b.receipt()['outcome'],'unfinished')
        for episode in (a,b):audit_trajectory(case,episode.receipt())

    def test_live_recovery_and_prefix_failures_have_exact_returns(self):
        worker=self.worker('recovery')
        case=self.case(connection='low_battery',regime='failed_send')
        episode=Episode(case,worker)
        while not episode.done:episode.step(continuation(episode.input()))
        trace=episode.receipt();audit_trajectory(case,trace)
        self.assertEqual(trace['outcome'],'completed')
        self.assertEqual(trace['prefix'][0]['observation']['error']['type'],'ConnectionError')
        self.assertAlmostEqual(trace['reward'],1-.02*len(trace['events']))

    def test_episode_limit_prevents_another_reset(self):
        worker=self.worker('episode-cap',max_episodes=1)
        episode=Episode(self.case(),worker);episode.step('finish')
        calls=worker.count
        with self.assertRaisesRegex(RuntimeError,'episode cap'):
            Episode(self.case(),worker)
        self.assertEqual(worker.count,calls)

    def test_call_limit_stops_before_extra_execution(self):
        worker=self.worker('call-cap',max_calls=2)
        with self.assertRaisesRegex(RuntimeError,'before execution'):
            Episode(self.case(),worker)
        self.assertEqual(worker.count,2)
        self.assertEqual(worker.budget['episodes_started'],0)

    def test_arbitrary_command_is_rejected(self):
        worker=self.worker('fixed-menu');episode=Episode(self.case(),worker)
        with self.assertRaisesRegex(RuntimeError,'Illegal or post-terminal action'):
            episode.step('run arbitrary shell text')
        self.assertEqual(worker.budget['branch_calls'],0)


if __name__=='__main__':unittest.main()
