from copy import deepcopy
import unittest
from tool_lab.revisioned_sqlite import World,execute
from tool_lab.revisioned_sqlite_audit import audit


class RevisionedSqliteTests(unittest.TestCase):
    def test_hidden_writer_preserves_identical_visible_prefix(self):
        worlds=[World('increment_latest',w,'cheap') for w in (False,True)]
        try:
            self.assertEqual(worlds[0].visible(),worlds[1].visible())
            self.assertNotEqual(worlds[0].start,worlds[1].start)
        finally:
            for w in worlds:w.close()

    def test_refusal_and_refresh_recover_without_overwriting_colleague(self):
        t=execute('increment_latest',True,'cheap',['checked_write','read','checked_write'])
        self.assertEqual(t['events'][0]['observation']['returncode'],75)
        self.assertEqual(t['events'][0]['before'],t['events'][0]['after'])
        self.assertEqual(t['final']['rows'][0],[1,16,'colleague',2])
        self.assertEqual(audit(t)['outcome'],'completed')
        t=execute('increment_latest',True,'cheap',['cached_write'])
        self.assertEqual(t['events'][0]['observation']['returncode'],0)
        self.assertEqual(audit(t)['outcome'],'incorrect')

    def test_same_command_changes_value_with_the_goal(self):
        a=execute('increment_latest',True,'cheap',['increment'])
        b=execute('approved_revision',True,'cheap',['increment'])
        self.assertEqual(a['events'][0]['input']['options'],b['events'][0]['input']['options'])
        self.assertEqual(audit(a)['outcome'],'completed')
        self.assertEqual(audit(b)['outcome'],'incorrect')
        self.assertEqual(audit(execute('approved_revision',True,'cheap',['finish']))['outcome'],'completed')

    def test_errors_retain_cost_and_do_not_supply_evidence(self):
        t=execute('increment_latest',True,'cheap',['missing_read','read','checked_write'])
        self.assertEqual(t['events'][0]['cache_before'],t['events'][0]['cache_after'])
        self.assertEqual(audit(t)['utility'],94)
        self.assertEqual(audit(t)['failed_commands'],1)

    def test_independent_auditor_rejects_corrupted_execution_and_private_input(self):
        t=execute('increment_latest',True,'cheap',['increment'])
        for mutation in ('protected','cost','input','code'):
            bad=deepcopy(t)
            if mutation=='protected':bad['events'][0]['after']['rows'][1][1]=0
            elif mutation=='cost':bad['events'][0]['cost']=0
            elif mutation=='input':bad['events'][0]['input']['hidden_writer']=True
            else:bad['events'][0]['observation']['returncode']=75
            with self.assertRaises(ValueError):audit(bad)


if __name__=='__main__':unittest.main()
