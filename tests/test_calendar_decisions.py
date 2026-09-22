import copy
import unittest

from tool_lab.calendar_decisions import (World,fixtures,branch,audit_branch,independent_transition,
    public_input,continuation,classify,epoch)


class CalendarExecutionTests(unittest.TestCase):
    def setUp(self):self.world=World();self.cases=fixtures()
    def tearDown(self):self.world.close()
    def case(self,structure='adjacency',world=0,regime='hidden',occurrence='first'):
        return next(c for c in self.cases if (c['structure'],c['world'],c['regime'],c['request']['occurrence'])==(structure,world,regime,occurrence))

    def test_asset_conversion_matches_independent_utc_and_elapsed_duration(self):
        c=self.case('fold_choice',occurrence='first');self.world.reset(c)
        result=self.world.execute('inspect')['result']['conversions']
        self.assertEqual(result['first'],dict(start_utc='2026-11-01T05:30:00+00:00',end_utc='2026-11-01T06:00:00+00:00'))
        self.assertEqual(result['second'],dict(start_utc='2026-11-01T06:30:00+00:00',end_utc='2026-11-01T07:00:00+00:00'))

    def test_adjacency_and_other_calendar_are_allowed_overlap_is_rejected(self):
        for world,expected in [(0,'completed'),(1,'completed'),(2,'unfinished'),(3,'completed')]:
            c=self.case(world=world);t=branch(c,self.world,'book_first','stop_now');audit_branch(c,t)
            self.assertEqual(t['outcome'],expected)
            if world==2:self.assertEqual(t['events'][0]['observation']['returncode'],65)

    def test_atomic_rollback_and_real_partial_autocommit_loss(self):
        c=self.case('reschedule',world=2)
        a=branch(c,self.world,'atomic_first','stop_now');b=branch(c,self.world,'sequential_first','stop_now')
        self.assertEqual(a['before'],a['after']);self.assertEqual(a['outcome'],'unfinished')
        self.assertNotEqual(b['before'],b['after']);self.assertEqual(b['outcome'],'incorrect')
        self.assertNotIn(200,[r[0] for r in b['after']['events']])
        for t in (a,b):self.assertEqual(t['events'][0]['observation']['returncode'],65);audit_branch(c,t)

    def test_same_visible_history_has_opposing_executed_outcomes(self):
        a=self.case('reschedule',world=0);b=self.case('reschedule',world=2)
        traces=[branch(c,self.world,'sequential_first','stop_now') for c in (a,b)]
        self.assertEqual(traces[0]['input'],traces[1]['input'])
        self.assertEqual({t['outcome'] for t in traces},{'completed','incorrect'})

    def test_wrong_occurrence_calendar_duration_and_collateral_are_not_success(self):
        c=self.case('fold_choice',occurrence='second')
        for action in ('book_first','wrong_calendar'):
            t=branch(c,self.world,action,'stop_now');audit_branch(c,t)
            self.assertEqual(t['outcome'],'incorrect');self.assertEqual(t['events'][0]['observation']['returncode'],0)
        t=branch(c,self.world,'book_second','stop_now')
        self.assertEqual(classify(c,t['before'],t['before']),'unfinished')
        for corruption in ('duration','calendar','unrelated','schema'):
            changed=copy.deepcopy(t['after'])
            if corruption=='schema':changed['schema']=[]
            else:
                row=next(r for r in changed['events'] if r[0]==(999 if corruption=='unrelated' else 200))
                if corruption=='duration':row[3]+=60
                elif corruption=='calendar':row[1]='personal'
                else:row[5]='damaged'
            self.assertEqual(classify(c,t['before'],changed),'incorrect')

    def test_failed_prefix_is_actual_error_and_recovery_uses_current_rows(self):
        c=self.case(regime='failed');t=branch(c,self.world,'inspect','evidence_then_complete');audit_branch(c,t)
        self.assertEqual(t['prefix'][0]['observation']['returncode'],65)
        self.assertEqual(t['prefix'][0]['observation']['result']['detail'],'FOREIGN KEY constraint failed')
        self.assertEqual(t['outcome'],'completed')
        self.assertAlmostEqual(t['cost'],.09)

    def test_fresh_evidence_stops_on_conflict_and_goals_change_selected_occurrence(self):
        c=self.case(world=2,regime='fresh');t=branch(c,self.world,'finish','stop_now')
        self.assertEqual(continuation(t['input']),'finish')
        menus=[]
        for occurrence in ('first','second'):
            c=self.case('fold_choice',regime='fresh',occurrence=occurrence)
            t=branch(c,self.world,'finish','stop_now')
            self.assertEqual(continuation(t['input']),'book_'+occurrence);menus.append(t['input']['options'])
        self.assertEqual(*menus)

    def test_tampered_ground_truth_observation_transition_and_unexpected_failure(self):
        c=self.case();t=branch(c,self.world,'inspect','evidence_then_complete');audit_branch(c,t)
        for field in ('outcome','observation','transition'):
            bad=copy.deepcopy(t)
            if field=='outcome':bad['outcome']='incorrect'
            elif field=='observation':bad['events'][0]['observation']['result']['events']=[]
            else:bad['events'][0]['after']['audit']=[]
            with self.assertRaises(ValueError):audit_branch(c,bad)
        self.world.reset(c);(self.world.root/'calendar.sqlite').write_bytes(b'not a database')
        with self.assertRaises(RuntimeError):self.world.execute('inspect')

    def test_reservation_of_whole_family_and_no_private_labels_in_input(self):
        self.assertEqual(len(self.cases),80)
        self.assertEqual({c['split'] for c in self.cases},{'reserved_transfer'})
        c=self.case();item=public_input(c,[],0);poison=copy.deepcopy(c)
        poison.update(world=999,expected_span=[0,1],initial_events=[])
        self.assertEqual(item,public_input(poison,[],0))


if __name__=='__main__':unittest.main()
