from collections import Counter
import unittest
from release_lab.balanced_plan import balanced_stream,make_schedule,CONFIG,GENERALIST_CONFIG


class BalancedTests(unittest.TestCase):
    def test_rare_groups_get_early_coverage_and_overlaps_do_not_duplicate(self):
        rows=[dict(id=str(i)) for i in range(30)]
        memberships={r['id']:['large'] for r in rows}
        memberships['0']=['rare','large'];memberships['1']=['other']
        selected=balanced_stream(rows,memberships,12,20260922)
        self.assertIn('0',selected[:3]);self.assertIn('1',selected[:3])
        self.assertEqual(len(selected),len(set(selected)))
        self.assertEqual(selected,balanced_stream(list(reversed(rows)),memberships,12,20260922))

    def test_caps_apply_globally_and_exhaustion_fails(self):
        rows=[dict(id='x'),dict(id='y')];membership={'x':['a','b'],'y':['a','b']}
        counts=Counter(balanced_stream(rows,membership,4,12,max_visits=2))
        self.assertEqual(counts,{'x':2,'y':2})
        with self.assertRaises(ValueError):balanced_stream(rows,membership,5,12,max_visits=2)

    def test_coverage_first_visits_every_question_before_repeating(self):
        rows=[dict(id=str(i)) for i in range(30)]
        memberships={r['id']:['large'] for r in rows};memberships['0']=['rare','large']
        selected=balanced_stream(rows,memberships,70,12,max_visits=3,coverage_first=True)
        self.assertEqual(len(set(selected[:30])),30)
        self.assertEqual(len(set(selected[30:60])),30)
        self.assertLessEqual(max(Counter(selected).values()),3)

    def test_semantic_forecast_types_are_separate_groups(self):
        rows=[];servers={}
        for pool,family,task,n in [('general_train','general','question',6),('tools_train','tools','tool',6),
                                   ('appworld_new_forecast','appworld','success',2),('appworld_new_forecast','appworld','change',2)]:
            for i in range(n):
                ident=pool+task+str(i);rows.append(dict(id=ident,role='train',family=family,task=task,source_refs=[{'pool':pool,'row_id':ident}]))
                if pool=='tools_train':servers[ident]=['server']
        config=dict(CONFIG,max_steps=2,per_step={'general':2,'tools':2,'verified':2})
        steps,groups=make_schedule(rows,servers,config)
        self.assertEqual([len(s) for s in steps],[6,6])
        for r in rows:
            if r['family']=='appworld':self.assertEqual(groups[r['id']],['appworld::'+r['task']])
        rows[0]['role']='development'
        with self.assertRaises(ValueError):make_schedule(rows,servers,config)

    def test_long_recipe_counts_general_replay_and_caps_tool_revisits(self):
        rows=[];servers={}
        for source,n in [('general_train',8),('tools_train',2),('retail',2)]:
            for i in range(n):
                ident=source+str(i)
                rows.append(dict(id=ident,role='train',family=source,task='question',source_refs=[{'pool':source,'row_id':ident}]))
                if source=='tools_train':servers[ident]=['server']
        config=dict(GENERALIST_CONFIG,max_steps=4,per_step={'general':2,'tools':1,'verified':1})
        steps,_=make_schedule(rows,servers,config);counts=Counter(x for step in steps for x in step)
        self.assertEqual(len(counts),12)
        self.assertEqual(sum(counts.values()),16)
        self.assertTrue(all(n==1 for key,n in counts.items() if key.startswith('general')))
        self.assertTrue(all(n<=3 for key,n in counts.items() if key.startswith('tools')))
        with self.assertRaises(ValueError):make_schedule(rows,servers,dict(config,max_steps=5))

    def test_progress_allows_learning_before_release_but_never_retention_damage(self):
        import copy
        from release_lab.balanced_pilot import learning_progress
        from release_lab.pilot_metrics import gates
        baseline=dict(general=dict(macro=dict(accuracy=.8,log_loss=.5),slices={'rules':dict(accuracy=.8)}),
            tools=dict(macro=dict(accuracy=.6,log_loss=.8)),outcomes=dict(by_group={'a':dict(brier=.3,log_loss=.4)}))
        metrics=copy.deepcopy(baseline);metrics['tools']['macro']['log_loss']=.7
        self.assertFalse(gates(metrics,baseline)['qualifies'])
        state=learning_progress(metrics,baseline,GENERALIST_CONFIG,.8,7,512)
        self.assertIsNone(state['stop_reason']);self.assertEqual(state['stale'],0)
        state=learning_progress(metrics,baseline,GENERALIST_CONFIG,.7,7,128)
        self.assertIsNone(state['stop_reason'])
        state=learning_progress(metrics,baseline,GENERALIST_CONFIG,.7,7,512)
        self.assertEqual(state['stop_reason'],'tool_log_loss_plateau_after_minimum_dose')
        metrics['general']['macro']['accuracy']=.78
        self.assertEqual(learning_progress(metrics,baseline,GENERALIST_CONFIG,.8,0,128)['stop_reason'],'retention_breach')


if __name__=='__main__':unittest.main()
