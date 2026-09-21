from collections import Counter
import unittest
from release_lab.balanced_plan import balanced_stream,make_schedule,CONFIG


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


if __name__=='__main__':unittest.main()
