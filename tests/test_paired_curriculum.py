import unittest

from tool_lab.paired_curriculum import target, rotate, canonical
from scale_lab.common import ROOT


class PairedCurriculumTests(unittest.TestCase):
    def row(self,label):
        return dict(input=dict(state='observed',question='Question?',options=[dict(id=k,description=k.upper()) for k in 'abc']),supervision=label)

    def test_uncertain_outcomes_follow_semantic_options(self):
        row=self.row(target(list('abc'),probabilities={'a':.25,'b':.75,'c':0},semantics='outcome_distribution'))
        for shift in range(3):
            item,label,_=rotate(row,shift)
            self.assertEqual(dict(zip((o['id'] for o in item['options']),label['probabilities'])),{'a':.25,'b':.75,'c':0})
            self.assertEqual(item['state'],'observed');self.assertEqual(item['question'],'Question?')
        self.assertEqual(row['supervision']['probabilities'],[.25,.75,0])

    def test_acceptable_ties_remain_sets(self):
        row=self.row(target(list('abc'),acceptable=['a','c'],semantics='acceptable_choice_set'))
        for shift in range(3):
            item,label,_=rotate(row,shift)
            self.assertEqual({item['options'][i]['id'] for i in label['indices']},{'a','c'})
            self.assertNotIn('probabilities',label)

    def test_preference_distribution_is_not_outcome_confidence(self):
        label=target(list('abc'),probabilities={'a':.5,'b':0,'c':.5},semantics='decision_distribution')
        self.assertEqual(rotate(self.row(label),2)[1]['semantics'],'decision_distribution')

    def test_every_option_visits_every_position(self):
        row=self.row(target(list('abc'),acceptable=['a'],semantics='acceptable_choice_set'))
        seen={k:set() for k in 'abc'}
        for shift in range(3):
            item,_,_=rotate(row,shift)
            for i,o in enumerate(item['options']):seen[o['id']].add(i)
        self.assertEqual(seen,{k:{0,1,2} for k in 'abc'})

    def test_invalid_mass_or_mixed_semantics_rejected(self):
        for p in ({'a':.2,'b':.2},{'a':float('nan'),'b':1},{'a':-1,'b':2}):
            with self.assertRaises(ValueError):target(['a','b'],probabilities=p,semantics='outcome_distribution')
        with self.assertRaises(ValueError):target(['a','b'],acceptable=['a'],semantics='outcome_distribution')
        with self.assertRaises(ValueError):target(['a','b'],probabilities={'a':1,'b':0},semantics='acceptable_choice_set')

    def test_held_out_ownership_cannot_be_renamed(self):
        row=dict(self.row(target(list('abc'),acceptable=['a'],semantics='acceptable_choice_set')),id='id',group_id='group',split='development')
        with self.assertRaises(ValueError):canonical('shell','config',row,ROOT/'fixture',row['supervision'],task='decision')
        row['split']='train'
        with self.assertRaises(ValueError):canonical('shell','report',row,ROOT/'fixture',row['supervision'],task='decision')

    def test_merged_public_inputs_retain_all_ownership_aliases(self):
        row=dict(self.row(target(list('abc'),probabilities={'a':.25,'b':.75,'c':0},semantics='outcome_distribution')),
            id='id',role='train_candidate',source_members=[{'lineage':{'group_id':g}} for g in ('first','second')])
        result=canonical('source_registry','reservation',row,ROOT/'fixture',row['supervision'],task='forecast')
        self.assertEqual(result['source_group_ids'],['first','second'])
        self.assertEqual(result['ownership_component'],'reservation')
        self.assertFalse(result['training_admitted'])


if __name__=='__main__':unittest.main()
