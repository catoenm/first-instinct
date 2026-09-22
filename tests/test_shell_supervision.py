from copy import deepcopy
import itertools
import json
import unittest

from tool_lab.shell_supervision import make_case, split_for, verify


class ShellSupervisionTests(unittest.TestCase):
    def test_combination_splits_and_seed_lineage(self):
        groups = {'train': set(), 'validation': set(), 'test': set()}
        for family, bits in itertools.product(('config', 'sqlite', 'report'), itertools.product((0, 1), repeat=3)):
            a, b = make_case(family, bits, 11), make_case(family, bits, 12)
            self.assertEqual(a['group_id'], b['group_id'])
            self.assertEqual(a['split'], split_for(bits))
            self.assertEqual(len(a['commands']), 6)
            groups[a['split']].add(a['group_id'])
        self.assertEqual({k:len(v) for k,v in groups.items()}, {'train':12,'validation':6,'test':6})
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['train'] & groups['validation'])

    def test_verifier_rejects_type_changes_protected_changes_and_extra_files(self):
        case = make_case('config', (1, 1, 1), 17)
        before = {'base.json': {'sha256': 'base'}, 'reference.txt': {'sha256': 'reference'}, 'overlay.json': {'sha256': 'old'}}
        after = deepcopy(before)
        after['overlay.json'] = {'text': json.dumps(case['expected']['expected']), 'sha256': 'changed'}
        branch = {'before':before,'after':after,'returncode':0}
        self.assertTrue(verify(case, branch))
        broken = deepcopy(branch); broken['after']['base.json']['sha256'] = 'changed'
        self.assertFalse(verify(case, broken))
        broken = deepcopy(branch); broken['after']['unexpected.txt'] = {'sha256':'extra'}
        self.assertFalse(verify(case, broken))
        broken = deepcopy(branch); value = json.loads(broken['after']['overlay.json']['text'])
        service = next(k for k in value if k != 'unrelated'); value[service]['enabled'] = 0
        broken['after']['overlay.json']['text'] = json.dumps(value)
        self.assertFalse(verify(case, broken))

    def test_exit_code_is_not_success_label(self):
        case = make_case('report', (0, 0, 0), 18)
        before = {'reference.txt':{'sha256':'a'},'sales.csv':{'sha256':'b'}}
        self.assertFalse(verify(case, {'before':before,'after':deepcopy(before),'returncode':0}))
        after = deepcopy(before); after['report.json'] = {'text':json.dumps(case['expected']['expected'])}
        self.assertTrue(verify(case, {'before':before,'after':after,'returncode':1}))


if __name__ == '__main__': unittest.main()
