import copy
import unittest

from tool_lab.appworld_transfer_evaluation_report import checks


class TransferGateTests(unittest.TestCase):
    def fixture(self):
        base={'transfer':{'decision':{'macro':{'return':.1}},'success':{'macro':{'brier':.5}}},
              'retention':{'macro':{'accuracy':.9,'log_loss':.2}},
              'format_original':{'decision':{'macro':{'return':.2}},'success':{'macro':{'brier':.4}}},
              'format_shared':{'decision':{'macro':{'return':.2}},'success':{'macro':{'brier':.4}}}}
        result={'original':base,'selected40':copy.deepcopy(base)}
        result['selected40']['transfer']['decision']['macro']['return']=.2
        result['selected40']['transfer']['success']['macro']['brier']=.3
        return result

    def test_all_gates_are_required(self):
        self.assertTrue(checks(self.fixture())['joint_passed'])
        for group,key,value in [('transfer','direct',-.2),('transfer','brier',.6),
                                ('retention','accuracy',.8),('retention','log_loss',.4),
                                ('format','return',-.2),('format','brier',.6)]:
            data=self.fixture()
            if group=='transfer':
                label,metric=('decision','return') if key=='direct' else ('success','brier')
                data['selected40']['transfer'][label]['macro'][metric]=value
            elif group=='retention': data['selected40']['retention']['macro'][key]=value
            else:
                label='decision' if key=='return' else 'success'
                data['original']['format_shared'][label]['macro'][key]=value
            self.assertFalse(checks(data)['joint_passed'])


if __name__=='__main__':unittest.main()
