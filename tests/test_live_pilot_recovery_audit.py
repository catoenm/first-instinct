import copy
import unittest

from tool_lab.live_pilot_recovery_audit import verify_learning_prefix


def fixture():
    events,ledger,rollouts=[],[],[]
    schedules={}
    for update in (1,2):
        reset=dict(task='synthetic-task',condition='world',fee_index=update)
        schedules[update]=dict(case_ids=['case'],retail=[reset],forecast_ids=['f'],replay_ids=['r'])
        rollouts.extend([
            dict(update=update,case_id='case',actor_events=[dict(encoded_input=dict(id='s',task='shell_action'))]),
            dict(update=update,reset_identity=reset,actor_events=[dict(row=dict(id='t',task='retail_live_action'))])])
        ledger.append(dict(update=update,phase='on_policy_check',transitions=2,max_absolute_probability_delta=0.))
        for component,ids in [('policy',['s','t']),('outcome',['f']),('replay',['r'])] if update==1 else [('policy',['s'])]:
            ledger.append(dict(update=update,phase='completed_backward',component=component,ids=ids))
        if update==1:
            end=dict(update=update,phase='accepted',accepted=True,physical_steps=1,
                diagnostic=dict(mean_full_kl=0.,max_full_kl=0.,by_contract={
                    k:dict(mean=0.,maximum=0.) for k in ('native','behavior')}))
            ledger.append(end)
            step=dict(end);step.pop('update');events.append(dict(update=update,step=step))
        else:
            ledger.append(dict(update=update,phase='interrupted_rejected',physical_steps=0,
                parameters_and_optimizer_restored=True,checkpoint_eligible=False,exception='DeadlineReached'))
    return events,ledger,schedules,rollouts


class PartialRecoveryTests(unittest.TestCase):
    def test_interrupted_backwards_never_become_an_accepted_update(self):
        result=verify_learning_prefix(*fixture(),arm='hybrid')
        self.assertEqual(result['accepted_updates'],1)
        self.assertEqual(result['interrupted_collected_transitions'],2)
        self.assertEqual(result['interrupted_completed_policy_backwards'],1)
        self.assertEqual(result['interrupted_physical_optimizer_steps'],0)
        self.assertFalse(result['interrupted_update_retained'])

    def test_rejects_unearned_update_or_unconfirmed_rollback(self):
        for field,value in [('physical_steps',1),('parameters_and_optimizer_restored',False),('checkpoint_eligible',True)]:
            parts=copy.deepcopy(fixture());parts[1][-1][field]=value
            with self.assertRaises(ValueError):verify_learning_prefix(*parts,arm='hybrid')

    def test_rejects_forecast_or_world_substitution(self):
        parts=copy.deepcopy(fixture())
        next(e for e in parts[1] if e.get('component')=='outcome')['ids']=['invented']
        with self.assertRaises(ValueError):verify_learning_prefix(*parts,arm='hybrid')
        parts=copy.deepcopy(fixture());parts[3][-1]['reset_identity']=dict(task='another',condition='world',fee_index=2)
        with self.assertRaises(ValueError):verify_learning_prefix(*parts,arm='hybrid')


if __name__=='__main__':unittest.main()
