import json
from pathlib import Path
import tempfile
import unittest

from tool_lab.oracle_capacity_accounting import episodes, inspect_arm


def collection(index, mode):
    reset = dict(goal='goal', intervened=False, profile='cheap')
    return [dict(phase='episode_started', index=index, reset=reset, mode=mode),
        dict(phase='episode_completed', index=index, mode=mode, receipt=dict(trace=reset,
            actors=[dict(row=dict(id='a', input_ids=[1, 2]))]))]


class CapacityAccountingTests(unittest.TestCase):
    def test_training_evaluation_and_partial_collection_remain_separate(self):
        training = collection(13, 'sampled_behavior')+collection(14, 'sampled_behavior')[:1]
        result = episodes({'baseline-database-events.jsonl': collection(1, 'greedy_native'),
            'train-17-database-events.jsonl': training})
        self.assertEqual(result['training']['completed_episodes'], 1)
        self.assertEqual(result['training']['pending_episodes'], 1)
        self.assertEqual(result['evaluation']['completed_episodes'], 1)
        training.append(dict(phase='episode_interrupted', index=14, error='DeadlineReached'))
        result = episodes({'train-17-database-events.jsonl': training})
        self.assertEqual(result['training']['interrupted_episodes'], 1)
        self.assertEqual(result['training']['pending_episodes'], 0)

    def test_duplicate_closure_and_wrong_behavior_mode_fail(self):
        rows = collection(1, 'sampled_behavior')
        with self.assertRaises(ValueError): episodes({'train-1-database-events.jsonl': rows+rows[-1:]})
        with self.assertRaises(ValueError): episodes({'baseline-database-events.jsonl': rows})

    def test_collected_episode_is_not_counted_as_optimizer_consumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run = dict(arm='reward', status='training', selected_update=0, parent_adapter_sha256='parent',
                freeze_sha256='data', accepted_steps=0, physical_optimizer_attempts=0, updates=1)
            (directory/'run.json').write_text(json.dumps(run))
            (directory/'train-1-database-events.jsonl').write_text('\n'.join(map(json.dumps, collection(1, 'sampled_behavior')))+'\n')
            result = inspect_arm(directory)
            self.assertEqual(result['execution']['training']['collected_actor_transitions'], 1)
            self.assertEqual(result['actual_learning']['accepted_transactions'], 0)
            self.assertEqual(result['actual_learning']['components'], {})
            run.update(status='complete', accepted_steps=1)
            (directory/'run.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError): inspect_arm(directory)

    def test_teacher_presentations_and_rollback_are_both_counted(self):
        from tool_lab.live_pilot_accounting import summarize
        ledger = []
        for update, phase in ((1, 'accepted'), (2, 'rejected')):
            for event in ('started_backward', 'completed_backward'):
                ledger.append(dict(update=update, phase=event, component='teacher', ids=['a', 'b']))
            ledger.append(dict(update=update, phase='optimizer_attempt'))
            ledger.append(dict(update=update, phase=phase, physical_steps=1, accepted=phase == 'accepted',
                parameters_and_optimizer_restored=phase != 'accepted'))
        result = summarize(ledger); component = result['components']['teacher']
        self.assertEqual(result['accepted_transactions'], 1); self.assertEqual(result['rejected_transactions'], 1)
        self.assertEqual(component['completed_backward_presentations'], 4); self.assertEqual(component['unique_question_ids'], 2)
        self.assertEqual(component['presentations_in_accepted_transactions'], 2)
        self.assertEqual(component['presentations_in_rejected_transactions'], 2)


if __name__ == '__main__': unittest.main()
