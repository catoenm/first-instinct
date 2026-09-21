import copy
import json
import unittest

import torch

from test_retail_live import make_episode
from tool_lab.retail_actor import actor_input, audit_actor_trace, collect
from tool_lab.retail_live import RetailEpisode


class Tokenizer:
    def apply_chat_template(self, messages, **unused):
        return json.dumps(messages)

    def encode(self, prompt, **unused):
        return list(prompt.encode())


class ScriptPolicy:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, rows):
        self.calls += 1
        values = []
        for row in rows:
            action = self.action(self.calls) if callable(self.action) else self.action
            values.append([10. if option == action else -10. for option in row['option_ids']])
        return torch.tensor(values), torch.zeros(len(rows)), None


def run(episodes, action, max_tokens=20000):
    policy = ScriptPolicy(action)
    return collect(policy, Tokenizer(), episodes, max_tokens, lambda: None, sample=False)


class RetailActorTests(unittest.TestCase):
    def test_identical_visible_histories_hide_different_terminal_outcomes(self):
        left, _ = make_episode('000')
        right, _ = make_episode('100')
        self.assertEqual(actor_input(left.observation()), actor_input(right.observation()))
        records, traces = run([left, right], 'stop')
        self.assertEqual([r['return'] for r in records], [0., 20.])
        self.assertEqual(traces[0]['actor_events'][0]['input'], traces[1]['actor_events'][0]['input'])
        self.assertTrue(all(r['row']['target_indices'] == [] for r in records))

    def test_horizon_charges_all_attempts_and_pays_success_once(self):
        episode, _ = make_episode('100')
        records, traces = run([episode], 'read_user')
        self.assertEqual([r['reward'] for r in records], [-2.] * 5 + [18.])
        self.assertEqual([r['return'] for r in records], [8., 10., 12., 14., 16., 18.])
        self.assertEqual(audit_actor_trace(traces[0]['receipt'], traces[0]['actor_events']), 8.)
        self.assertTrue(traces[0]['actor_events'][-1]['terminal'])
        self.assertEqual(len(records), 6)

    def test_refused_write_costs_money_without_becoming_a_success_label(self):
        original, state = make_episode('100')
        episode = RetailEpisode('profile_address', original.observation()['context'], copy.deepcopy(state),
            lambda *_: {'error_type': 'ValueError', 'message': 'documented refusal'}, lambda: state, '2', '6')
        records, traces = run([episode], lambda step: 'write_order' if step == 1 else 'stop')
        self.assertEqual([r['reward'] for r in records], [-6., 20.])
        self.assertEqual(records[0]['return'], 14.)
        self.assertEqual(traces[0]['receipt']['terminal_verdict']['success'], True)

    def test_private_fields_changed_rewards_and_invalid_histories_are_rejected(self):
        episode, _ = make_episode()
        observation = episode.observation()
        changed = [dict(observation, initial={'secret': True}),
                   dict(observation, costs={**observation['costs'], 'terminal_success': '1'}),
                   dict(observation, context={**observation['context'], 'truth': True}),
                   dict(observation, turns_remaining=5)]
        for item in changed:
            with self.assertRaises(ValueError):
                actor_input(item)

    def test_audit_rejects_self_labels_wrong_likelihoods_and_early_rewards(self):
        episode, _ = make_episode('100')
        _, traces = run([episode], lambda step: 'read_user' if step == 1 else 'stop')
        original = traces[0]
        for corruption in ('target', 'likelihood', 'reward', 'action', 'hidden'):
            trace = copy.deepcopy(original)
            event = trace['actor_events'][0]
            if corruption == 'target':
                event['row']['target_indices'] = [0]
            elif corruption == 'likelihood':
                event['old_logp'] = -5.
            elif corruption == 'reward':
                event['reward'] = 18.
            elif corruption == 'action':
                event['action'] = 'stop'
            else:
                event['input']['state'] += ' hidden initial state'
            with self.assertRaises(ValueError):
                audit_actor_trace(trace['receipt'], trace['actor_events'])

    def test_overlength_and_infrastructure_errors_are_not_training_examples(self):
        episode, _ = make_episode()
        before = episode.observation()
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            run([episode], 'stop', max_tokens=1)
        self.assertEqual(before, episode.observation())
        broken, _ = make_episode(fail=True)
        with self.assertRaises(OSError):
            run([broken], 'read_user')
        with self.assertRaises(RuntimeError):
            broken.private_record()
        with self.assertRaises(ValueError):
            run([episode, episode], 'stop')


if __name__ == '__main__':
    unittest.main()
