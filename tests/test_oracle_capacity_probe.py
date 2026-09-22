import copy
from types import SimpleNamespace
import unittest

import torch

from tests.test_general_rl import TinyTokenizer
from tests.test_decision_learning_v2 import model
from tests.test_live_contracts import ARGS, forecast_rows
from scale_lab.common import encode
from tool_lab.expanded_learning import gradient_diagnostic
from tool_lab.oracle_capacity_probe import diagnostic_records
from tool_lab.revisioned_live import collect_episode, learning_records


class OracleProbeTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(981); torch.set_num_threads(1)

    def test_real_zero_return_final_stop_reproduces_and_corrects_probe_selection(self):
        policy = model(); tokenizer = TinyTokenizer(); actions = iter(('read', 'read', 'read', 'finish'))
        def encode_input(item): return encode(tokenizer, item, 12000)
        @torch.no_grad()
        def decide(row):
            scores, values, _ = policy([row])
            return dict(action=next(actions), probabilities=scores[0].softmax(-1).tolist(), value=float(values[0]))
        receipt = collect_episode('increment_latest', False, 'cheap', decide, encode_input,
            policy_identity='fixture_diagnostic_only')
        records = learning_records(receipt, encode_input); original = copy.deepcopy(records)
        longest = max(records, key=lambda r: len(r['row']['input_ids']))
        self.assertEqual(longest['advantage'], 0.)
        with self.assertRaisesRegex(ValueError, 'Objective has no language gradient'):
            gradient_diagnostic(policy, [longest], [], [], SimpleNamespace(**ARGS), lambda: None)
        selected, note = diagnostic_records(records)
        result = gradient_diagnostic(policy, selected, forecast_rows(), [], SimpleNamespace(**ARGS), lambda: None)
        self.assertGreater(result['components']['actor']['language_l2'], 0.)
        self.assertGreater(result['components']['value']['critic_l2'], 0.)
        self.assertEqual(result['components']['value']['language_l2'], 0.)
        self.assertEqual(note['zero_advantage_records'], 1)
        self.assertFalse(note['training_rollouts_filtered']); self.assertEqual(records, original)

    def test_all_zero_advantages_fail_explicitly_without_fabricating_signal(self):
        row = dict(advantage=0., row=dict(id='zero', input_ids=[1, 2]))
        with self.assertRaisesRegex(ValueError, 'No nonzero earned advantage'): diagnostic_records([row])
        with self.assertRaises(ValueError): diagnostic_records([dict(row, advantage=float('nan'))])


if __name__ == '__main__': unittest.main()
