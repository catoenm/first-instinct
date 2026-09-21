import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from general_lab.rl import snapshot, _hash_trainable
from test_general_rl import TinyTokenizer
from test_retail_live import make_episode
from tests.test_live_contracts import ARGS, forecast_rows, policy
from tool_lab.filesystem_decisions import fixtures
from tool_lab.expanded_runtime import reservation_cases
from tool_lab.expanded_pool import Pool
from tool_lab.live_contracts import collect_retail
from tool_lab.live_mixed import collect_shell, shell_records, validate_record, learning_step


class MixedContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1520)
        torch.set_num_threads(1)

    def collect(self, model):
        selected = [fixtures()[0], reservation_cases()[0]]
        with tempfile.TemporaryDirectory() as tmp:
            pool = Pool(Path(tmp), Path('unused'), Path('unused'), backend='catalog', workers=1)
            try:
                records, traces = collect_shell(pool, model, TinyTokenizer(), selected, 24000, lambda: None)
            finally:
                pool.close()
        return selected, records, traces

    def test_real_shell_receipts_keep_scale_one_and_uncentered_advantages(self):
        model = policy()
        selected, records, traces = self.collect(model)
        self.assertEqual({t['family'] for t in traces}, {'filesystem_scope', 'reservation'})
        for r in records:
            validate_record(r)
            self.assertEqual(r['return'], r['raw_return'])
            self.assertEqual(r['advantage'], r['return']-r['old_value'])
            self.assertEqual(r['policy_trainable_sha256'], _hash_trainable(snapshot(model)))
            self.assertEqual(r['row']['task'], 'shell_action')
        for change in ('reward', 'probability', 'observation', 'duplicate'):
            altered = copy.deepcopy(traces)
            if change == 'reward': altered[0]['actor_events'][0]['reward'] += 1
            if change == 'probability': altered[0]['actor_events'][0]['old_probabilities'][0] += .2
            if change == 'observation': altered[0]['actor_events'][0]['observation'] = {'invented': True}
            if change == 'duplicate': altered[1] = altered[0]
            with self.assertRaises(ValueError): shell_records(selected, altered)

    def test_all_three_arms_can_use_shared_actor_without_task_relabeling(self):
        for arm in ('outcome', 'reward', 'hybrid'):
            model = policy()
            _, records, _ = self.collect(model)
            retail, _ = collect_retail(model, TinyTokenizer(), [make_episode('100')[0]], 24000, lambda: None)
            records += retail
            outcomes = forecast_rows()
            replay = [{**r, 'task': 'general_replay', 'target_indices': [0]} for r in outcomes]
            probes = [r['row'] for r in records]
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            events = []
            result = learning_step(model, opt, [] if arm == 'outcome' else records,
                [] if arm == 'reward' else outcomes, replay, probes,
                SimpleNamespace(**ARGS, arm=arm), lambda: None, events.append)
            self.assertTrue(result['accepted'])
            self.assertEqual({r['row']['task'] for r in records}, {'shell_action', 'retail_live_action'})
            self.assertEqual(sum(e['phase'] == 'optimizer_attempt' for e in events), 1)
            self.assertEqual(set(result['metrics']) & {'policy_loss', 'outcome_loss'},
                {'outcome_loss'} if arm == 'outcome' else {'policy_loss'} if arm == 'reward' else {'policy_loss', 'outcome_loss'})

    def test_stale_critic_and_wrong_units_block_before_any_optimizer_attempt(self):
        model = policy()
        _, records, _ = self.collect(model)
        replay = [{**r, 'task': 'general_replay', 'target_indices': [0]} for r in forecast_rows()]
        bad = copy.deepcopy(records[0]);bad['reward_scale'] = 20.
        with self.assertRaises(ValueError): validate_record(bad)
        with torch.no_grad(): next(model.value.parameters()).add_(.1)
        events = []
        with self.assertRaises(ValueError):
            learning_step(model, torch.optim.AdamW(model.parameters(), lr=1e-5), records, [], replay,
                [r['row'] for r in records], SimpleNamespace(**ARGS, arm='reward'), lambda: None, events.append)
        self.assertFalse(events)


if __name__ == '__main__': unittest.main()
