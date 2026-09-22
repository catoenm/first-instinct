"""Offline gradient, return-accounting and checkpoint controls for outcome v2.

Tiny networks exercise the real loss/optimizer code. No foundation download,
model service, graphics processor or paid call is used.
"""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from general_lab import outcome_train as train
from general_lab.rl import DeadlineReached, snapshot
from scale_lab.common import MODELS, ROOT, file_hash


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return json.dumps(messages)

    def encode(self, text, **_kwargs):
        return [1 + ord(character) % 39 for character in text]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / 'tiny-tokenizer.json').write_text('{}')


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(80, 8)
        self.transform = nn.Linear(8, 8)
        self.dropout = nn.Dropout(.3)
        self.lm_head = nn.Linear(8, 80, bias=False)
        # Only the internal transformation is adapted; vocabulary layers stay fixed.
        self.embedding.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        hidden = self.dropout(torch.tanh(self.transform(pooled)))[:, None, :]
        return SimpleNamespace(logits=self.lm_head(hidden))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / 'tiny-state.pt')


def args(**changes):
    defaults = dict(arm='hybrid', batch_size=2, clip=.2, value_weight=.5,
                    entropy_weight=.01, replay_weight=.25, cost_weight=.25,
                    learning_rate=.003, value_learning_rate=.005,
                    max_tokens=10000, seed=19, epochs_per_update=2)
    return SimpleNamespace(**(defaults | changes))


def row(index, task='workflow/outcome'):
    return {'id': f'row-{index}-{task}', 'root_id': f'root-{index}',
            'group_id': 'fixture-group', 'task': task,
            'input_ids': [1 + index % 5, 8, 11 + index % 7],
            'option_ids': ['left', 'right'], 'target_indices': [index % 2]}


def records_for(policy, rows=None):
    rows = rows or [row(i) for i in range(3)]
    policy.eval()
    with torch.no_grad():
        logits, values, _ = policy(rows)
        probabilities = logits.softmax(-1)
    return [{'row': item, 'action': index % 2,
             'old_logp': float(probabilities[index, index % 2].log()),
             'old_probabilities': probabilities[index].tolist(),
             'old_value': float(values[index]), 'return': .3 + index * .2,
             'reward': .3 + index * .2, 'advantage': (-1.) ** index * .8}
            for index, item in enumerate(rows)]


@dataclass(frozen=True)
class ForcedScenario:
    name: str = 'forced-return-fixture'


@dataclass(frozen=True)
class ForcedTape:
    fixed: int = 1


@dataclass(frozen=True)
class ForcedObservation:
    tick: int
    terminal: bool


class ForcedEpisode:
    closed = []

    def __init__(self, _scenario, _tape):
        self.tick = 0

    def observe(self):
        return ForcedObservation(self.tick, self.tick == 2)

    def input(self):
        options = ['a', 'b'] if self.tick == 0 else ['finish']
        return {'state': 'Visible two-step fixture.', 'question': 'Choose an action.',
                'options': [{'id': x, 'description': x} for x in options]}

    def step(self, action):
        assert action in [x['id'] for x in self.input()['options']]
        self.tick += 1
        return {'observation': self.observe(), 'reward_cents': -10 if self.tick == 1 else 80,
                'terminal': self.observe().terminal}

    def truth_receipt(self):
        return {'return_cents': 70}

    def close(self):
        self.closed.append(self)


class OutcomeGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.policy = train.DetachedValuePolicy(TinyLanguage(), list(range(40, 76)), 0, 'cpu')
        self.args = args()
        self.check = lambda: None

    def test_nonzero_critic_weights_still_cannot_change_language_gradients(self):
        # A zero-initialized critic would hide an accidental connection on its
        # first backward. Deliberately make the value weights nonzero first.
        with torch.no_grad():
            self.policy.value.weight.fill_(.7)
        records = records_for(self.policy)
        self.policy.zero_grad(set_to_none=True)
        train.component_loss(self.policy, 'value', records, self.args).backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.policy.language.parameters()))
        self.assertGreater(float(self.policy.value.weight.grad.norm()), 0.)

    def test_actor_and_forecast_losses_change_internal_transform_not_frozen_head(self):
        records = records_for(self.policy)
        for component, examples in [('actor', records), ('outcome', [row(i) for i in range(3)]),
                                    ('cost', [row(i, 'workflow/cost') for i in range(3)])]:
            with self.subTest(component=component):
                self.policy.zero_grad(set_to_none=True)
                train.component_loss(self.policy, component, examples, self.args).backward()
                self.assertGreater(float(self.policy.language.transform.weight.grad.norm()), 0.)
                self.assertIsNone(self.policy.language.embedding.weight.grad)
                self.assertIsNone(self.policy.language.lm_head.weight.grad)
                self.assertTrue(all(p.grad is None for p in self.policy.value.parameters()))

    def test_uneven_microbatches_match_the_whole_batch_gradient(self):
        examples = [row(i) for i in range(5)]
        weight = .37
        self.policy.zero_grad(set_to_none=True)
        loss = train.component_loss(self.policy, 'outcome', examples, self.args)
        (weight * loss).backward()
        expected = train.gradient_vector([p for p in self.policy.language.parameters() if p.requires_grad])
        self.policy.zero_grad(set_to_none=True)
        measured = train.backward_component(self.policy, 'outcome', examples, self.args, self.check, weight)
        actual = train.gradient_vector([p for p in self.policy.language.parameters() if p.requires_grad])
        self.assertAlmostEqual(measured, float(loss.detach()), places=6)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-5))

    def test_gradient_diagnostic_audits_all_rows_and_preserves_every_parameter(self):
        with torch.no_grad():
            self.policy.value.weight.fill_(.2)
        records = records_for(self.policy)
        outcomes, costs, replay = [row(i) for i in range(5)], [row(i + 3) for i in range(4)], [row(8)]
        before = deepcopy(self.policy.state_dict())
        diagnostic = train.gradient_diagnostic(self.policy, records, outcomes, costs, replay, self.args, self.check)
        self.assertEqual({k: x['rows'] for k, x in diagnostic['components'].items()},
                         {'actor': 3, 'value': 3, 'outcome': 5, 'cost': 4, 'replay': 1})
        self.assertEqual(diagnostic['components']['value']['language_l2'], 0.)
        self.assertGreater(diagnostic['components']['value']['critic_l2'], 0.)
        for component in ('actor', 'outcome', 'cost', 'replay'):
            self.assertGreater(diagnostic['components'][component]['language_l2'], 0.)
            self.assertEqual(diagnostic['components'][component]['critic_l2'], 0.)
        self.assertTrue(diagnostic['weights_unchanged'])
        self.assertEqual(diagnostic['optimizer_steps'], 0)
        for name, parameter in self.policy.state_dict().items():
            self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assertTrue(all(p.grad is None for p in self.policy.parameters()))

    def test_forced_singleton_rewards_contribute_to_previous_policy_returns(self):
        ForcedEpisode.closed = []
        module = SimpleNamespace(EpisodeAdapter=ForcedEpisode)
        def world(*_args, **_kwargs):
            return module, ForcedScenario(), ForcedTape(), 'forced-root'
        with patch.object(train, 'ENVIRONMENTS', ('fixture',)), patch.object(train, 'root_world', world):
            records, traces = train.collect(self.policy, TinyTokenizer(), 11, 1, 1, 10000, 2, self.check)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]['reward'], -.1)
        self.assertAlmostEqual(records[0]['return'], .7)
        self.assertAlmostEqual(traces[0]['return'], .7)
        self.assertEqual([s['forced_public_action'] for s in traces[0]['steps']], [False, True])
        self.assertEqual(traces[0]['steps'][1]['probabilities'], {'finish': 1.})
        self.assertEqual(len(ForcedEpisode.closed), 1)

    def test_post_step_commit_callback_survives_audit_interrupt_with_changed_weights(self):
        records = records_for(self.policy)
        optimizer = torch.optim.SGD(self.policy.parameters(), lr=.1)
        before = snapshot(self.policy.language)
        commits = []
        with patch.object(train, 'rollout_kl', side_effect=DeadlineReached('during post-step audit')):
            with self.assertRaises(DeadlineReached):
                train.update(self.policy, optimizer, records, [row(3)], [row(4)], [row(5)],
                             self.args, self.check, lambda metrics: commits.append(metrics))
        self.assertEqual(len(commits), 1)
        self.assertGreater(commits[0]['language_gradient_norm'], 0.)
        self.assertNotIn('post_step', commits[0])
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in snapshot(self.policy.language).items()))

    def test_actor_only_update_increases_probability_of_rewarded_action(self):
        item = row(0)
        records = records_for(self.policy, [item])
        records[0]['advantage'] = 1.
        old = records[0]['old_probabilities'][0]
        pure_actor = args(arm='reward', value_weight=0., entropy_weight=0., replay_weight=0.)
        result = train.update(self.policy, torch.optim.SGD(self.policy.parameters(), lr=.1),
                              records, [], [], [], pure_actor, self.check)
        with torch.no_grad():
            current = float(self.policy([item])[0].softmax(-1)[0, 0])
        self.assertGreater(current, old)
        self.assertGreater(result['post_step']['mean_full_kl'], 0.)

    def test_post_step_divergence_matches_all_options_not_only_sampled_action(self):
        records = records_for(self.policy)
        unchanged = train.rollout_kl(self.policy, records, self.args, self.check)
        self.assertAlmostEqual(unchanged['mean_full_kl'], 0., places=6)
        self.assertAlmostEqual(unchanged['mean_sampled_action_ratio'], 1., places=6)
        with torch.no_grad():
            self.policy.language.transform.bias.add_(.5)
            logits = self.policy([r['row'] for r in records])[0]
        expected = []
        for record, values in zip(records, logits.log_softmax(-1)):
            p = torch.tensor(record['old_probabilities'])
            expected.append(float((p * (p.log() - values)).sum()))
        actual = train.rollout_kl(self.policy, records, self.args, self.check)
        self.assertAlmostEqual(actual['mean_full_kl'], sum(expected) / len(expected), places=6)
        self.assertAlmostEqual(actual['max_full_kl'], max(expected), places=6)

    def test_forward_accounting_counts_real_rows_and_unpadded_tokens(self):
        first = row(0)
        second = row(1)
        second['input_ids'] = [2, 4, 6, 8, 10]
        self.policy([first, second])
        self.policy([second])
        self.assertEqual(self.policy.forward_counts, {'batches': 2, 'questions': 3, 'input_tokens': 13})

    def test_language_and_critic_clipping_coefficients_match_independent_gradient_bounds(self):
        records = records_for(self.policy)
        for record in records:
            record['advantage'] *= 100.
            record['return'] *= 100.
        result = train.update(self.policy, torch.optim.SGD(self.policy.parameters(), lr=.001),
                              records, [], [], [], args(arm='reward', value_weight=10., replay_weight=0.), self.check)
        for component in ('language', 'critic'):
            norm = result[component + '_gradient_norm']
            self.assertGreater(norm, 1.)
            self.assertAlmostEqual(result[component + '_clip_coefficient'], 1. / (norm + 1e-6), places=7)
        self.assertLessEqual(float(train.gradient_vector(list(self.policy.language.parameters())).norm()), 1.00001)
        self.assertLessEqual(float(train.gradient_vector(list(self.policy.value.parameters())).norm()), 1.00001)

    def test_retention_gate_uses_macro_metrics_and_both_limits(self):
        baseline = {'macro_log_loss': .3, 'macro_accuracy': .8, 'accuracy': .2}
        self.assertTrue(train.eligible({'macro_log_loss': .4, 'macro_accuracy': .77, 'accuracy': 0.}, baseline))
        self.assertFalse(train.eligible({'macro_log_loss': .401, 'macro_accuracy': .99}, baseline))
        self.assertFalse(train.eligible({'macro_log_loss': .1, 'macro_accuracy': .769}, baseline))
        self.assertFalse(train.eligible({'macro_log_loss': math.nan, 'macro_accuracy': .99}, baseline))


class TrainingHarness:
    """Exercise real train control flow with a tiny saved network and local rows."""
    def __init__(self, root, arm='hybrid'):
        self.root = Path(root)
        self.data, self.raw_data, self.adapter = (self.root / name for name in ('prepared', 'raw', 'adapter'))
        for path in (self.data, self.raw_data, self.adapter):
            path.mkdir()
        (self.adapter / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': MODELS['qwen35-9b']['id']}))
        for name in ('validation-audit.jsonl', 'test-audit.jsonl'):
            (self.raw_data / name).write_text('')
        self.raw = {'outputs': {p.name: file_hash(p) for p in self.raw_data.glob('*.jsonl')},
                    'source_sha256': {'general_lab/workflow_environment.py': file_hash(ROOT / 'general_lab/workflow_environment.py')}}
        (self.raw_data / 'manifest.json').write_text(json.dumps(self.raw))
        examples = [row(i, task) for i in range(6) for task in ('workflow/outcome', 'workflow/cost')]
        self.write_rows('train.jsonl', examples)
        self.write_rows('replay.jsonl', [row(i + 10, 'general') for i in range(6)])
        self.write_rows('retention.jsonl', [row(30, 'general')])
        manifest = {'model': MODELS['qwen35-9b'], 'label_token_ids': list(range(40, 76)),
                    'outputs': {p.name: file_hash(p) for p in self.data.glob('*.jsonl')},
                    'raw_manifest_sha256': file_hash(self.raw_data / 'manifest.json')}
        (self.data / 'manifest.json').write_text(json.dumps(manifest))
        self.args = args(adapter=self.adapter, data=self.data, raw_data=self.raw_data, output=self.root / 'run',
                         arm=arm, device='cpu', max_updates=2, forecast_worlds=2, episodes_per_environment=1,
                         replay_rows=2, max_hours=1., max_kl=100., eval_every=1,
                         validation_worlds=1, test_worlds=1, patience=10, min_updates=2)
        self.model = TinyLanguage()
        self.evaluations, self.epoch_inputs, self.test_receipts = [], [], []
        self.validation_rewards = iter([.1, .2, .3])

    def write_rows(self, name, rows):
        (self.data / name).write_text(''.join(json.dumps(value) + '\n' for value in rows))

    def freeze(self):
        # The loading function is mocked; these bytes exercise the real artifact
        # hash gate without writing a large model or parsing fake weights.
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'tiny-adapter-fixture')
        self.args.freeze = self.root / 'freeze.json'
        receipt = {
            'prepared_manifest_sha256': file_hash(self.data / 'manifest.json'),
            'starting_adapter_sha256': file_hash(self.adapter / 'adapter_model.safetensors'),
            'starting_adapter_files_sha256': {path.name: file_hash(path) for path in self.adapter.iterdir() if path.is_file()},
            'files': {name: file_hash(ROOT / name) for name in
                      ('general_lab/outcome_train.py', 'general_lab/workflow_environment.py', 'docs/outcome-v2-protocol.md')},
        }
        self.args.freeze.write_text(json.dumps(receipt))
        return receipt

    def fake_collect(self, policy, *_args, **_kwargs):
        return records_for(policy), [{'id': 'tiny-rollout', 'steps': [], 'return': .3}]

    def evaluate(self, _predictor, split, _count, **_kwargs):
        receipt = json.loads((self.args.output / 'run.json').read_text())
        self.evaluations.append((split, receipt['status'], receipt.get('selected_update')))
        if split == 'test':
            self.test_receipts.append(receipt)
        value = next(self.validation_rewards) if split == 'validation' else .123
        return {'controller_reward': value}, [], []

    def run(self, *, update=None, retention=None, diagnostic=None):
        original_update = train.update
        def recorded_update(policy, optimizer, records, outcomes, costs, replay, params, check, on_commit=None):
            self.epoch_inputs.append({'arm': params.arm, 'learning_rates': [group['lr'] for group in optimizer.param_groups],
                                      'outcomes': [r['id'] for r in outcomes], 'costs': [r['id'] for r in costs],
                                      'replay': [r['id'] for r in replay]})
            return (update or original_update)(policy, optimizer, records, outcomes, costs, replay, params, check, on_commit)
        with ExitStack() as stack:
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained', return_value=TinyTokenizer()))
            stack.enter_context(patch.object(train, 'label_token_ids', return_value=list(range(40, 76))))
            stack.enter_context(patch.object(train, 'load_model', return_value=self.model))
            stack.enter_context(patch.object(train, 'collect', side_effect=self.fake_collect))
            stack.enter_context(patch.object(train, 'update', side_effect=recorded_update))
            stack.enter_context(patch.object(train, 'gradient_diagnostic', side_effect=diagnostic or (lambda *_args: {'fixture': True})))
            stack.enter_context(patch.object(train, 'evaluate_rows', return_value=[]))
            stack.enter_context(patch.object(train, 'macro_metrics', side_effect=retention or (lambda _rows: {'macro_log_loss': .3, 'macro_accuracy': .8})))
            stack.enter_context(patch('general_lab.outcome_evaluate.evaluate_suite', side_effect=self.evaluate))
            stack.enter_context(patch('peft.utils.save_and_load.load_peft_weights', side_effect=lambda path, **_kw: torch.load(Path(path) / 'tiny-state.pt', weights_only=True)))
            stack.enter_context(patch('peft.set_peft_model_state_dict', side_effect=lambda model, weights: model.load_state_dict(weights)))
            stack.enter_context(patch.object(train.signal, 'signal'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return train.train(self.args)


class OutcomeTrainingControlTests(unittest.TestCase):
    def test_test_inference_occurs_only_after_selection_and_archived_checkpoints_match(self):
        with TemporaryDirectory() as directory:
            harness = TrainingHarness(directory)
            result = harness.run()
            self.assertEqual(result['status'], 'complete')
            self.assertEqual((result['updates'], result['optimizer_steps'], result['selected_update']), (2, 4, 2))
            self.assertEqual(result['selected_optimizer_steps'], 4)
            self.assertEqual(result['fully_completed_updates'], 2)
            self.assertIsNone(result['partial_update'])
            self.assertEqual([e[0] for e in harness.evaluations], ['validation', 'validation', 'validation', 'test'])
            self.assertEqual(harness.evaluations[-1][1:], ('final_evaluation', 2))
            locked = harness.test_receipts[0]
            self.assertEqual(locked['selected_optimizer_steps'], 4)
            self.assertEqual(locked['fully_completed_updates'], 2)
            self.assertIsNone(locked['partial_update'])
            latest = torch.load(harness.args.output / 'latest/tiny-state.pt', weights_only=True)
            best = torch.load(harness.args.output / 'best/tiny-state.pt', weights_only=True)
            self.assertTrue(all(torch.equal(latest[name], best[name]) for name in latest))
            self.assertFalse(result['selection']['test_used'])
            self.assertEqual(result['code_sha256']['general_lab/outcome_train.py'], file_hash(ROOT / 'general_lab/outcome_train.py'))

    def test_kl_guard_stops_after_committed_epoch_and_records_partial_update(self):
        with TemporaryDirectory() as directory:
            harness = TrainingHarness(directory)
            original = train.update
            def guarded(*positional):
                result = original(*positional)
                result['post_step']['mean_full_kl'] = .5
                return result
            harness.args.max_kl = .02
            result = harness.run(update=guarded)
            self.assertEqual(result['status'], 'early_stopped_complete')
            self.assertEqual(result['stop_reason'], 'post_step_kl_guard')
            self.assertEqual((result['updates'], result['optimizer_steps']), (1, 1))
            self.assertEqual(result['fully_completed_updates'], 0)
            self.assertEqual(result['partial_update'], {'update': 1, 'completed_epochs': 1})
            ledger = [json.loads(line) for line in (harness.args.output / 'optimizer-steps.jsonl').read_text().splitlines()]
            self.assertEqual([(row['update'], row['epoch'], row['optimizer_step']) for row in ledger], [(1, 1, 1)])
            self.assertEqual(len(harness.epoch_inputs), 1)
            locked = harness.test_receipts[0]
            self.assertEqual(locked['selected_optimizer_steps'], 1)
            self.assertEqual(locked['fully_completed_updates'], 0)
            self.assertEqual(locked['partial_update'], {'update': 1, 'completed_epochs': 1})

    def test_interrupt_after_first_optimizer_step_saves_changed_checkpoint_and_receipt(self):
        with TemporaryDirectory() as directory:
            harness = TrainingHarness(directory)
            before = deepcopy(harness.model.state_dict())
            with patch.object(train, 'rollout_kl', side_effect=DeadlineReached('interrupted after commit')):
                result = harness.run()
            self.assertEqual(result['status'], 'bounded_stop')
            self.assertEqual((result['updates'], result['optimizer_steps'], result['fully_completed_updates']), (1, 1, 0))
            self.assertEqual(result['selected_update'], 0)
            self.assertEqual(result['partial_update'], {'update': 1, 'completed_epochs': 1})
            self.assertEqual([x[0] for x in harness.evaluations], ['validation'])
            latest = torch.load(harness.args.output / 'latest/tiny-state.pt', weights_only=True)
            self.assertTrue(any(not torch.equal(latest[name], before[name]) for name in before))
            self.assertGreater(result['language_parameter_audit']['changed_elements'], 0)
            self.assertEqual(len((harness.args.output / 'optimizer-steps.jsonl').read_text().splitlines()), 1)

    def test_interrupt_after_last_epoch_counts_full_committed_update_without_claiming_evaluation(self):
        with TemporaryDirectory() as directory:
            harness = TrainingHarness(directory)
            audits = [0]
            original = train.rollout_kl
            def interrupt_second_audit(*positional):
                audits[0] += 1
                if audits[0] == 2:
                    raise DeadlineReached('interrupted after second committed epoch')
                return original(*positional)
            with patch.object(train, 'rollout_kl', side_effect=interrupt_second_audit):
                result = harness.run()
            self.assertEqual(result['status'], 'bounded_stop')
            self.assertEqual((result['updates'], result['optimizer_steps'], result['fully_completed_updates']), (1, 2, 1))
            self.assertIsNone(result['partial_update'])
            self.assertEqual(result['selected_optimizer_steps'], 0)
            self.assertEqual(harness.test_receipts, [])
            self.assertEqual(len((harness.args.output / 'optimizer-steps.jsonl').read_text().splitlines()), 2)

    def test_retention_failure_preserves_update_zero_despite_higher_reward(self):
        with TemporaryDirectory() as directory:
            harness = TrainingHarness(directory)
            calls = [0]
            def retention(_rows):
                calls[0] += 1
                return {'macro_log_loss': .3 if calls[0] == 1 else .5, 'macro_accuracy': .8}
            result = harness.run(retention=retention)
            self.assertEqual(result['selected_update'], 0)
            self.assertEqual(result['selected_optimizer_steps'], 0)
            self.assertEqual(result['best_validation_controller_reward'], .1)
            events = [json.loads(line) for line in (harness.args.output / 'training.jsonl').read_text().splitlines()]
            self.assertTrue(all(not event['retention_eligible'] for event in events))
            self.assertEqual([x[0] for x in harness.evaluations], ['validation', 'validation', 'validation', 'test', 'test'])

    def test_arms_share_learning_rates_forecast_world_order_and_replay_schedule(self):
        traces = {}
        for arm in ('outcome', 'reward', 'hybrid'):
            with TemporaryDirectory() as directory:
                harness = TrainingHarness(directory, arm)
                result = harness.run()
                self.assertEqual(result['status'], 'complete')
                traces[arm] = harness.epoch_inputs
        baseline = traces['hybrid']
        for arm, events in traces.items():
            self.assertEqual(len(events), len(baseline))
            for actual, expected in zip(events, baseline):
                for key in ('learning_rates', 'outcomes', 'costs', 'replay'):
                    self.assertEqual(actual[key], expected[key], (arm, key))

    def test_changed_environment_source_fails_before_model_loading(self):
        with TemporaryDirectory() as directory:
            harness = TrainingHarness(directory)
            harness.raw['source_sha256']['general_lab/workflow_environment.py'] = '0' * 64
            (harness.raw_data / 'manifest.json').write_text(json.dumps(harness.raw))
            prepared = json.loads((harness.data / 'manifest.json').read_text())
            prepared['raw_manifest_sha256'] = file_hash(harness.raw_data / 'manifest.json')
            (harness.data / 'manifest.json').write_text(json.dumps(prepared))
            with patch.object(train, 'load_model') as load, patch.object(train.signal, 'signal'):
                with self.assertRaisesRegex(ValueError, 'Environment changed'):
                    train.train(harness.args)
            load.assert_not_called()

    def test_valid_freeze_pins_exact_adapter_data_code_and_protocol(self):
        with TemporaryDirectory() as directory:
            harness = TrainingHarness(directory)
            frozen = harness.freeze()
            result = harness.run()
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['freeze_sha256'], file_hash(harness.args.freeze))
            self.assertEqual(result['starting_adapter_sha256'], frozen['starting_adapter_files_sha256'])
            self.assertEqual(result['prepared_manifest_sha256'], frozen['prepared_manifest_sha256'])

    def test_frozen_artifact_tampering_fails_before_loading_any_model(self):
        for changed in ('prepared_manifest', 'adapter_weights', 'adapter_config', 'protocol'):
            with self.subTest(changed=changed), TemporaryDirectory() as directory:
                harness = TrainingHarness(directory)
                frozen = harness.freeze()
                if changed == 'prepared_manifest':
                    path = harness.data / 'manifest.json'
                    value = json.loads(path.read_text())
                    value['unrecorded_change'] = True
                    path.write_text(json.dumps(value))
                elif changed == 'adapter_weights':
                    (harness.adapter / 'adapter_model.safetensors').write_bytes(b'changed-model')
                elif changed == 'adapter_config':
                    path = harness.adapter / 'adapter_config.json'
                    value = json.loads(path.read_text())
                    value['lora_alpha'] = 999
                    path.write_text(json.dumps(value))
                else:
                    # An expected protocol hash that does not match the actual
                    # current file must fail just like an edited protocol.
                    frozen['files']['docs/outcome-v2-protocol.md'] = '0' * 64
                    harness.args.freeze.write_text(json.dumps(frozen))
                with patch.object(train, 'load_model') as load, \
                        patch('transformers.AutoTokenizer.from_pretrained') as tokenizer, \
                        patch.object(train.signal, 'signal'):
                    with self.assertRaisesRegex(ValueError, 'Frozen'):
                        train.train(harness.args)
                load.assert_not_called()
                tokenizer.assert_not_called()

    def test_cli_default_has_two_hour_bound_and_optional_freeze(self):
        argv = ['outcome_train', '--adapter', '/adapter', '--data', '/data', '--raw-data', '/raw',
                '--output', '/output', '--arm', 'hybrid', '--seed', '47']
        with patch('sys.argv', argv):
            parsed = train.arguments()
        self.assertEqual(parsed.max_hours, 2.)
        self.assertIsNone(parsed.freeze)


if __name__ == '__main__':
    unittest.main()
