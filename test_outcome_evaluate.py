"""Decision and probability-contract tests using tiny executable fake worlds."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from general_lab import outcome_data
from general_lab.outcome_evaluate import evaluate_audits, evaluate_suite, expected_utility


@dataclass(frozen=True)
class Scenario:
    name: str = 'toy'


@dataclass(frozen=True)
class Tape:
    bit: bool
    private_marker: str = 'PRIVATE_TAPE_NOT_FOR_MODEL'


@dataclass(frozen=True)
class Observation:
    tick: int = 0
    spent_cents: int = 0
    signal: bool | None = None
    terminal: bool = False


class Episode:
    def __init__(self, scenario, tape):
        self.scenario, self.tape = scenario, tape
        self.observation, self.outcome = Observation(), None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def observe(self):
        return self.observation

    def legal_actions(self):
        return Toy.legal_actions(self.scenario, self.observation)

    def input(self):
        return item('actor', self.observation, self.legal_actions())

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError('Illegal toy action')
        obs = self.observation
        cost = {'inspect': 10, 'expensive_inspect': 90, 'act': 0, 'skip': 0}[action]
        signal, terminal, payoff = obs.signal, action in ('act', 'skip'), 0
        if terminal:
            self.outcome = 'skipped' if action == 'skip' else 'good' if self.tape.bit else 'bad'
            payoff = Toy.terminal_utilities(self.scenario)[self.outcome]
        else:
            signal = self.tape.bit
        self.observation = Observation(obs.tick + 1, obs.spent_cents + cost, signal, terminal)
        return {'observation': self.observation, 'reward_cents': payoff - cost,
                'cost_cents': cost, 'terminal': terminal}

    def truth_receipt(self):
        if not self.observation.terminal:
            raise ValueError('No early truth')
        return {'tape': asdict(self.tape), 'outcome': self.outcome,
                'success': self.outcome == 'good', 'spent_cents': self.observation.spent_cents,
                'return_cents': Toy.terminal_utilities(self.scenario)[self.outcome] - self.observation.spent_cents}


def item(kind, observation, options, action=None):
    return {'state': json.dumps({'kind': kind, 'signal': observation.signal,
                                'tick': observation.tick, 'action': action}),
            'question': 'Choose the next action.' if kind == 'actor' else 'Predict this fixed-continuation marginal.',
            'options': [{'id': key, 'description': key} for key in options]}


class Toy:
    EpisodeAdapter = Episode

    @staticmethod
    def mechanism_id(scenario):
        return scenario.name

    @staticmethod
    def legal_actions(scenario, observation):
        if observation.terminal:
            return []
        if observation.tick == 0:
            return ['inspect', 'expensive_inspect', 'act', 'skip']
        return ['act', 'skip'] if observation.signal else ['skip']

    @staticmethod
    def terminal_utilities(scenario):
        return {'good': 100, 'bad': -100, 'skipped': 0}

    @staticmethod
    def continuation_action(scenario, observation):
        return 'act' if observation.signal else 'skip'

    @staticmethod
    def forecast_inputs(scenario, observation, action):
        costs = ['0', '10', '90'] if action in ('inspect', 'expensive_inspect') else ['0']
        return {'outcome_input': item('outcome', observation, ['good', 'bad', 'skipped'], action),
                'cost_input': item('cost', observation, costs, action),
                'cost_values': {key: int(key) for key in costs}}

    @staticmethod
    def exact_forecast(scenario, observation, action):
        inputs = Toy.forecast_inputs(scenario, observation, action)
        return {'outcome_probabilities': public_prediction(inputs['outcome_input']),
                'cost_probabilities': public_prediction(inputs['cost_input'])}


def public_prediction(question, wrong_cost=False, sabotage_after_inspection=False):
    public = json.loads(question['state'])
    result = {option['id']: 0. for option in question['options']}
    action = public['action']
    if public['kind'] == 'actor':
        result['act' if 'act' in result else next(iter(result))] = 1.
    elif public['kind'] == 'cost':
        cost = {'inspect': 10, 'expensive_inspect': 90, 'act': 0, 'skip': 0}[action]
        if wrong_cost and action in ('inspect', 'expensive_inspect'):
            cost = 90 if action == 'inspect' else 0
        result[str(cost)] = 1.
    elif sabotage_after_inspection and public['tick'] > 0 and public['signal']:
        result['good' if action == 'skip' else 'bad'] = 1.
    elif action == 'skip':
        result['skipped'] = 1.
    elif action in ('inspect', 'expensive_inspect'):
        result.update(good=.5, skipped=.5)
    elif public['signal'] is None:
        result.update(good=.5, bad=.5)
    else:
        result['good' if public['signal'] else 'bad'] = 1.
    return result


def root_world(name, split, index, seed=None):
    return Toy, Scenario(name), Tape(bool(index % 2)), f'{name}-{split}-{seed}-{index}'


def audit_row(root='audit-root', identity='audit-pair'):
    inputs = Toy.forecast_inputs(Scenario(), Observation(), 'inspect')
    return {'id': identity, 'root_id': root, 'environment': 'toy', 'group_id': 'toy:mechanism',
            'split': 'validation', 'action': 'inspect', **inputs,
            'terminal_utilities': Toy.terminal_utilities(Scenario()),
            'outcome_target': 'good', 'cost_target': '10',
            **Toy.exact_forecast(Scenario(), Observation(), 'inspect')}


class OutcomeEvaluateTests(unittest.TestCase):
    def predictor(self, **kwargs):
        def callback(items):
            self.assertTrue(items)
            for question in items:
                self.assertEqual(set(question), {'state', 'question', 'options'})
                self.assertGreater(len(question['options']), 1)
                self.assertNotIn('PRIVATE_TAPE_NOT_FOR_MODEL', json.dumps(question))
                self.assertNotIn('outcome_target', json.dumps(question))
            return [public_prediction(question, **kwargs) for question in items], {'input_tokens': 11 * len(items)}
        return callback

    def evaluate(self, callback=None, **kwargs):
        with patch.object(outcome_data, 'root_world', side_effect=root_world) as worlds:
            result = evaluate_suite(callback or self.predictor(), 'validation', 2,
                                    environment_names=('toy',), **kwargs)
            self.assertEqual([call.args[3] for call in worlds.call_args_list], [outcome_data.SEEDS['validation']] * 2)
            return result

    def test_information_value_replanning_and_same_root_tapes(self):
        metrics, traces, audits = self.evaluate()
        self.assertEqual(audits, [])
        self.assertAlmostEqual(metrics['controller_reward'], .4)
        self.assertAlmostEqual(metrics['actor_reward'], 0.)
        self.assertAlmostEqual(metrics['policies']['fixed_continuation']['reward'], 0.)
        self.assertAlmostEqual(metrics['policies']['exact_controller']['reward'], .4)
        for root in traces:
            policies = root['policies']
            self.assertEqual(policies['controller']['steps'][0]['action'], 'inspect')
            candidate = policies['controller']['steps'][0]['decision']['candidates'][0]
            self.assertEqual(candidate['expected_future_return_cents'], 40.)
            for policy in policies.values():
                self.assertEqual(policy['private_outcome']['tape'], root['private_root']['tape'])
                self.assertEqual(sum(step['reward_cents'] for step in policy['steps']), policy['reward_cents'])
        usage = metrics['inference']['total']
        self.assertGreater(usage['known_questions'], 0)
        self.assertEqual(metrics['inference']['input_tokens'], usage['model_questions'] * 11)
        self.assertLess(usage['batch_calls'], usage['model_questions'])

    def test_wrong_predicted_cost_changes_action_but_actual_cost_is_charged(self):
        metrics, traces, _ = self.evaluate(self.predictor(wrong_cost=True))
        self.assertAlmostEqual(metrics['controller_reward'], -.4)
        self.assertAlmostEqual(metrics['policies']['controller']['cost'], .9)
        self.assertAlmostEqual(metrics['policies']['exact_controller']['reward'], .4)
        for root in traces:
            first = root['policies']['controller']['steps'][0]
            self.assertEqual(first['action'], 'expensive_inspect')
            self.assertEqual(first['cost_cents'], 90)
            self.assertEqual(first['decision']['chosen_expected_future_return_cents'], 50)

    def test_outcome_and_cost_marginals_preserve_asymmetric_utilities(self):
        a, b = {'zero': .1, 'one': .8, 'two': .1}, {'zero': .2, 'one': .8, 'two': 0.}
        self.assertGreater(expected_utility(b, {'0': 1}, {'zero': -100, 'one': 100, 'two': -1000}, {'0': 0}),
                           expected_utility(a, {'0': 1}, {'zero': -100, 'one': 100, 'two': -1000}, {'0': 0}))
        self.assertGreater(expected_utility(a, {'0': 1}, {'zero': -1000, 'one': 100, 'two': -100}, {'0': 0}),
                           expected_utility(b, {'0': 1}, {'zero': -1000, 'one': 100, 'two': -100}, {'0': 0}))
        # The paired outcome/cost draw can be correlated. Additive utility
        # needs the two expectations, not a fabricated independence product.
        row = audit_row()
        row['cost_probabilities'] = {'0': 0., '10': .5, '90': .5}
        row['cost_target'] = '90'
        summary, records = evaluate_audits(self.predictor(), [row], 'validation', ('toy',))
        self.assertEqual(records[0]['predicted_expected_utility_cents'], 40.)
        self.assertEqual(records[0]['exact_expected_utility_cents'], 0.)
        self.assertEqual(summary['macro']['expected_utility_mae'], .4)
        self.assertAlmostEqual(summary['macro']['marginals']['outcome']['brier'], .5)
        self.assertAlmostEqual(summary['macro']['marginals']['cost']['brier'], 2.)
        self.assertAlmostEqual(summary['macro']['marginals']['cost']['exact_distribution_mse'], 1 / 6)

    def test_fixed_continuation_audits_are_not_replanning_calibration(self):
        row = audit_row()
        metrics, traces, predictions = self.evaluate(self.predictor(sabotage_after_inspection=True), audit_rows=[row])
        self.assertEqual(metrics['policies']['controller']['success_rate'], 0.)
        self.assertEqual(metrics['audit']['macro']['marginals']['outcome']['exact_distribution_mse'], 0.)
        self.assertEqual(predictions[0]['outcome']['probabilities']['good'], .5)
        self.assertIn('not final outcomes', predictions[0]['calibration_scope'])
        self.assertIn('not end-to-end', metrics['calibration_scope'])
        self.assertIn('not a fully optimal planner', metrics['reference'])
        for root in traces:
            self.assertEqual(root['policies']['controller']['private_outcome']['outcome'], 'skipped')

    def test_audit_file_loading_and_invalid_distributions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'validation-audit.jsonl'
            path.write_text(json.dumps(audit_row()) + '\n')
            metrics, _, predictions = self.evaluate(audit_path=path)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(metrics['audit']['macro']['roots'], 1)
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            self.evaluate(lambda items: [{option['id']: .5 for option in item['options']} for item in items])
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            self.evaluate(lambda items: [{'outside_menu': 1.} for _ in items])
        with self.assertRaisesRegex(ValueError, 'split differs'):
            evaluate_audits(self.predictor(), [audit_row()], 'test', ('toy',))

    def test_audit_macro_weights_environments_then_roots(self):
        # Three rows from one root must not count as three independent worlds.
        first = [audit_row('many', f'p{i}') for i in range(3)]
        second = audit_row('single', 'p3')
        second['outcome_probabilities'] = {'good': 0., 'bad': 0., 'skipped': 1.}
        summary, _ = evaluate_audits(self.predictor(), first + [second], 'validation', ('toy',))
        self.assertEqual(summary['macro']['roots'], 2)
        self.assertAlmostEqual(summary['macro']['marginals']['outcome']['exact_distribution_mse'], 1 / 12)
        other = audit_row('other-root', 'other-pair')
        other['environment'] = 'other'
        other['outcome_probabilities'] = {'good': 0., 'bad': 1., 'skipped': 0.}
        summary, _ = evaluate_audits(self.predictor(), first + [second, other], 'validation', ('toy', 'other'))
        self.assertEqual(summary['macro']['roots'], 3)
        self.assertAlmostEqual(summary['macro']['marginals']['outcome']['exact_distribution_mse'], 7 / 24)


if __name__ == '__main__':
    unittest.main()
