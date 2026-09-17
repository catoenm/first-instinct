import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from calibration_lab.forecast_audit import ForecastExercise,exercise_update,train_one
from calibration_lab.inspection_environment import World,generate
from calibration_lab.inspection_train import Policy,ScalarNetwork,digest
from calibration_lab.probability_benchmark import describe,export,numeric_metrics,read_jsonl,records


class ForecastAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exercise_rewards_and_hidden_inputs(self):
        first=World(np.array([[.5,.7,1,.8,0,.02,1,0]],dtype=np.float32))
        second=World(first.data.copy());second.data[:,6]=0
        a,b=ForecastExercise(first),ForecastExercise(second)
        np.testing.assert_array_equal(a.observations,b.observations)
        np.testing.assert_allclose(a.score([20,10]),[1,.75])
        np.testing.assert_allclose(b.score([20,10]),[0,.75])
        with self.assertRaises(ValueError): a.score([21,0])

    def test_audit_does_not_directly_train_buy_head_or_consume_interaction_randomness(self):
        policy,critic=Policy('forecast'),ScalarNetwork()
        optimizer=torch.optim.Adam(policy.parameters(),lr=.0005)
        value_optimizer=torch.optim.Adam(critic.parameters(),lr=.001)
        generator=torch.Generator().manual_seed(19)
        before=torch.get_rng_state().clone()
        weights=policy.buy.weight.detach().clone()
        world=generate(np.random.default_rng(7),32)
        with patch.object(World,'posterior',side_effect=AssertionError('Leak')):
            exercise_update(policy,critic,optimizer,value_optimizer,ForecastExercise(world),generator,.01)
        self.assertTrue(torch.equal(torch.get_rng_state(),before))
        self.assertTrue(torch.equal(policy.buy.weight,weights))
        self.assertIsNone(policy.buy.weight.grad)

    def test_schedules_match_observations_and_checkpoint_rule_without_posteriors(self):
        validation=generate(np.random.default_rng(9),32)
        with tempfile.TemporaryDirectory() as tmp,patch.object(World,'posterior',side_effect=AssertionError('Leak')):
            root=Path(tmp)
            for recipe in ('terminal_only','audit_early','audit_continuous','supervised'):
                info=train_one(root,recipe,101,validation,rollouts=4,batch_size=16,audit_worlds=4,every=4)
                self.assertEqual(info['selected_rollout'],4)
                self.assertEqual(info['counts']['audit_states'],0 if recipe=='terminal_only' else 32)
            logs=[read_jsonl(root/f'{recipe}-s101'/'audits.jsonl')
                  for recipe in ('audit_early','audit_continuous','supervised')]
            for key in ('world_sha256','observation_sha256'):
                self.assertEqual([r[key] for r in logs[0]],[r[key] for r in logs[1]])
                self.assertEqual([r[key] for r in logs[1]],[r[key] for r in logs[2]])
            self.assertEqual([r['rollout'] for r in logs[0]],[1,1,1,1])
            self.assertEqual([r['rollout'] for r in logs[1]],[1,2,3,4])
            names=['terminal_only','audit_early','audit_continuous']
            self.assertEqual(len({digest(root/f'{n}-s101'/'initial.safetensors') for n in names}),1)
            streams=[read_jsonl(root/f'{n}-s101'/'interaction.jsonl') for n in names]
            self.assertEqual(len({tuple(r['world_sha256'] for r in log) for log in streams}),1)

    def test_ground_truth_matches_enumeration_and_invariances(self):
        items=records(13,8)
        for item in items:
            p,r,s,q,copy,price,revealed,reading=item['observation']
            masses=[]
            for y in (0,1):
                mass=(p if y else 1-p)*(r if y==s else 1-r)
                if revealed and not copy:
                    mass *= q if y==reading else 1-q
                masses.append(mass)
            self.assertAlmostEqual(item['expected_probability'],masses[1]/sum(masses),places=12)
        result=numeric_metrics(items,[r['expected_probability'] for r in items])
        for values in result.values():
            for value in values.values(): self.assertAlmostEqual(value,0.,places=12)

    def test_public_text_has_no_answer_fields_and_numeric_parameters_round_trip(self):
        items=records(14,2)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'benchmark';export(folder,items)
            public=read_jsonl(folder/'requests.jsonl.gz');answers=read_jsonl(folder/'answers.jsonl.gz')
            self.assertEqual(len(public),48)
            self.assertEqual({r['id'] for r in public},{r['id'] for r in answers})
            for row in public:
                self.assertNotIn('expected_probability',row)
                self.assertNotIn('optimal_inspection',row)
            for item in items:
                record=json.loads(describe(item['observation'],1).split('Case record:\n')[1].split('\n')[0])
                self.assertEqual(np.float32(record['prior_on']),np.float32(item['observation'][0]))


if __name__=='__main__':
    unittest.main()
