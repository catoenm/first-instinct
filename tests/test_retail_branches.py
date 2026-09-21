import copy
import unittest

from test_retail_live import make_episode
from tool_lab.retail_branches import assemble_questions, prefix_observation, identities
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import DESTINATION, conditions
from tool_lab.retail_history import question


def fixtures():
    episode, _ = make_episode('000')
    root = dict(id='test', task='profile_address', observation=episode.observation())
    artifacts = []
    for condition in conditions('profile_address'):
        for action in root['observation']['menu']:
            for replica in (0, 1):
                compatible = condition.startswith('0')
                artifacts.append(dict(slot=f'{condition}-{action["id"]}-{replica}',
                    identity=dict(root='test', task='profile_address', condition=condition,
                                  action=action['id'], replica=replica), compatible=compatible,
                    success=condition.endswith('1') if compatible else None))
    return [root], artifacts


class RetailBranchTests(unittest.TestCase):
    def test_conditioned_prior_retains_uncertainty_and_excludes_replay_mass(self):
        roots, artifacts = fixtures()
        for artifact in artifacts:
            if artifact['identity']['replica']:
                artifact['success'] = True  # Aggregation ignores replay labels; execution audit must compare them.
        rows = assemble_questions(roots, artifacts)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row['soft_target'] == [.5, .5] for row in rows))
        self.assertTrue(all(row['posterior_members'] == 4 and row['original_prior_members'] == 8 for row in rows))

    def test_missing_duplicate_and_incompatible_labels_are_rejected(self):
        roots, original = fixtures()
        for mode in ('missing', 'duplicate', 'label'):
            artifacts = copy.deepcopy(original)
            if mode == 'missing': artifacts.pop(0)
            if mode == 'duplicate': artifacts.append(artifacts[0])
            if mode == 'label':
                next(a for a in artifacts if not a['compatible'] and a['identity']['replica'] == 0)['success'] = False
            with self.assertRaises(ValueError): assemble_questions(roots, artifacts)

    def test_posterior_cannot_depend_on_future_action(self):
        roots, artifacts = fixtures()
        changed = next(a for a in artifacts if a['compatible'] and a['identity']['replica'] == 0)
        changed['compatible'] = False; changed['success'] = None
        with self.assertRaises(ValueError): assemble_questions(roots, artifacts)

    def test_world_membership_is_not_collapsed_when_current_states_converge(self):
        roots, artifacts = fixtures()
        for artifact in artifacts:
            if artifact['compatible']:
                artifact['same_current_state_hash'] = 'converged'
                artifact['success'] = artifact['identity']['condition'] == '000'
        rows = assemble_questions(roots, artifacts)
        self.assertTrue(all(row['soft_target'] == [.75, .25] for row in rows))
        self.assertTrue(all(row['posterior_members'] == 4 for row in rows))

    def test_preservation_uses_original_state_not_state_after_a_wrong_write(self):
        episode, initial = make_episode('000')
        wrong = copy.deepcopy(initial)
        wrong['orders']['#SYN1']['address'] = copy.deepcopy(DESTINATION)
        final = copy.deepcopy(wrong)
        final['users']['synthetic']['address'] = copy.deepcopy(DESTINATION)
        self.assertFalse(verify(initial, final, 'profile_address')['success'])
        self.assertTrue(verify(wrong, final, 'profile_address')['success'])

    def test_five_turn_prefix_leaves_one_action_and_preserves_all_history(self):
        episode, _ = make_episode('100')
        for _ in range(6): episode.step('read_user')
        receipt = episode.private_record()
        obs = prefix_observation(receipt, 5)
        self.assertEqual(obs['turns_remaining'], 1)
        self.assertEqual(len(obs['history']), 5)
        self.assertIn('stop immediately', question(obs, 'write_profile')['question'])
        with self.assertRaises(ValueError): prefix_observation(receipt, 6)

    def test_identity_coverage_includes_every_original_world_and_alternative(self):
        roots, _ = fixtures()
        planned = list(identities(roots))
        self.assertEqual(len(planned), 8*5*2)
        self.assertEqual(len({tuple(i.values()) for i in planned}), len(planned))


if __name__ == '__main__':
    unittest.main()
