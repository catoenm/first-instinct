"""Offline checks for the verified worlds and language-network PPO contract."""

from dataclasses import replace
import itertools
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from general_lab.environments import (Episode, HELDOUT_SKINS, SKINS, Sensor, action_values,
                                      legal_actions, make_episode, optimal_actions, posterior,
                                      public_input, step, terminal_reward, terminal_values,
                                      warmstart_rows)
from general_lab.rl import (DeadlineReached, LanguagePolicy, clipped_policy_loss, collect,
                            evaluate, forecast_rows, parameter_audit, snapshot, update)
from scale_lab.common import targets


class TinyTokenizer:
    """Character encoding for exercising the real collection/update code offline."""
    def apply_chat_template(self, messages, **unused):
        return json.dumps(messages)

    def encode(self, text, **unused):
        return [1 + ord(char) % 39 for char in text]


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(96, 16)
        self.transform = nn.Linear(16, 16)
        self.dropout = nn.Dropout(.5)
        self.lm_head = nn.Linear(16, 96, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, **unused):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        hidden = self.dropout(torch.tanh(self.transform(pooled)))[:, None, :]
        return SimpleNamespace(logits=self.lm_head(hidden))


class EnvironmentTests(unittest.TestCase):
    def test_posterior_matches_exhaustive_bayes(self):
        s = make_episode(19).scenario
        for mask in range(4):
            indices = [i for i in range(2) if mask & (1 << i)]
            for reports in itertools.product((False, True), repeat=len(indices)):
                observations = tuple(zip(indices, reports))
                masses = []
                for truth in (False, True):
                    weight = s.prior if truth else 1 - s.prior
                    for index, report in observations:
                        sensor = s.sensors[index]
                        positive = sensor.sensitivity if truth else 1 - sensor.specificity
                        weight *= positive if report else 1 - positive
                    masses.append(weight)
                self.assertAlmostEqual(posterior(s, observations), masses[1] / sum(masses))
        with self.assertRaises(ValueError):
            posterior(s, ((0, True), (0, True)))

    def test_planner_values_match_direct_enumeration(self):
        s = make_episode(81).scenario
        observed = ((0, True),)
        q = posterior(s, observed)
        sensor = s.sensors[1]
        result = -sensor.cost
        for truth in (False, True):
            event_probability = q if truth else 1 - q
            for report in (False, True):
                chance_positive = sensor.sensitivity if truth else 1 - sensor.specificity
                report_probability = chance_positive if report else 1 - chance_positive
                after = observed + ((1, report),)
                action = max(terminal_values(s, posterior(s, after)), key=terminal_values(s, posterior(s, after)).get)
                result += event_probability * report_probability * terminal_reward(s, action, truth)
        self.assertAlmostEqual(action_values(s, observed)["inspect_1"], result)

    def test_value_of_perfect_free_information_and_expensive_information(self):
        s = make_episode(20).scenario
        perfect = replace(s, sensors=(Sensor("perfect", 1., 1., 0.), s.sensors[1]))
        self.assertAlmostEqual(action_values(perfect)["inspect_0"], perfect.correct_reward)
        self.assertIn("inspect_0", optimal_actions(perfect))
        expensive = replace(s, sensors=tuple(replace(x, cost=100.) for x in s.sensors))
        self.assertFalse(any(x.startswith("inspect") for x in optimal_actions(expensive)))

    def test_hidden_truth_and_unobserved_reports_do_not_change_input(self):
        first = make_episode(16)
        second = Episode(first.scenario, not first.truth, tuple(not x for x in first.reports))
        self.assertEqual(public_input(first.scenario, first.observations(0)),
                         public_input(second.scenario, second.observations(0)))
        text = json.dumps(public_input(first.scenario, (), "forecast"))
        self.assertNotIn(first.scenario.identity, text)
        self.assertNotIn("posterior", text)
        self.assertNotIn('"truth"', text)
        visible = ((0, first.reports[0]),)
        for kind in ("action", "forecast"):
            self.assertEqual(public_input(first.scenario, visible, kind), public_input(second.scenario, visible, kind))

    def test_costs_and_terminal_reward_accounted_once(self):
        episode = make_episode(82)
        observed, r1, terminal = step(episode, (), "inspect_0")
        self.assertFalse(terminal)
        self.assertEqual(r1, -episode.scenario.sensors[0].cost)
        with self.assertRaises(ValueError):
            step(episode, observed, "inspect_0")
        observed, r2, terminal = step(episode, observed, "inspect_1")
        self.assertEqual(legal_actions(observed), ["affirm", "deny", "abstain"])
        _, r3, terminal = step(episode, observed, "affirm" if episode.truth else "deny")
        self.assertTrue(terminal)
        self.assertAlmostEqual(r1 + r2 + r3, episode.scenario.correct_reward - sum(s.cost for s in episode.scenario.sensors))

    def test_splits_and_warmstart_contract(self):
        normal = [make_episode(i, "train").scenario for i in range(20)]
        shifted = [make_episode(i, "shift").scenario for i in range(20)]
        heldout = [make_episode(i, "new_domain").scenario for i in range(20)]
        self.assertTrue(all(s.sensors[0].sensitivity >= .7 for s in normal))
        self.assertTrue(all(s.sensors[0].sensitivity <= .66 for s in shifted))
        self.assertTrue(all(s.sensors[0].cost < .015 or s.sensors[0].cost > 1.10 for s in shifted))
        self.assertTrue({s.domain for s in heldout}.isdisjoint({s[0] for s in SKINS}))
        train = list(warmstart_rows("train", 100, 42))
        val = list(warmstart_rows("validation", 100, 42))
        self.assertEqual(train, list(warmstart_rows("train", 100, 42)))
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in val})
        for row in train + val:
            self.assertTrue(targets(row))
            self.assertNotIn("provenance", row["input"])

    def test_training_requires_both_information_and_terminal_decisions(self):
        chosen = [optimal_actions(make_episode(i).scenario)[0] for i in range(1000)]
        self.assertEqual(set(chosen), {"affirm", "deny", "abstain", "inspect_0", "inspect_1"})
        inspection_fraction = sum(action.startswith("inspect") for action in chosen) / len(chosen)
        self.assertGreater(inspection_fraction, .25)
        self.assertLess(inspection_fraction, .75)


class LanguagePPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(89)
        self.policy = LanguagePolicy(TinyLanguage(), list(range(40, 76)), 0, "cpu")
        self.tokenizer = TinyTokenizer()
        self.check = lambda: None
        self.args = SimpleNamespace(seed=47, update_number=1, epoch_number=0, batch_size=4, clip=.2,
                                    value_weight=.5, entropy_weight=.01, forecast_weight=.5,
                                    forecast_loss="log", replay_weight=.1)

    def test_ppo_gradient_sign_clipping_and_detachment(self):
        positive = torch.tensor([0.], requires_grad=True)
        negative = torch.tensor([0.], requires_grad=True)
        loss, _ = clipped_policy_loss(positive, torch.tensor([0.]), torch.tensor([1.]))
        loss.backward(); self.assertLess(float(positive.grad), 0.)
        loss, _ = clipped_policy_loss(negative, torch.tensor([0.]), torch.tensor([-1.]))
        loss.backward(); self.assertGreater(float(negative.grad), 0.)
        saturated = torch.tensor([math.log(1.5)], requires_grad=True)
        loss, _ = clipped_policy_loss(saturated, torch.tensor([0.]), torch.tensor([1.]))
        loss.backward(); self.assertEqual(float(saturated.grad), 0.)
        with self.assertRaises(ValueError):
            clipped_policy_loss(positive, torch.tensor([0.], requires_grad=True), torch.tensor([1.]))

    def test_collected_actions_are_legal_and_returns_include_all_future_rewards(self):
        episodes = [make_episode(i) for i in range(16)]
        records, traces = collect(self.policy, self.tokenizer, episodes, 4, 10000, self.check)
        self.assertTrue(records)
        for trace in traces:
            self.assertAlmostEqual(trace["return"], sum(s["reward"] for s in trace["steps"]))
            for state in trace["steps"]:
                self.assertIn(state["action"], legal_actions(tuple(tuple(x) for x in state["observations"])))
        for index, record in enumerate(records):
            following = [r for r in records[index:] if r["episode"] == record["episode"]]
            self.assertAlmostEqual(record["return"], sum(r["reward"] for r in following))

    def test_actual_update_changes_language_network_with_pure_policy_gradient(self):
        initial = snapshot(self.policy.language)
        records, _ = collect(self.policy, self.tokenizer, [make_episode(i) for i in range(12)], 4, 10000, self.check)
        forecasts = forecast_rows(self.tokenizer, 12, 118, 10000)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=.002)
        result = update(self.policy, optimizer, records, forecasts, [], self.args, self.check, audit_policy=True)
        audit = parameter_audit(self.policy.language, initial)
        self.assertGreater(result["pure_policy_language_gradient"]["nonzero_elements"], 0)
        self.assertGreater(audit["changed_elements"], 0)
        self.assertGreater(result["forecast_loss"], 0)
        self.assertTrue(math.isfinite(result["gradient_norm"]))
        self.assertEqual(self.policy.language.dropout.p, 0.)

    def test_reward_only_update_still_changes_language_trunk(self):
        self.args.forecast_weight = 0.; self.args.value_weight = 0.; self.args.entropy_weight = 0.
        initial = snapshot(self.policy.language)
        records, _ = collect(self.policy, self.tokenizer, [make_episode(i) for i in range(12)], 4, 10000, self.check)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=.002)
        result = update(self.policy, optimizer, records, [], [], self.args, self.check, audit_policy=True)
        audit = parameter_audit(self.policy.language, initial)
        self.assertGreater(audit["changed_elements"], 0)
        self.assertGreater(result["pure_policy_language_gradient"]["nonzero_elements"], 0)

    def test_real_qwen_adapter_hook_and_checkpointing(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from general_lab.rl import gradient_audit
        config = Qwen3_5Config(
            text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                         "vocab_size": 96, "layer_types": ["full_attention", "full_attention"], "max_position_embeddings": 1024},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                           "out_hidden_size": 64, "num_position_embeddings": 64})
        language = Qwen3_5ForConditionalGeneration(config)
        language.requires_grad_(False)
        language = get_peft_model(language, LoraConfig(r=4, lora_alpha=8, lora_dropout=.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        language.enable_input_require_grads()
        policy = LanguagePolicy(language, list(range(40, 76)), 0, "cpu")
        rows = [{"input_ids": [1, 2, 3, 4, 5], "option_ids": ["inspect", "act", "abstain"], "target_indices": [0]},
                {"input_ids": [2, 4, 6], "option_ids": ["yes", "no"], "target_indices": [1]}]
        policy.eval()
        with torch.no_grad():
            old = policy(rows)[0].log_softmax(-1)[:, 0]
        policy.train()
        logits, values, _ = policy(rows)
        loss, ratios = clipped_policy_loss(logits.log_softmax(-1)[:, 0], old, torch.tensor([1., -1.]))
        loss.backward(retain_graph=True)
        audit = gradient_audit(language)
        (values - torch.tensor([.5, -.1])).square().mean().backward()
        self.assertGreater(audit["nonzero_elements"], 0)
        self.assertTrue(any("lora_B" in name for name in audit["examples"]))
        self.assertTrue(torch.allclose(ratios, torch.ones_like(ratios), atol=1e-5))
        self.assertTrue(logits[1, 2].isneginf())

    def test_common_state_forecast_audit_and_policy_tree(self):
        metrics, forecasts, traces = evaluate(self.policy, self.tokenizer, "shift", 4, 88, 4, 10000, self.check)
        self.assertEqual(len(forecasts), 16)
        self.assertEqual(len(traces), 4)
        self.assertGreater(metrics["forecast"]["brier"], 0.)
        for trace in traces:
            self.assertLessEqual(trace["expected_purchases"], 2. + 1e-6)
            self.assertGreaterEqual(trace["expected_evidence_cost"], 0.)
            self.assertGreaterEqual(trace["expected_regret_to_oracle_planner"], -1e-6)
        def stop():
            raise DeadlineReached("test")
        with self.assertRaises(DeadlineReached):
            evaluate(self.policy, self.tokenizer, "test", 1, 88, 4, 10000, stop)

    def test_exact_policy_evaluation_agrees_with_oracle(self):
        seed = 71
        states = [(), ((0, False),), ((0, True),), ((1, False),), ((1, True),)] + [
            ((0, a), (1, b)) for a in (False, True) for b in (False, True)]
        lookup = {}
        for index in range(4):
            episode = make_episode(seed * 100003 + index, "test")
            identity = episode.scenario.identity
            for mask in range(4):
                q = posterior(episode.scenario, episode.observations(mask))
                lookup[f"{identity}-mask-{mask}"] = {"yes": q, "no": 1 - q}
            for state_index, observed in enumerate(states):
                chosen = optimal_actions(episode.scenario, observed)[0]
                lookup[f"{identity}-action-state-{state_index}"] = {
                    action: float(action == chosen) for action in legal_actions(observed)}
        def oracle(unused_policy, rows, unused_batch_size, unused_check):
            return [lookup[r["id"]] for r in rows]
        with patch("general_lab.rl._probabilities", oracle):
            metrics, _, traces = evaluate(self.policy, self.tokenizer, "test", 4, seed, 4, 10000, self.check)
        self.assertAlmostEqual(metrics["policy"]["expected_regret_to_oracle_planner"], 0.)
        self.assertAlmostEqual(metrics["forecast"]["mean_squared_error_to_exact_posterior"], 0.)
        for row in traces:
            self.assertAlmostEqual(row["policy_reward"], row["oracle_planner_realized_reward"])


if __name__ == "__main__":
    unittest.main()
