import unittest

from tool_lab.contextual_shell_report import context_metrics


class ContextualReportTests(unittest.TestCase):
    def cases(self):
        rows, predictions = [], []
        for i, target in enumerate(('yes', 'no', 'no', 'yes')):
            rows.append(dict(id=str(i), bundle_id='pair', task='example/completion_forecast',
                             input=dict(question=f'Goal: {i % 2}\n\nCommand: inspect'), target=dict(option_ids=[target])))
            predictions.append(dict(id=str(i), target_ids=[target], choice='yes', probabilities={'yes': .8, 'no': .2}))
        return rows, predictions

    def test_always_yes_gets_no_complete_quartet_credit(self):
        rows, predictions = self.cases()
        result = context_metrics(rows, predictions)
        self.assertEqual(result['entire_quartets_correct'], 0)
        self.assertEqual(result['decision_quartets'], 1)
        self.assertAlmostEqual(result['binary_forecast_brier'], .34)
        for prediction in predictions:
            prediction['choice'] = prediction['target_ids'][0]
        self.assertEqual(context_metrics(rows, predictions)['entire_quartets_correct'], 1)

    def test_missing_context_cannot_inflate_consistency(self):
        rows, predictions = self.cases()
        with self.assertRaises(ValueError):
            context_metrics(rows[:-1], predictions[:-1])


if __name__ == '__main__':
    unittest.main()
